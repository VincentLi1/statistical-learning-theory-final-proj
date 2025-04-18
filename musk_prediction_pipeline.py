# musk_prediction_pipeline.py – v6
"""Forecast next‑day PMF of Elon‑Musk tweet counts.

Models included
---------------
* **Linear Regression** (plain logits → softmax)
* **Polynomial‑kernel Ridge Regression** (`--model poly`, default degree 3)
* **ARIMA → Poisson PMF** (`--model arima`)
* **LSTM** sequence model (`--model lstm`)
* **Tiny Transformer** sequence model (`--model trans`)

`--model all` runs every one of the above.

All models are evaluated with **cross‑entropy** (negative log‑likelihood).
Progress bars appear every 5 epochs for sequence models.
"""

import argparse, warnings, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from tqdm import tqdm
import nltk, torch, torch.nn as nn
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

BIN_WIDTH = 25
MAX_BIN   = 1000
N_BINS    = MAX_BIN // BIN_WIDTH + 1
sia = SentimentIntensityAnalyzer()


def onehot_count(c: int) -> np.ndarray:
    idx = min(c // BIN_WIDTH, N_BINS - 1)
    v   = np.zeros(N_BINS, dtype=np.float32)
    v[idx] = 1.0
    return v


def add_lags(df, col: str, lags=(1, 2, 3)):
    for l in lags:
        df[f"{col}_lag{l}"] = df[col].shift(l)

# ---------------- tweet & polymarket loaders ----------------

def load_tweet_daily(csv: Path, days: int) -> pd.DataFrame:
    t = pd.read_csv(csv, parse_dates=["createdAt"], low_memory=False)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    t = t[t["createdAt"] >= cutoff]
    t["date"] = (
        t["createdAt"]
         .dt.tz_convert("UTC")
         .dt.tz_localize(None)
         .dt.floor("D")
    )
    daily = (
        t.groupby("date")
         .agg(tweet_count=("id",   "count"),
              text_concat=("fullText", " ".join))
         .sort_index()
    )
    daily["vader_mean"] = daily["text_concat"].apply(
        lambda txt: sia.polarity_scores(str(txt))["compound"]
    )
    daily = daily.drop(columns=["text_concat"])
    for c in ("tweet_count", "vader_mean"):
        add_lags(daily, c)
    return daily


def load_poly_daily(csv: Path, days: int) -> pd.DataFrame:
    if not csv.exists():
        warnings.warn("Polymarket CSV missing – skipping")
        return pd.DataFrame()

    # detect timestamp column
    with open(csv) as f:
        header = f.readline().strip().split(",")
    ts_col = next(c for c in header if "timestamp" in c.lower()).strip('"')

    p = pd.read_csv(csv)
    p[ts_col] = (
        pd.to_datetime(p[ts_col], errors="coerce")
          .dt.tz_localize(None)  # make tz-naïve
    )
    cutoff = (pd.Timestamp.utcnow() - pd.Timedelta(days=days)).tz_localize(None)
    p = p[p[ts_col] >= cutoff]

    # compute expected tweet count per minute
    prob_cols = [c for c in p.columns if any(ch in c for ch in "-+<–")]
    def mid(col: str) -> float:
        col = col.strip()
        if col.endswith("+"):
            return float(col[:-1]) + BIN_WIDTH / 2
        if col.startswith("<"):
            return float(col[1:]) / 2
        lo, hi = map(int, col.replace('–', '-').split('-'))
        return (lo + hi) / 2

    mids = {c: mid(c) for c in prob_cols}
    p["exp"] = p[prob_cols].fillna(0).mul(pd.Series(mids)).sum(axis=1)
    p["date"] = p[ts_col].dt.floor("D")

    daily = p.groupby("date").agg(poly_exp_mean=("exp", "mean"))
    add_lags(daily, "poly_exp_mean")
    return daily

# ---------------- dataset builder ----------------

def build_dataset(tweet_csv: Path, poly_csv: Path, days: int):
    td = load_tweet_daily(tweet_csv, days)
    pd = load_poly_daily(poly_csv, days)
    df = td.join(pd, how="left").fillna(0)
    df["target"] = df["tweet_count"].shift(-1)
    df = df.dropna()

    y = np.vstack(df["target"].astype(int).apply(onehot_count))
    X = df.drop(columns=["target"]).values.astype(np.float32)
    return X, y

# ---------------- models ----------------

def softmax_np(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def linreg_model(X, y):
    from sklearn.linear_model import LinearRegression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, shuffle=False)
    m = LinearRegression().fit(Xtr, np.log(ytr + 1e-12))
    ce = log_loss(yte, softmax_np(m.predict(Xte)))
    print(f"Linear Regression cross entropy: {ce:.4f}")
    return m

# ----- Kernelized (Polynomial) Ridge Regression -----

def poly_model(X, y, degree: int = 3):
    from sklearn.kernel_ridge import KernelRidge
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, shuffle=False)
    model = KernelRidge(kernel="polynomial", degree=degree, alpha=1.0)
    model.fit(Xtr, np.log(ytr + 1e-12))
    ce = log_loss(yte, softmax_np(model.predict(Xte)))
    print(f"Polynomial Kernel Ridge (degree={degree}) cross entropy: {ce:.4f}")
    return model

# ----- ARIMA → Poisson PMF -----

def arima_model(tweet_csv: Path, days: int):
    """Fit ARIMA on daily tweet counts and convert mean forecast to a Poisson PMF."""
    import scipy.stats as st
    from statsmodels.tsa.arima.model import ARIMA

    # ... [omitted for brevity, unchanged code] ...
    ce = log_loss(y_true, y_pred, labels=list(range(N_BINS)))
    print(f"ARIMA→Poisson cross entropy: {ce:.4f}")
    return model

# ----- Sequence models -----

class LSTMHead(nn.Module):
    def __init__(self,in_dim,hid,seq):
        super().__init__(); self.seq=seq
        self.lstm=nn.LSTM(in_dim,hid,batch_first=True); self.fc=nn.Linear(hid*seq,N_BINS)
    def forward(self,x): o,_=self.lstm(x); o=o.reshape(o.size(0),-1); return self.fc(o)

class TinyTrans(nn.Module):
    def __init__(self,in_dim,d_model=32):
        super().__init__()
        self.inp=nn.Linear(in_dim,d_model)
        self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model,4,64,batch_first=True),2)
        self.fc=nn.Linear(d_model,N_BINS)
    def forward(self,x): z=self.enc(self.inp(x)); return self.fc(z[:,-1])

def train_seq(model_cls,X,y,epochs,bs=32):
    SEQ=7; device='cuda' if torch.cuda.is_available() else 'cpu'
    seqs=np.stack([X[i-SEQ:i] for i in range(SEQ,len(X))]).astype(np.float32)
    tars=y[SEQ:].astype(np.float32)
    ds=torch.utils.data.TensorDataset(torch.from_numpy(seqs),torch.from_numpy(tars))
    loader=torch.utils.data.DataLoader(ds,batch_size=bs,shuffle=False)
    net=model_cls.to(device); opt=torch.optim.Adam(net.parameters(),1e-3); ce_fn=nn.CrossEntropyLoss()
    from math import ceil
    steps_per_epoch = ceil(len(loader))
    for ep in range(1, epochs + 1):
        net.train(); running = 0; n = 0
        loop = tqdm(loader, leave=False, disable=(ep % 5 != 0))
        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = ce_fn(net(xb), yb.argmax(1))
            loss.backward(); opt.step()
            running += loss.item() * xb.size(0); n += xb.size(0)
            if ep % 5 == 0:
                loop.set_description(f"epoch {ep:03d}/{epochs}")
        if ep % 5 == 0:
            print(f"epoch {ep:03d}/{epochs} cross entropy: {running / n:.4f}")
    return net


# ---------------- CLI ----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tweet_csv", default="all_musk_posts.csv")
    ap.add_argument("--poly_csv",
                    default="./polymarket_data_csvs/polymarket-price-recent-day-minute-by-minute.csv",
                    help="Path to Polymarket minute‑level CSV")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--model", choices=["linreg","poly","arima","lstm","trans","all"],
                    default="linreg")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    X, y = build_dataset(Path(args.tweet_csv), Path(args.poly_csv), args.days)
    print(f"Feature matrix {X.shape}, labels {y.shape}")

    def run_all():
        linreg_model(X, y)
        poly_model(X, y)
        arima_model(Path(args.tweet_csv), args.days)
        train_seq(LSTMHead(X.shape[1], 64, 7), X, y, args.epochs)
        train_seq(TinyTrans(X.shape[1]), X, y, args.epochs)

    if args.model == "linreg":
        linreg_model(X, y)
    elif args.model == "poly":
        poly_model(X, y)
    elif args.model == "arima":
        arima_model(Path(args.tweet_csv), args.days)
    elif args.model == "lstm":
        train_seq(LSTMHead(X.shape[1], 64, 7), X, y, args.epochs)
    elif args.model == "trans":
        train_seq(TinyTrans(X.shape[1]), X, y, args.epochs)
    else:
        run_all()
