# musk_prediction_pipeline.py – v6
"""Forecast next‑day PMF of Elon‑Musk tweet counts.

Models included
---------------
* **Linear Regression** (plain logits → softmax)
* **Polynomial‑kernel Ridge Regression** (`--model poly`, default degree 3)
* **ARIMA → Poisson PMF** (`--model arima`)
* **LSTM** sequence model (`--model lstm`)
* **Tiny Transformer** sequence model (`--model trans`)

`--model all` runs every one of the above.

All models are evaluated with **cross‑entropy** (negative log‑likelihood).
Progress bars appear every 5 epochs for sequence models.
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


def save_predictions(y_pred, model_name):
    """Save model predictions to CSV"""
    bins = [f"{i*BIN_WIDTH}-{(i+1)*BIN_WIDTH}" for i in range(N_BINS-1)]
    bins.append(f"{MAX_BIN}+")
    df = pd.DataFrame(y_pred, columns=bins)
    df.to_csv(f"{model_name}_pmf_predictions.csv", index=False)
    print(f"Saved predictions to {model_name}_pmf_predictions.csv")

def linreg_model(X, y):
    from sklearn.linear_model import LinearRegression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, shuffle=False)
    m = LinearRegression().fit(Xtr, np.log(ytr + 1e-12))
    preds = softmax_np(m.predict(Xte))
    ce = log_loss(yte, preds)
    print(f"Linear Regression cross entropy: {ce:.4f}")
    save_predictions(preds, "lin_reg")
    return m

# ----- Kernelized (Polynomial) Ridge Regression -----

def poly_model(X, y, degree: int = 3):
    from sklearn.kernel_ridge import KernelRidge
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, shuffle=False)
    model = KernelRidge(kernel="polynomial", degree=degree, alpha=1.0)
    model.fit(Xtr, np.log(ytr + 1e-12))
    preds = softmax_np(model.predict(Xte))
    ce = log_loss(yte, preds)
    print(f"Polynomial Kernel Ridge (degree={degree}) cross entropy: {ce:.4f}")
    save_predictions(preds, "poly_ridge")
    return model

# ----- ARIMA → Poisson PMF -----

def arima_model(tweet_csv: Path, days: int):
    """Fit ARIMA on daily tweet counts and convert mean forecast to a Poisson PMF."""
    import scipy.stats as st
    from statsmodels.tsa.arima.model import ARIMA

    # Load and prepare data
    t = pd.read_csv(tweet_csv, parse_dates=["createdAt"], low_memory=False)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    t = t[t["createdAt"] >= cutoff]
    t["date"] = t["createdAt"].dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("D")
    daily = t.groupby("date").size().rename("count")
    
    # Split into train/test
    train_size = int(len(daily) * 0.8)
    train, test = daily[:train_size], daily[train_size:]
    
    # Fit ARIMA model
    model = ARIMA(train, order=(7,1,0))  # Using 7-day seasonality
    model_fit = model.fit()
    
    # Make predictions
    forecast = model_fit.forecast(steps=len(test))
    
    # Convert to PMF
    y_true = np.array([onehot_count(c) for c in test])
    y_pred = np.zeros((len(test), N_BINS))
    for i, mu in enumerate(forecast):
        # Create Poisson PMF
        x = np.arange(N_BINS) * BIN_WIDTH
        pmf = st.poisson.pmf(x, mu)
        y_pred[i] = pmf / pmf.sum()  # Normalize to sum to 1
    
    # Save predictions
    save_predictions(y_pred, "arima")
    
    # Compute cross-entropy manually since y_true is one-hot
    ce = -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))
    print(f"ARIMA→Poisson cross entropy: {ce:.4f}")
    return model_fit

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

def train_seq(model_cls, X, y, epochs, bs=32, model_name="seq"):
    SEQ=7; device='cuda' if torch.cuda.is_available() else 'cpu'
    seqs=np.stack([X[i-SEQ:i] for i in range(SEQ,len(X))]).astype(np.float32)
    tars=y[SEQ:].astype(np.float32)
    ds=torch.utils.data.TensorDataset(torch.from_numpy(seqs),torch.from_numpy(tars))
    loader=torch.utils.data.DataLoader(ds,batch_size=bs,shuffle=False)
    net=model_cls.to(device); opt=torch.optim.Adam(net.parameters(),1e-3); ce_fn=nn.CrossEntropyLoss()
    
    # Training loop
    for ep in range(1, epochs + 1):
        net.train(); running = 0; n = 0
        loop = tqdm(loader, leave=False, disable=(ep % 5 != 0))
        all_preds = []
        for xb, yb in loop:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = net(xb)
            loss = ce_fn(logits, yb.argmax(1))
            loss.backward(); opt.step()
            running += loss.item() * xb.size(0); n += xb.size(0)
            
            # Save predictions from last epoch
            if ep == epochs:
                probs = torch.softmax(logits, dim=1)
                all_preds.append(probs.detach().cpu().numpy())
                
            if ep % 5 == 0:
                loop.set_description(f"epoch {ep:03d}/{epochs}")
        if ep % 5 == 0:
            print(f"epoch {ep:03d}/{epochs} cross entropy: {running / n:.4f}")
            
    # Save final predictions
    if all_preds:
        final_preds = np.vstack(all_preds)
        save_predictions(final_preds, model_name)
    
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
        train_seq(LSTMHead(X.shape[1], 64, 7), X, y, args.epochs, model_name="lstm")
        train_seq(TinyTrans(X.shape[1]), X, y, args.epochs, model_name="transformer")

    if args.model == "linreg":
        linreg_model(X, y)
    elif args.model == "poly":
        poly_model(X, y)
    elif args.model == "arima":
        arima_model(Path(args.tweet_csv), args.days)
    elif args.model == "lstm":
        train_seq(LSTMHead(X.shape[1], 64, 7), X, y, args.epochs, model_name="lstm")
    elif args.model == "trans":
        train_seq(TinyTrans(X.shape[1]), X, y, args.epochs, model_name="transformer")
    else:
        run_all()
