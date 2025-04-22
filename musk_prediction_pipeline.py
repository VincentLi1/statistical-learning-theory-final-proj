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
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from tqdm import tqdm
import nltk, torch, torch.nn as nn
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

# Open output file for writing results
output_file = open('model_results.txt', 'w', encoding='utf-8')

def log_result(message):
    """Write message to both console and file"""
    print(message)
    output_file.write(message + '\n')
    output_file.flush()

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

def get_bin_ranges():
    """Get human-readable bin ranges for output"""
    bins = [f"{i*BIN_WIDTH}-{(i+1)*BIN_WIDTH}" for i in range(N_BINS-1)]
    bins.append(f"{MAX_BIN}+")
    return bins

def get_week_progress(end_date=None):
    try:
        # Try to read the minute-by-minute data
        df = pd.read_csv('./polymarket_data_csvs/polymarket-price-recent-day-minute-by-minute.csv')
        df['Date (UTC)'] = pd.to_datetime(df['Date (UTC)'], format='%m-%d-%Y %H:%M')
        
        # Get the last timestamp from the data
        last_data_date = df['Date (UTC)'].max()
        
        if end_date:
            target_date = pd.to_datetime(end_date)
        else:
            target_date = pd.Timestamp.now()
        
        # Calculate days remaining based on the difference between target date and last data date
        days_remaining = (target_date - last_data_date).total_seconds() / (24 * 60 * 60)
        days_remaining = max(0, days_remaining)  # Ensure non-negative
        
        # Calculate week progress
        week_progress = max(0, min(1, 1 - (days_remaining / 7)))  # Cap between 0 and 1
        
        return week_progress, days_remaining
        
    except Exception as e:
        print(f"Warning: No data for current week – using default time-based calculation ({str(e)})")
        # Fallback to time-based calculation
        if end_date:
            target_date = pd.to_datetime(end_date)
        else:
            target_date = pd.Timestamp.now()
            
        start_of_week = target_date - pd.Timedelta(days=target_date.weekday())
        week_progress = (target_date - start_of_week).total_seconds() / (7 * 24 * 60 * 60)
        days_remaining = 7 - (week_progress * 7)
        return week_progress, days_remaining

def get_current_week_tweets(tweet_csv: Path, current_count: int = None) -> int:
    """Get the current week's tweet count up to today"""
    if current_count is not None:
        return current_count
        
    t = pd.read_csv(tweet_csv, parse_dates=["createdAt"], low_memory=False)
    # Convert all times to UTC and make timezone-naive
    t["date"] = pd.to_datetime(t["createdAt"]).dt.tz_convert("UTC").dt.tz_localize(None).dt.floor("D")
    today = pd.Timestamp.utcnow().tz_localize(None).floor("D")
    week_start = today - pd.Timedelta(days=today.dayofweek)
    current_week = t[t["date"] >= week_start]
    return len(current_week)

def save_model_predictions(y_pred, model_name: str):
    """Save model predictions to CSV"""
    bins = get_bin_ranges()
    df = pd.DataFrame(y_pred, columns=bins)
    df.to_csv(f"{model_name}_pmf_predictions.csv", index=False)
    log_result(f"Saved predictions to {model_name}_pmf_predictions.csv")

def save_predictions(probs_dict, prediction_type: str, end_date: pd.Timestamp = None, current_week_tweets: int = None):
    """Save predictions from all models to results file"""
    log_result(f"\n{prediction_type} predictions:")
    log_result("-" * 40)
    
    # Add week progress info for weekly predictions
    if "week" in prediction_type.lower():
        week_progress, days_remaining = get_week_progress(end_date)
        log_result(f"Current week progress: {week_progress:.1%}")
        log_result(f"Remaining days: {days_remaining:.1f}")
        if current_week_tweets is not None:
            log_result(f"Current week tweets: {current_week_tweets}")
        log_result("-" * 40)
    
    bins = get_bin_ranges()
    
    # Find most likely bin for each model
    for model_name, probs in probs_dict.items():
        max_bin = bins[np.argmax(probs)]
        prob = probs.max()
        log_result(f"{model_name:12} Most likely: {max_bin:10} (p={prob:.3f})")
        
        # Show top 3 most likely bins
        top3_idx = np.argsort(probs)[-3:][::-1]
        log_result(f"{'':12} Top 3 bins:")
        for idx in top3_idx:
            log_result(f"{'':12} {bins[idx]:10} p={probs[idx]:.3f}")
        log_result("")

def predict_tomorrow(model, X, model_type: str):
    """Generate prediction for tomorrow using the most recent data"""
    import scipy.stats as st
    
    if model_type == "seq":
        # For sequence models (LSTM/Transformer), use last 7 days
        x = torch.from_numpy(X[-7:].astype(np.float32)).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    elif model_type == "arima":
        # For ARIMA, use the forecast directly
        forecast = model.forecast(steps=1)[0]
        x = np.arange(N_BINS) * BIN_WIDTH
        probs = st.poisson.pmf(x, forecast)
        probs = probs / probs.sum()  # Normalize
    else:
        # For other models (Linear/Poly), use most recent features
        x = X[-1:] 
        if model_type == "linear":
            logits = model.predict(x)
        else:  # polynomial
            logits = model.predict(x)
        probs = softmax_np(logits)[0]
    
    return probs

def predict_week_total(model, X, model_type: str, current_week_tweets: int, end_date: pd.Timestamp = None):
    """Generate prediction for end of week total tweet count"""
    import scipy.stats as st
    
    week_progress, days_remaining = get_week_progress(end_date)
    
    if model_type == "seq":
        # For sequence models, predict next remaining days and sum
        x = torch.from_numpy(X[-7:].astype(np.float32)).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            # Convert daily PMF to weekly total PMF using convolution
            daily_probs = probs
            weekly_probs = daily_probs.copy()
            for _ in range(int(days_remaining) - 1):
                weekly_probs = np.convolve(weekly_probs, daily_probs, mode='full')[:N_BINS]
            weekly_probs = weekly_probs / weekly_probs.sum()
            # Shift by current week tweets
            shift_idx = min(int(current_week_tweets / BIN_WIDTH), N_BINS - 1)
            weekly_probs = np.roll(weekly_probs, shift_idx)
            weekly_probs[:shift_idx] = 0
            weekly_probs = weekly_probs / weekly_probs.sum()
            return weekly_probs
    elif model_type == "arima":
        # For ARIMA, forecast remaining days and add to current count
        forecast = model.forecast(steps=int(days_remaining)).sum()
        expected_weekly = current_week_tweets + forecast
        # Use normal distribution for weekly total
        x = np.arange(N_BINS) * BIN_WIDTH
        probs = st.norm.pdf(x, expected_weekly, np.sqrt(expected_weekly))  # Using sqrt(expected) as std dev
        probs = probs / probs.sum()  # Normalize
        return probs
    else:
        # For other models, predict daily and use convolution for weekly
        x = X[-1:] 
        if model_type == "linear":
            logits = model.predict(x)
        else:  # polynomial
            logits = model.predict(x)
        daily_probs = softmax_np(logits)[0]
        # Convert daily PMF to weekly total PMF using convolution
        weekly_probs = daily_probs.copy()
        for _ in range(int(days_remaining) - 1):
            weekly_probs = np.convolve(weekly_probs, daily_probs, mode='full')[:N_BINS]
        weekly_probs = weekly_probs / weekly_probs.sum()
        # Shift by current week tweets
        shift_idx = min(int(current_week_tweets / BIN_WIDTH), N_BINS - 1)
        weekly_probs = np.roll(weekly_probs, shift_idx)
        weekly_probs[:shift_idx] = 0
        weekly_probs = weekly_probs / weekly_probs.sum()
        return weekly_probs

def save_tomorrow_predictions(probs_dict):
    """Save tomorrow's predictions from all models to results file"""
    log_result("\nPredictions for tomorrow's tweet counts:")
    log_result("-" * 40)
    
    bins = get_bin_ranges()
    
    # Find most likely bin for each model
    for model_name, probs in probs_dict.items():
        max_bin = bins[np.argmax(probs)]
        prob = probs.max()
        log_result(f"{model_name:12} Most likely: {max_bin:10} (p={prob:.3f})")
        
        # Show top 3 most likely bins
        top3_idx = np.argsort(probs)[-3:][::-1]
        log_result(f"{'':12} Top 3 bins:")
        for idx in top3_idx:
            log_result(f"{'':12} {bins[idx]:10} p={probs[idx]:.3f}")
        log_result("")

def linreg_model(X, y):
    from sklearn.linear_model import LinearRegression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, shuffle=False)
    m = LinearRegression().fit(Xtr, np.log(ytr + 1e-12))
    preds = softmax_np(m.predict(Xte))
    ce = log_loss(yte, preds)
    log_result(f"Linear Regression cross entropy: {ce:.4f}")
    save_model_predictions(preds, "lin_reg")
    
    # Get tomorrow's prediction
    tmr_pred = predict_tomorrow(m, X, "linear")
    return m, tmr_pred

def poly_model(X, y, degree: int = 3):
    from sklearn.kernel_ridge import KernelRidge
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, shuffle=False)
    model = KernelRidge(kernel="polynomial", degree=degree, alpha=1.0)
    model.fit(Xtr, np.log(ytr + 1e-12))
    preds = softmax_np(model.predict(Xte))
    ce = log_loss(yte, preds)
    log_result(f"Polynomial Kernel Ridge (degree={degree}) cross entropy: {ce:.4f}")
    save_model_predictions(preds, "poly_ridge")
    
    # Get tomorrow's prediction
    tmr_pred = predict_tomorrow(model, X, "poly")
    return model, tmr_pred

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
    save_model_predictions(y_pred, "arima")
    
    # Compute cross-entropy manually since y_true is one-hot
    ce = -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))
    log_result(f"ARIMA->Poisson cross entropy: {ce:.4f}")
    
    # Get tomorrow's prediction
    tmr_pred = predict_tomorrow(model_fit, X, "arima")
    return model_fit, tmr_pred

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
            log_result(f"epoch {ep:03d}/{epochs} cross entropy: {running / n:.4f}")
            
    # Save final predictions
    if all_preds:
        final_preds = np.vstack(all_preds)
        save_model_predictions(final_preds, model_name)
    
    # Get tomorrow's prediction
    tmr_pred = predict_tomorrow(net, X, "seq")
    return net, tmr_pred


# ---------------- CLI ----------------

if __name__ == "__main__":
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--tweet_csv", default="all_musk_posts.csv")
        ap.add_argument("--poly_csv",
                        default="./polymarket_data_csvs/polymarket-price-recent-day-minute-by-minute.csv",
                        help="Path to Polymarket minute‑level CSV")
        ap.add_argument("--days", type=int, default=365)
        ap.add_argument("--model", choices=["linreg","poly","arima","lstm","trans","all"],
                        default="linreg")
        ap.add_argument("--epochs", type=int, default=30)
        ap.add_argument("--end_date", type=str, default=None,
                        help="End date for predictions in YYYY-MM-DD format")
        ap.add_argument("--current_tweets", type=int, default=None,
                        help="Current week's tweet count (overrides reading from CSV)")
        args = ap.parse_args()

        # Parse end date if provided
        end_date = None
        if args.end_date:
            try:
                end_date = pd.Timestamp(args.end_date)
                log_result(f"Using end date: {end_date.strftime('%Y-%m-%d')}")
            except ValueError:
                print("Error: Invalid date format. Please use YYYY-MM-DD")
                sys.exit(1)

        X, y = build_dataset(Path(args.tweet_csv), Path(args.poly_csv), args.days)
        log_result(f"Feature matrix {X.shape}, labels {y.shape}")

        tomorrow_preds = {}
        week_total_preds = {}
        current_week_tweets = get_current_week_tweets(Path(args.tweet_csv), args.current_tweets)
        if args.current_tweets is not None:
            log_result(f"Using provided current week tweet count: {current_week_tweets}")

        def run_all():
            # Train models and get predictions
            m1, tomorrow_preds["Linear   "] = linreg_model(X, y)
            m2, tomorrow_preds["Poly     "] = poly_model(X, y)
            m3, tomorrow_preds["ARIMA    "] = arima_model(Path(args.tweet_csv), args.days)
            m4, tomorrow_preds["LSTM     "] = train_seq(LSTMHead(X.shape[1], 64, 7), X, y, args.epochs, model_name="lstm")
            m5, tomorrow_preds["Transform"] = train_seq(TinyTrans(X.shape[1]), X, y, args.epochs, model_name="transformer")
            
            # Get week total predictions
            week_total_preds["Linear   "] = predict_week_total(m1, X, "linear", current_week_tweets, end_date)
            week_total_preds["Poly     "] = predict_week_total(m2, X, "poly", current_week_tweets, end_date)
            week_total_preds["ARIMA    "] = predict_week_total(m3, X, "arima", current_week_tweets, end_date)
            week_total_preds["LSTM     "] = predict_week_total(m4, X, "seq", current_week_tweets, end_date)
            week_total_preds["Transform"] = predict_week_total(m5, X, "seq", current_week_tweets, end_date)

        if args.model == "linreg":
            m, tmr = linreg_model(X, y)
            tomorrow_preds["Linear   "] = tmr
            week_total_preds["Linear   "] = predict_week_total(m, X, "linear", current_week_tweets, end_date)
        elif args.model == "poly":
            m, tmr = poly_model(X, y)
            tomorrow_preds["Poly     "] = tmr
            week_total_preds["Poly     "] = predict_week_total(m, X, "poly", current_week_tweets, end_date)
        elif args.model == "arima":
            m, tmr = arima_model(Path(args.tweet_csv), args.days)
            tomorrow_preds["ARIMA    "] = tmr
            week_total_preds["ARIMA    "] = predict_week_total(m, X, "arima", current_week_tweets, end_date)
        elif args.model == "lstm":
            m, tmr = train_seq(LSTMHead(X.shape[1], 64, 7), X, y, args.epochs, model_name="lstm")
            tomorrow_preds["LSTM     "] = tmr
            week_total_preds["LSTM     "] = predict_week_total(m, X, "seq", current_week_tweets, end_date)
        elif args.model == "trans":
            m, tmr = train_seq(TinyTrans(X.shape[1]), X, y, args.epochs, model_name="transformer")
            tomorrow_preds["Transform"] = tmr
            week_total_preds["Transform"] = predict_week_total(m, X, "seq", current_week_tweets, end_date)
        else:
            run_all()
            
        # Save predictions once at the end
        save_predictions(tomorrow_preds, "Tomorrow's tweet count", end_date, current_week_tweets)
        save_predictions(week_total_preds, "End of week total tweet count", end_date, current_week_tweets)
    finally:
        output_file.close()
