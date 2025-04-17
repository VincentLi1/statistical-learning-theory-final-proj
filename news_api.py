# news_sentiment_pipeline.py  – multi‑week, CLI‑configurable
"""Pull Elon Musk news for an arbitrary look‑back window, score with VADER (and
HF if available), and aggregate weekly sentiment.

Usage examples
--------------
Fetch the last **30 days** and keep HF disabled to avoid torch install:

    python news_sentiment_pipeline.py --days 30 --hf off

Fetch 90 days and use HF (requires torch or tf):

    python news_sentiment_pipeline.py -d 90 --hf on
"""

import argparse
import datetime as dt
import re
import sys
import requests
import pandas as pd
from bs4 import BeautifulSoup

###########################################################################
# 0)  ----------  Parse CLI args
###########################################################################

parser = argparse.ArgumentParser(description="Elon Musk weekly news sentiment")
parser.add_argument("--days", "-d", type=int, default=30, help="How many days back to fetch (default=30)")
parser.add_argument("--hf", choices=["on", "off"], default="on", help="Enable HuggingFace model if available")
args = parser.parse_args()
DAYS_BACK = args.days

print(f"▶ Fetching last {DAYS_BACK} days of news …")

###########################################################################
# 1)  ----------  VADER setup (auto‑download lexicon)
###########################################################################

import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
print("✔ VADER ready")

###########################################################################
# 2)  ----------  Optional Hugging Face model
###########################################################################

USE_HF = False
if args.hf == "on":
    try:
        from transformers import pipeline
        hf_sent = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            top_k=None,
        )
        USE_HF = True
        print("✔ HF sentiment model loaded")
    except Exception as e:
        print("⚠️  HF pipeline unavailable (", e, ") – continuing with VADER only")
else:
    print("ℹ️  HF disabled via CLI flag – VADER‑only mode")

###########################################################################
# 3)  ----------  Fetch utilities (GDELT → Google RSS fallback)
###########################################################################

HEADERS = {"User-Agent": "Mozilla/5.0 (news sentiment script)"}


def fetch_gdelt(query: str, days_back: int, maxrecords: int = 250):
    start = (dt.date.today() - dt.timedelta(days=days_back)).isoformat()
    q = requests.utils.quote(query)
    url = (
        "https://api.gdeltproject.org/api/v2/docapi/search"
        f"?query={q}&filter=SourceCommonName:news&maxrecords={maxrecords}"
        "&format=json&sort=HybridRel&mode=ArtList"
        f"&filter=PublishDate>={start}"
    )
    print("→ GDELT", url)
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return r.json().get("articles", [])
    except Exception as e:
        print("⚠️  GDELT failed:", e)
        return []


def fetch_google_rss(query: str, days_back: int, max_items: int = 500):
    import feedparser

    rss_url = (
        "https://news.google.com/rss/search?q="
        + requests.utils.quote(query)
        + f"+when:{days_back}d&hl=en-US&gl=US&ceid=US:en"
    )
    print("→ Google News RSS", rss_url)
    feed = feedparser.parse(rss_url)
    return [
        {"url": e.link, "title": e.title, "published": e.published}
        for e in feed.entries[:max_items]
    ]

###########################################################################
# 4)  ----------  Fetch articles
###########################################################################

QUERY = "Elon Musk"
articles = fetch_gdelt(QUERY, DAYS_BACK)
if len(articles) < 10:
    print("🔄 Fallback to Google News RSS …")
    articles = fetch_google_rss(QUERY, DAYS_BACK)

print(f"Fetched {len(articles)} articles total")
if not articles:
    sys.exit("No data – aborting.")

###########################################################################
# 5)  ----------  Scrape full text (simple)
###########################################################################

def get_full_text(url: str) -> str:
    try:
        html = requests.get(url, headers=HEADERS, timeout=5).text
        soup = BeautifulSoup(html, "html.parser")
        return " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))[:20_000]
    except Exception:
        return ""

for art in articles:
    art.setdefault("text", get_full_text(art.get("url", "")))

###########################################################################
# 6)  ----------  Clean text & sentiment scoring
###########################################################################

clean = lambda s: re.sub(r"\s+", " ", s or "").strip()

for art in articles:
    art["clean"] = clean(" ".join([art.get("title", ""), art.get("snippet", ""), art.get("text", "")]))
    art["vader"] = sia.polarity_scores(art["clean"])["compound"]

if USE_HF:
    hf_out = hf_sent([a["clean"] for a in articles], batch_size=16)
    for art, res in zip(articles, hf_out):
        pos_prob = next(x["score"] for x in res if x["label"] == "POSITIVE")
        art["hf"] = 2 * pos_prob - 1
else:
    for art in articles:
        art["hf"] = None

###########################################################################
# 7)  ----------  Weekly aggregation
###########################################################################

df = pd.DataFrame(articles)

if "seendate" in df.columns:
    date_series = pd.to_datetime(df["seendate"], errors="coerce")
elif "published" in df.columns:
    date_series = pd.to_datetime(df["published"], errors="coerce")
else:
    date_series = pd.to_datetime("today")

df["week"] = date_series.dt.to_period("W").dt.start_time
weekly = (
    df.groupby("week", dropna=True)
      .agg(vader_mean=("vader", "mean"),
            hf_mean=("hf", "mean"),
            n_articles=("url", "nunique"))
      .reset_index()
      .sort_values("week")
)
print("\nWeekly sentiment summary:\n", weekly)
weekly.to_csv("musksentiment_weekly.csv", index=False)
print("CSV written → musksentiment_weekly.csv")
