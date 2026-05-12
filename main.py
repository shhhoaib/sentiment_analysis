"""
main.py  –  Sentiment Analysis Platform
========================================
Runs the full pipeline:
  1. Load sample data (Amazon reviews, social media, news)
  2. Analyse each record with the VADER+TextBlob ensemble engine
  3. Generate 8 visualisation charts
  4. Produce a self-contained HTML report
  5. Export results to CSV
"""

import sys
import os
import time
import pandas as pd

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data.sample_data import ALL_DATA
from models.sentiment_engine import SentimentEngine
from utils.visualizer import generate_all
from utils.report_generator import generate_html_report

REPORTS_DIR = os.path.join(ROOT, "reports")
CHARTS_DIR  = os.path.join(REPORTS_DIR, "charts")
CSV_PATH    = os.path.join(REPORTS_DIR, "results.csv")
HTML_PATH   = os.path.join(REPORTS_DIR, "report.html")

os.makedirs(CHARTS_DIR, exist_ok=True)


def _banner(msg: str):
    print(f"\n{'─'*60}\n  {msg}\n{'─'*60}")


def derive_insights(df: pd.DataFrame) -> list[str]:
    insights = []
    counts = df["label"].value_counts(normalize=True) * 100
    pos = counts.get("positive", 0)
    neg = counts.get("negative", 0)

    if pos > 55:
        insights.append(f"Overall sentiment is strongly positive ({pos:.0f}%). "
                        "Leverage this in marketing campaigns and customer testimonials.")
    elif neg > 40:
        insights.append(f"Negative sentiment is elevated ({neg:.0f}%). "
                        "Prioritise product/service improvements and proactive outreach.")
    else:
        insights.append("Sentiment is mixed. Segment your data by product line or "
                        "demographics to find pockets of dissatisfaction.")

    by_source = df.groupby("source")["label"].value_counts(normalize=True).unstack(fill_value=0)
    for src in by_source.index:
        neg_rate = by_source.loc[src].get("negative", 0) * 100
        pos_rate = by_source.loc[src].get("positive", 0) * 100
        if neg_rate > 40:
            insights.append(f"{src.capitalize()} data shows {neg_rate:.0f}% negative sentiment — "
                            "consider dedicated response and damage-control strategies here.")
        if pos_rate > 60:
            insights.append(f"{src.capitalize()} is a strong positive signal ({pos_rate:.0f}%) — "
                            "great source for case studies and social proof content.")

    dom_emo = df["dominant_emotion"].value_counts().idxmax()
    emo_map = {
        "joy":      "High joy levels — excellent for brand ambassador programmes.",
        "anger":    "Anger is the dominant emotion — immediate resolution and empathy responses needed.",
        "fear":     "Fear signals present — provide clear, reassuring communication and FAQs.",
        "sadness":  "Sadness detected — consider a loyalty or recovery programme for at-risk customers.",
        "trust":    "Trust is the dominant emotion — amplify reliability messaging in ads.",
        "surprise": "Surprise dominates — lean into novelty and 'wow factor' in product storytelling.",
        "disgust":  "Disgust signals detected — review quality control and service standards urgently.",
        "anticipation": "High anticipation — ideal timing for product launch or exclusive preview campaigns.",
    }
    if dom_emo in emo_map:
        insights.append(emo_map[dom_emo])

    high_subj = df[df["textblob_subjectivity"] > 0.7]
    if len(high_subj):
        pct = len(high_subj) / len(df) * 100
        insights.append(f"{pct:.0f}% of texts are highly subjective — "
                        "rich territory for qualitative insights and persona development.")

    avg_conf = df["confidence"].mean() * 100
    insights.append(f"Average model confidence is {avg_conf:.1f}%. "
                    "Texts below 60% confidence may benefit from manual review.")

    return insights


def main():
    t0 = time.time()
    _banner("Step 1 / 4 — Loading and analysing data")

    engine = SentimentEngine(vader_weight=0.6, textblob_weight=0.4)
    print(f"  Analysing {len(ALL_DATA)} records …")
    results = engine.analyze_batch(ALL_DATA)

    # Build DataFrame
    rows = []
    for res, rec in zip(results, ALL_DATA):
        row = {
            "record_id":             res.record_id,
            "source":                res.source,
            "text":                  res.text[:120] + ("…" if len(res.text) > 120 else ""),
            "label":                 res.label,
            "confidence":            res.confidence,
            "vader_compound":        res.vader_compound,
            "textblob_polarity":     res.textblob_polarity,
            "textblob_subjectivity": res.textblob_subjectivity,
            "dominant_emotion":      res.dominant_emotion,
            "keywords":              res.keywords,
        }
        row.update(res.emotions)  # joy, anger, fear, …
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  ✓ {len(df)} records processed")
    print(df["label"].value_counts().to_string())

    # ── CSV ───────────────────────────────────────────────────────────────────
    df.to_csv(CSV_PATH, index=False)
    print(f"\n  CSV saved → {CSV_PATH}")

    # ── Charts ────────────────────────────────────────────────────────────────
    _banner("Step 2 / 4 — Generating charts")
    chart_paths = generate_all(df, CHARTS_DIR)
    print(f"  ✓ {len(chart_paths)} charts saved to {CHARTS_DIR}")

    # ── Insights ──────────────────────────────────────────────────────────────
    _banner("Step 3 / 4 — Deriving business insights")
    insights = derive_insights(df)
    for i, ins in enumerate(insights, 1):
        print(f"  {i}. {ins}")

    # ── HTML report ───────────────────────────────────────────────────────────
    _banner("Step 4 / 4 — Building HTML report")
    generate_html_report(df, chart_paths, HTML_PATH, insights)
    print(f"  ✓ Report saved → {HTML_PATH}")

    elapsed = time.time() - t0
    _banner(f"Done in {elapsed:.1f}s")
    print(f"""
  Output files
  ────────────────────────────────────────
  CSV results  : {CSV_PATH}
  HTML report  : {HTML_PATH}
  Charts       : {CHARTS_DIR}/
  ────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
