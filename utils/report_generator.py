"""
report_generator.py
Produces a self-contained HTML report from analysis results.
"""

import os
import base64
import datetime
import pandas as pd


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _chart_block(title: str, path: str, caption: str = "") -> str:
    data = _b64(path)
    cap = f'<p style="font-size:12px;color:#888;margin:4px 0 0">{caption}</p>' if caption else ""
    return f"""
    <div class="chart-card">
      <p class="chart-title">{title}</p>
      <img src="data:image/png;base64,{data}" alt="{title}" />
      {cap}
    </div>"""


def generate_html_report(df: pd.DataFrame, chart_paths: list[str],
                          out_path: str, insights: list[str]) -> str:
    now = datetime.datetime.now().strftime("%B %d, %Y %H:%M")

    # ── Summary stats
    total = len(df)
    counts = df["label"].value_counts()
    pos_n  = counts.get("positive", 0)
    neg_n  = counts.get("negative", 0)
    neu_n  = counts.get("neutral", 0)
    pos_p  = round(pos_n / total * 100, 1)
    neg_p  = round(neg_n / total * 100, 1)
    neu_p  = round(neu_n / total * 100, 1)
    avg_conf = round(df["confidence"].mean() * 100, 1)
    dom_emo = df["dominant_emotion"].mode()[0] if not df["dominant_emotion"].isna().all() else "—"

    chart_map = {}
    for p in chart_paths:
        fname = os.path.basename(p)
        chart_map[fname] = p

    def chart(fname, title, caption=""):
        p = chart_map.get(fname)
        if not p or not os.path.exists(p):
            return ""
        return _chart_block(title, p, caption)

    insight_rows = "".join(
        f'<li><span class="insight-icon">💡</span>{i}</li>' for i in insights
    )

    # ── Top records table
    top_df = df.nlargest(8, "confidence")[
        ["record_id", "source", "label", "confidence", "dominant_emotion", "keywords"]
    ].copy()
    top_df["confidence"] = top_df["confidence"].apply(lambda x: f"{x*100:.0f}%")
    top_df["keywords"] = top_df["keywords"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else x
    )
    def badge(label):
        colors = {"positive": "#3B6D11", "negative": "#A32D2D", "neutral": "#5F5E5A"}
        bgs    = {"positive": "#EAF3DE", "negative": "#FCEBEB", "neutral": "#F1EFE8"}
        c = colors.get(label, "#333")
        b = bgs.get(label, "#eee")
        return f'<span style="background:{b};color:{c};padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600">{label}</span>'

    rows_html = "".join(
        f"""<tr>
          <td>{r.record_id}</td>
          <td>{r.source.capitalize()}</td>
          <td>{badge(r.label)}</td>
          <td>{r.confidence}</td>
          <td>{r.dominant_emotion or "—"}</td>
          <td style="font-size:11px;color:#888">{r.keywords}</td>
        </tr>"""
        for r in top_df.itertuples()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sentiment Analysis Report</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        background:#F5F4F0;color:#2C2C2A;line-height:1.6}}
  .header{{background:#2C2C2A;color:#fff;padding:2.5rem 3rem}}
  .header h1{{font-size:26px;font-weight:600;margin-bottom:4px}}
  .header p{{font-size:13px;opacity:.6}}
  .container{{max-width:1100px;margin:0 auto;padding:2rem 2.5rem}}
  .metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:2rem}}
  .metric{{background:#fff;border-radius:10px;padding:1rem 1.25rem;border:0.5px solid #E0DDD6}}
  .metric .val{{font-size:26px;font-weight:600;line-height:1.1}}
  .metric .lbl{{font-size:11px;color:#888;margin-top:2px;text-transform:uppercase;letter-spacing:.05em}}
  .metric.pos .val{{color:#3B6D11}} .metric.neg .val{{color:#A32D2D}}
  .metric.neu .val{{color:#5F5E5A}} .metric.conf .val{{color:#185FA5}}
  .section-title{{font-size:15px;font-weight:600;color:#2C2C2A;margin:2rem 0 1rem;
                  padding-bottom:6px;border-bottom:1px solid #E0DDD6}}
  .chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:2rem}}
  .chart-grid.wide{{grid-template-columns:1fr}}
  .chart-card{{background:#fff;border-radius:10px;padding:1.25rem;border:0.5px solid #E0DDD6}}
  .chart-title{{font-size:12px;font-weight:600;color:#888;text-transform:uppercase;
                letter-spacing:.05em;margin-bottom:10px}}
  .chart-card img{{width:100%;border-radius:6px}}
  .insights{{background:#fff;border-radius:10px;padding:1.5rem;border:0.5px solid #E0DDD6;margin-bottom:2rem}}
  .insights ul{{list-style:none;display:flex;flex-direction:column;gap:10px}}
  .insights li{{font-size:14px;line-height:1.5;display:flex;align-items:flex-start;gap:8px}}
  .insight-icon{{flex-shrink:0}}
  table{{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;
         border:0.5px solid #E0DDD6}}
  th{{background:#F5F4F0;font-size:11px;text-transform:uppercase;letter-spacing:.05em;
      color:#888;padding:10px 14px;text-align:left;font-weight:600}}
  td{{padding:9px 14px;font-size:13px;border-top:0.5px solid #F0EDE6}}
  tr:hover td{{background:#FAFAF9}}
  .footer{{text-align:center;font-size:12px;color:#aaa;padding:2rem;margin-top:2rem}}
</style>
</head>
<body>
<div class="header">
  <h1>Sentiment Analysis Report</h1>
  <p>Generated {now} · VADER + TextBlob ensemble · {total} texts analysed</p>
</div>
<div class="container">

  <p class="section-title">Summary Metrics</p>
  <div class="metrics">
    <div class="metric"><div class="val">{total}</div><div class="lbl">Total texts</div></div>
    <div class="metric pos"><div class="val">{pos_p}%</div><div class="lbl">Positive · {pos_n}</div></div>
    <div class="metric neg"><div class="val">{neg_p}%</div><div class="lbl">Negative · {neg_n}</div></div>
    <div class="metric neu"><div class="val">{neu_p}%</div><div class="lbl">Neutral · {neu_n}</div></div>
    <div class="metric conf"><div class="val">{avg_conf}%</div><div class="lbl">Avg confidence</div></div>
    <div class="metric"><div class="val">{dom_emo.capitalize()}</div><div class="lbl">Top emotion</div></div>
  </div>

  <p class="section-title">Sentiment Overview</p>
  <div class="chart-grid">
    {chart("01_sentiment_distribution.png", "Sentiment Distribution")}
    {chart("02_sentiment_by_source.png", "Breakdown by Source")}
  </div>

  <p class="section-title">Emotion & Model Analysis</p>
  <div class="chart-grid">
    {chart("03_emotion_radar.png", "Emotion Profile Radar")}
    {chart("04_confidence_distribution.png", "Confidence Distribution")}
    {chart("05_model_comparison.png", "VADER vs TextBlob Comparison")}
    {chart("06_subjectivity_polarity.png", "Subjectivity vs Polarity")}
  </div>

  <p class="section-title">Cross-source Emotion Heatmap</p>
  <div class="chart-grid wide">
    {chart("07_emotion_heatmap.png", "Emotion Heatmap by Source")}
  </div>

  <p class="section-title">Keyword Frequency</p>
  <div class="chart-grid wide">
    {chart("08_wordcloud.png", "Word Cloud – Sentiment Keywords")}
  </div>

  <p class="section-title">Business Insights</p>
  <div class="insights">
    <ul>{insight_rows}</ul>
  </div>

  <p class="section-title">Highest-confidence Records</p>
  <table>
    <thead><tr>
      <th>ID</th><th>Source</th><th>Sentiment</th>
      <th>Confidence</th><th>Dominant Emotion</th><th>Keywords</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>

</div>
<div class="footer">Sentiment Analysis Platform · VADER + TextBlob · Python NLP</div>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
