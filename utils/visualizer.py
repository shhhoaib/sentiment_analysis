"""
visualizer.py
Generates all charts and the word cloud for the sentiment analysis report.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# ── Palette ────────────────────────────────────────────────────────────────────
COLORS = {
    "positive": "#3B6D11",
    "negative": "#A32D2D",
    "neutral":  "#5F5E5A",
    "background": "#FAFAF9",
    "grid":     "#E8E6DF",
}
EMOTION_COLORS = {
    "joy":          "#3B6D11",
    "trust":        "#0F6E56",
    "anticipation": "#185FA5",
    "surprise":     "#533AB7",
    "fear":         "#854F0B",
    "disgust":      "#993556",
    "anger":        "#A32D2D",
    "sadness":      "#185FA5",
}
SENTIMENT_PALETTE = [COLORS["positive"], COLORS["neutral"], COLORS["negative"]]


def _style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(COLORS["background"])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10, color="#2C2C2A")
    ax.set_xlabel(xlabel, fontsize=10, color="#5F5E5A")
    ax.set_ylabel(ylabel, fontsize=10, color="#5F5E5A")
    ax.tick_params(colors="#5F5E5A", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLORS["grid"])
    ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.5)
    ax.set_axisbelow(True)


# ── 1. Sentiment distribution bar chart ───────────────────────────────────────
def plot_sentiment_distribution(df: pd.DataFrame, out_dir: str) -> str:
    counts = df["label"].value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=COLORS["background"])
    bars = ax.bar(counts.index, counts.values,
                  color=[COLORS[l] for l in counts.index],
                  width=0.5, edgecolor="none")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#2C2C2A")
    _style(ax, "Overall Sentiment Distribution", ylabel="Number of texts")
    plt.tight_layout()
    path = os.path.join(out_dir, "01_sentiment_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
    plt.close()
    return path


# ── 2. Sentiment by source (grouped bar) ──────────────────────────────────────
def plot_by_source(df: pd.DataFrame, out_dir: str) -> str:
    pivot = (df.groupby(["source", "label"]).size()
               .unstack(fill_value=0)
               .reindex(columns=["positive", "neutral", "negative"], fill_value=0))
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=COLORS["background"])
    x = np.arange(len(pivot))
    w = 0.25
    for i, (col, color) in enumerate(zip(pivot.columns, SENTIMENT_PALETTE)):
        rects = ax.bar(x + i * w, pivot[col], w, color=color, label=col.capitalize(), edgecolor="none")
        for r in rects:
            if r.get_height() > 0:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.1,
                        str(int(r.get_height())), ha="center", va="bottom", fontsize=8, color="#2C2C2A")
    ax.set_xticks(x + w)
    ax.set_xticklabels([s.capitalize() for s in pivot.index])
    ax.legend(fontsize=9, framealpha=0)
    _style(ax, "Sentiment Breakdown by Source", ylabel="Count")
    plt.tight_layout()
    path = os.path.join(out_dir, "02_sentiment_by_source.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
    plt.close()
    return path


# ── 3. Emotion radar chart ─────────────────────────────────────────────────────
def plot_emotion_radar(df: pd.DataFrame, out_dir: str) -> str:
    emotions = ["joy", "anger", "fear", "sadness", "surprise", "trust", "disgust", "anticipation"]
    avg = [df[e].mean() for e in emotions]
    avg_norm = [v / (max(avg) or 1) for v in avg]

    angles = np.linspace(0, 2 * np.pi, len(emotions), endpoint=False).tolist()
    avg_norm += avg_norm[:1]
    angles += angles[:1]
    labels = [e.capitalize() for e in emotions]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True},
                           facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    ax.plot(angles, avg_norm, color="#185FA5", linewidth=2)
    ax.fill(angles, avg_norm, color="#185FA5", alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10, color="#2C2C2A")
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.grid(color=COLORS["grid"])
    ax.set_title("Average Emotion Profile", fontsize=13, fontweight="bold", pad=20, color="#2C2C2A")

    plt.tight_layout()
    path = os.path.join(out_dir, "03_emotion_radar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
    plt.close()
    return path


# ── 4. Confidence distribution histogram ──────────────────────────────────────
def plot_confidence(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=COLORS["background"])
    for label, color in COLORS.items():
        if label not in ("positive", "negative", "neutral"):
            continue
        subset = df[df["label"] == label]["confidence"]
        if len(subset):
            ax.hist(subset, bins=10, range=(0, 1), alpha=0.75, color=color,
                    label=label.capitalize(), edgecolor="none")
    ax.legend(fontsize=9, framealpha=0)
    _style(ax, "Confidence Score Distribution", xlabel="Confidence", ylabel="Count")
    plt.tight_layout()
    path = os.path.join(out_dir, "04_confidence_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
    plt.close()
    return path


# ── 5. VADER vs TextBlob scatter ──────────────────────────────────────────────
def plot_model_comparison(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=COLORS["background"])
    for label, color in [("positive", COLORS["positive"]),
                         ("negative", COLORS["negative"]),
                         ("neutral",  COLORS["neutral"])]:
        sub = df[df["label"] == label]
        ax.scatter(sub["vader_compound"], sub["textblob_polarity"],
                   color=color, alpha=0.75, s=50, edgecolors="none", label=label.capitalize())
    ax.axhline(0, color=COLORS["grid"], linewidth=0.8)
    ax.axvline(0, color=COLORS["grid"], linewidth=0.8)
    ax.legend(fontsize=9, framealpha=0)
    _style(ax, "VADER vs TextBlob Polarity Scores",
           xlabel="VADER compound", ylabel="TextBlob polarity")
    plt.tight_layout()
    path = os.path.join(out_dir, "05_model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
    plt.close()
    return path


# ── 6. Subjectivity vs polarity bubble ────────────────────────────────────────
def plot_subjectivity(df: pd.DataFrame, out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=COLORS["background"])
    for label, color in [("positive", COLORS["positive"]),
                         ("negative", COLORS["negative"]),
                         ("neutral",  COLORS["neutral"])]:
        sub = df[df["label"] == label]
        ax.scatter(sub["textblob_polarity"], sub["textblob_subjectivity"],
                   color=color, s=sub["confidence"] * 120, alpha=0.7,
                   edgecolors="none", label=label.capitalize())
    ax.axhline(0.5, color=COLORS["grid"], linewidth=0.8, linestyle="--")
    ax.axvline(0, color=COLORS["grid"], linewidth=0.8)
    ax.legend(fontsize=9, framealpha=0)
    _style(ax, "Polarity vs Subjectivity (bubble = confidence)",
           xlabel="Polarity", ylabel="Subjectivity")
    plt.tight_layout()
    path = os.path.join(out_dir, "06_subjectivity_polarity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
    plt.close()
    return path


# ── 7. Emotion heatmap by source ──────────────────────────────────────────────
def plot_emotion_heatmap(df: pd.DataFrame, out_dir: str) -> str:
    emotions = ["joy", "anger", "fear", "sadness", "surprise", "trust", "disgust", "anticipation"]
    heat = df.groupby("source")[emotions].mean()
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=COLORS["background"])
    sns.heatmap(heat, ax=ax, cmap="Blues", annot=True, fmt=".2f",
                linewidths=0.5, linecolor=COLORS["background"],
                cbar_kws={"shrink": 0.7})
    ax.set_title("Average Emotion Intensity by Source", fontsize=13, fontweight="bold",
                 pad=10, color="#2C2C2A")
    ax.set_xlabel("")
    ax.set_yticklabels([s.capitalize() for s in heat.index], rotation=0, fontsize=10)
    ax.set_xticklabels([e.capitalize() for e in emotions], rotation=30, ha="right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "07_emotion_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
    plt.close()
    return path


# ── 8. Word cloud ─────────────────────────────────────────────────────────────
def plot_wordcloud(df: pd.DataFrame, out_dir: str) -> str:
    all_keywords = []
    for kw_list in df["keywords"]:
        if isinstance(kw_list, list):
            all_keywords.extend(kw_list)
        elif isinstance(kw_list, str) and kw_list:
            all_keywords.extend(kw_list.split(", "))
    freq = Counter(all_keywords)
    if not freq:
        return ""
    wc = WordCloud(width=900, height=420, background_color=COLORS["background"],
                   colormap="RdYlGn", max_words=60, prefer_horizontal=0.85,
                   collocations=False).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=COLORS["background"])
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most Frequent Sentiment Keywords", fontsize=13, fontweight="bold",
                 pad=10, color="#2C2C2A")
    fig.patch.set_facecolor(COLORS["background"])
    plt.tight_layout()
    path = os.path.join(out_dir, "08_wordcloud.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
    plt.close()
    return path


# ── Master function ────────────────────────────────────────────────────────────
def generate_all(df: pd.DataFrame, out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    funcs = [
        plot_sentiment_distribution,
        plot_by_source,
        plot_emotion_radar,
        plot_confidence,
        plot_model_comparison,
        plot_subjectivity,
        plot_emotion_heatmap,
        plot_wordcloud,
    ]
    paths = []
    for fn in funcs:
        try:
            p = fn(df, out_dir)
            if p:
                paths.append(p)
        except Exception as exc:
            print(f"  [warn] {fn.__name__} failed: {exc}")
    return paths
