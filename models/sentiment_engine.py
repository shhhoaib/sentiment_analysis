"""
sentiment_engine.py
Core NLP engine: VADER + TextBlob ensemble with emotion lexicon detection.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# ── Emotion lexicon ────────────────────────────────────────────────────────────
EMOTION_LEXICON: dict[str, list[str]] = {
    "joy":      ["happy","happiness","joy","love","wonderful","amazing","fantastic",
                 "excellent","great","awesome","delight","thrilled","excited","glad",
                 "pleased","grateful","enjoy","brilliant","superb","beautiful","perfect",
                 "incredible","magnificent","cheerful","euphoric","elated","bliss"],
    "anger":    ["angry","furious","outraged","mad","frustrated","annoyed","irritated",
                 "hate","enraged","hostile","bitter","infuriated","livid","disgusted",
                 "appalled","terrible","horrible","awful","useless","pathetic","worst"],
    "fear":     ["afraid","scared","worried","anxious","nervous","panic","dread",
                 "terror","fear","alarming","crisis","urgent","danger","threat","risk",
                 "catastrophic","dire","warning","irreversible","devastating"],
    "sadness":  ["sad","disappointed","unhappy","depressed","miserable","grief","sorrow",
                 "regret","sorry","upset","unfortunate","loss","heartbroken","gloomy",
                 "despair","hopeless","melancholy","crying","tears","devastated"],
    "surprise": ["wow","incredible","unbelievable","unexpected","astonished","shocked",
                 "amazed","stunning","extraordinary","remarkable","breathtaking","record",
                 "unprecedented","breakthrough","revolutionary","surpassing"],
    "trust":    ["reliable","safe","secure","honest","genuine","confident","trust",
                 "dependable","recommend","consistent","transparent","authentic",
                 "credible","integrity","proven","verified","certified","guaranteed"],
    "disgust":  ["disgusting","revolting","nauseating","gross","repulsive","vile",
                 "offensive","appalling","hideous","loathsome","abhorrent","toxic"],
    "anticipation": ["hope","expect","await","look forward","exciting","promising",
                     "upcoming","potential","opportunity","future","optimistic","eager"],
}

# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class SentimentResult:
    text: str
    label: str                          # positive | negative | neutral
    confidence: float                   # 0.0 – 1.0
    vader_compound: float               # –1.0 to +1.0
    textblob_polarity: float            # –1.0 to +1.0
    textblob_subjectivity: float        # 0.0 (objective) – 1.0 (subjective)
    emotions: dict[str, float] = field(default_factory=dict)
    dominant_emotion: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    source: str = ""
    record_id: str = ""

# ── Helpers ────────────────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    """Lowercase, strip URLs/hashtags/mentions, normalise whitespace."""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^\w\s'!?.,]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def _emotion_scores(clean_text: str) -> dict[str, float]:
    """Return per-emotion intensity in [0, 1] based on lexicon hits."""
    words = re.findall(r"\b\w+\b", clean_text)
    counts: dict[str, int] = {e: 0 for e in EMOTION_LEXICON}
    for word in words:
        for emotion, vocab in EMOTION_LEXICON.items():
            if word in vocab:
                counts[emotion] += 1
    total = sum(counts.values()) or 1
    return {e: round(c / total, 4) for e, c in counts.items()}

def _extract_keywords(clean_text: str, top_n: int = 6) -> list[str]:
    """Return the most sentiment-relevant words."""
    all_vocab = {w for vocab in EMOTION_LEXICON.values() for w in vocab}
    hits = [w for w in re.findall(r"\b\w+\b", clean_text) if w in all_vocab]
    # deduplicate while preserving order
    seen: set[str] = set()
    unique = []
    for w in hits:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    return unique[:top_n]

# ── Main analyser ──────────────────────────────────────────────────────────────
class SentimentEngine:
    """Ensemble NLP engine combining VADER and TextBlob."""

    def __init__(self, vader_weight: float = 0.6, textblob_weight: float = 0.4):
        self._vader = SentimentIntensityAnalyzer()
        self._vw = vader_weight
        self._tbw = textblob_weight

    def analyze(self, text: str, source: str = "", record_id: str = "") -> SentimentResult:
        clean = _clean(text)

        # VADER
        vs = self._vader.polarity_scores(text)          # use original (handles emoji/caps)
        vader_compound = vs["compound"]

        # TextBlob
        blob = TextBlob(clean)
        tb_polarity = blob.sentiment.polarity
        tb_subjectivity = blob.sentiment.subjectivity

        # Ensemble score
        ensemble = self._vw * vader_compound + self._tbw * tb_polarity

        # Label + confidence
        if ensemble >= 0.05:
            label = "positive"
            confidence = min(1.0, 0.5 + ensemble * 0.5)
        elif ensemble <= -0.05:
            label = "negative"
            confidence = min(1.0, 0.5 + abs(ensemble) * 0.5)
        else:
            label = "neutral"
            confidence = 1.0 - abs(ensemble) * 2

        confidence = round(max(0.0, min(1.0, confidence)), 4)

        # Emotions
        emotions = _emotion_scores(clean)
        dominant = max(emotions, key=emotions.get)
        if emotions[dominant] == 0:
            dominant = None

        return SentimentResult(
            text=text,
            label=label,
            confidence=confidence,
            vader_compound=round(vader_compound, 4),
            textblob_polarity=round(tb_polarity, 4),
            textblob_subjectivity=round(tb_subjectivity, 4),
            emotions=emotions,
            dominant_emotion=dominant,
            keywords=_extract_keywords(clean),
            source=source,
            record_id=record_id,
        )

    def analyze_batch(self, records: list[dict]) -> list[SentimentResult]:
        results = []
        for rec in records:
            r = self.analyze(
                text=rec["text"],
                source=rec.get("source", ""),
                record_id=rec.get("id", ""),
            )
            results.append(r)
        return results
