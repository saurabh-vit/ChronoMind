"""
src/persona_drift.py — Adaptive Persona Engine for ChronoMind v2.

Tracks how each user's mood and tone evolve across conversation days.

Algorithm:
    1. Group messages by conversation_id (each ID = one "day" or session)
    2. Compute daily behavioral signals (sentiment, emoji freq, etc.)
    3. Infer mood + tone labels from signal thresholds
    4. Build a sorted timeline per user
    5. Detect drift events (consecutive mood/tone changes)
    6. Find probable drift triggers (keyword clusters, topic overlap)
    7. Save to persona/persona_drift.json

No external APIs. Uses VADER sentiment (bundled in NLTK).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── VADER sentiment (NLTK) ──────────────────────────────────────────────────
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    _VADER = SentimentIntensityAnalyzer()
    _HAS_VADER = True
except Exception:
    _HAS_VADER = False
    logger.warning("VADER not available; falling back to keyword-based sentiment.")

# ─── Constants ────────────────────────────────────────────────────────────────
EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002600-\U000027BF"
    "\U0001F900-\U0001F9FF]+",
    flags=re.UNICODE,
)

POSITIVE_KEYWORDS = {
    "love", "great", "awesome", "amazing", "happy", "wonderful", "fantastic",
    "excellent", "good", "nice", "thank", "thanks", "glad", "excited",
    "joy", "enjoy", "fun", "beautiful", "perfect", "yay", "cool", "best",
}
NEGATIVE_KEYWORDS = {
    "hate", "terrible", "awful", "bad", "sad", "angry", "frustrated",
    "upset", "depressed", "horrible", "worst", "disgusting", "annoying",
    "disappointing", "failed", "fail", "sick", "tired", "exhausted",
    "worried", "anxious", "stress", "stressed", "hopeless",
}
FORMAL_WORDS = {
    "therefore", "however", "furthermore", "consequently", "regarding",
    "pursuant", "accordingly", "notwithstanding", "hereby", "thus",
    "whereas", "aforesaid", "henceforth", "aforementioned",
}

# Trigger keywords per category
TRIGGER_CLUSTERS = {
    "job_search":      ["job", "work", "career", "interview", "resume", "hire", "fired", "layoff", "unemployed", "salary"],
    "relationship":    ["girlfriend", "boyfriend", "partner", "breakup", "divorce", "dating", "love", "relationship", "ex"],
    "health":          ["sick", "hospital", "doctor", "pain", "ill", "health", "disease", "therapy", "medication"],
    "financial":       ["money", "debt", "broke", "bills", "loan", "rent", "credit", "finance", "bankrupt"],
    "academic":        ["exam", "study", "grade", "school", "college", "university", "professor", "assignment", "thesis"],
    "family":          ["mom", "dad", "parent", "sibling", "brother", "sister", "family", "child", "kids", "baby"],
    "social":          ["friend", "party", "event", "meet", "gather", "social", "lonely", "isolated"],
}


def _sentiment_score(text: str) -> float:
    """Return compound sentiment score in [-1, 1]."""
    if _HAS_VADER:
        return _VADER.polarity_scores(text)["compound"]
    words = set(text.lower().split())
    pos = len(words & POSITIVE_KEYWORDS)
    neg = len(words & NEGATIVE_KEYWORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _compute_daily_signals(messages: List[Any]) -> Dict[str, float]:
    """Compute behavioral signals for a list of messages from one day/session."""
    if not messages:
        return {}

    texts = [m.text if hasattr(m, "text") else str(m) for m in messages]
    combined = " ".join(texts)
    words    = combined.split()
    n_msgs   = len(messages)

    scores      = [_sentiment_score(t) for t in texts]
    avg_sent    = sum(scores) / len(scores) if scores else 0.0
    positivity  = sum(1 for s in scores if s > 0.05) / n_msgs
    negativity  = sum(1 for s in scores if s < -0.05) / n_msgs

    questions   = sum(t.count("?") for t in texts)
    exclamations = sum(t.count("!") for t in texts)
    emojis      = sum(len(EMOJI_PATTERN.findall(t)) for t in texts)
    avg_length  = sum(len(t.split()) for t in texts) / n_msgs
    formal_cnt  = sum(1 for w in words if w.lower() in FORMAL_WORDS)

    return {
        "n_messages":        n_msgs,
        "avg_sentiment":     round(avg_sent, 4),
        "positivity_ratio":  round(positivity, 4),
        "negativity_ratio":  round(negativity, 4),
        "question_freq":     round(questions / n_msgs, 4),
        "exclamation_freq":  round(exclamations / n_msgs, 4),
        "emoji_freq":        round(emojis / n_msgs, 4),
        "avg_msg_length":    round(avg_length, 2),
        "formal_word_ratio": round(formal_cnt / max(1, len(words)), 4),
    }


def _infer_mood(signals: Dict[str, float]) -> str:
    """Infer mood label from behavioral signals."""
    s = signals.get("avg_sentiment", 0.0)
    q = signals.get("question_freq", 0.0)
    e = signals.get("exclamation_freq", 0.0)
    em = signals.get("emoji_freq", 0.0)
    neg = signals.get("negativity_ratio", 0.0)
    pos = signals.get("positivity_ratio", 0.0)

    if neg > 0.40 and s < -0.25:
        return "frustrated"
    if s < -0.20 and neg > 0.30:
        return "emotional"
    if s > 0.30 and e > 0.5 and em > 0.3:
        return "excited"
    if s > 0.15 and em > 0.2:
        return "playful"
    if q > 0.6 and s > -0.1:
        return "curious"
    return "neutral"


def _infer_tone(signals: Dict[str, float]) -> str:
    """Infer tone label from behavioral signals."""
    formal = signals.get("formal_word_ratio", 0.0)
    length = signals.get("avg_msg_length", 0.0)
    q      = signals.get("question_freq", 0.0)
    e      = signals.get("exclamation_freq", 0.0)
    emoji  = signals.get("emoji_freq", 0.0)
    sent   = signals.get("avg_sentiment", 0.0)

    if formal > 0.015 or length > 20:
        return "formal"
    if sent > 0.15 and q > 0.4:
        return "supportive"
    if emoji > 0.3 or (e > 0.5 and length < 8):
        return "casual"
    if length < 6 and e < 0.2:
        return "direct"
    return "casual"


def _detect_trigger(
    day_idx: int,
    timeline: List[Dict],
    all_messages: List[Any],
) -> Dict[str, Any]:
    """
    Find the probable trigger for a drift event.

    Looks at messages in the window surrounding the drift point for
    keyword cluster matches.
    """
    # Collect messages near the drift (±2 sessions)
    start = max(0, day_idx - 2)
    end   = min(len(timeline), day_idx + 2)
    window_msgs = []
    for i in range(start, end):
        entry = timeline[i]
        mid_msg = (entry.get("start_msg", 0) + entry.get("end_msg", 0)) // 2
        # grab a window around mid_msg from the full message list
        low = max(0, mid_msg - 50)
        high = min(len(all_messages), mid_msg + 50)
        window_msgs.extend(all_messages[low:high])

    if not window_msgs:
        return {"trigger": "unknown", "evidence_messages": [], "keywords": []}

    texts = [m.text if hasattr(m, "text") else str(m) for m in window_msgs]
    combined_lower = " ".join(texts).lower()
    words = combined_lower.split()
    word_counts = Counter(words)

    best_cluster = "unknown"
    best_score   = 0
    for cluster, keywords in TRIGGER_CLUSTERS.items():
        score = sum(word_counts.get(kw, 0) for kw in keywords)
        if score > best_score:
            best_score   = score
            best_cluster = cluster

    top_keywords = [kw for kw in TRIGGER_CLUSTERS.get(best_cluster, [])
                    if word_counts.get(kw, 0) > 0][:5]

    entry = timeline[day_idx]
    return {
        "trigger":          best_cluster.replace("_", " ").title() if best_cluster != "unknown" else "Unknown",
        "evidence_messages": [entry.get("start_msg", 0), entry.get("end_msg", 0)],
        "keywords":          top_keywords,
        "cluster_score":     best_score,
    }


def analyze_persona_drift(
    messages: List[Any],
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Main entry point for persona drift analysis.

    Args:
        messages: All Message objects (must have .user, .text, .conversation_id)
        output_path: Where to save persona_drift.json (optional)

    Returns:
        Full drift analysis dict
    """
    logger.info("Analyzing persona drift across %d messages...", len(messages))

    # Group messages by user, then by conversation_id (day/session)
    user_sessions: Dict[str, Dict[int, List[Any]]] = defaultdict(lambda: defaultdict(list))
    for msg in messages:
        user_sessions[msg.user][msg.conversation_id].append(msg)

    all_results: Dict[str, Any] = {}

    for user, sessions in user_sessions.items():
        logger.info("  Processing drift for '%s': %d sessions", user, len(sessions))

        # Sort sessions by conversation_id
        sorted_conv_ids = sorted(sessions.keys())

        timeline: List[Dict] = []
        for conv_id in sorted_conv_ids:
            msgs = sessions[conv_id]
            signals = _compute_daily_signals(msgs)
            if not signals:
                continue
            mood = _infer_mood(signals)
            tone = _infer_tone(signals)
            timeline.append({
                "day":       conv_id,
                "mood":      mood,
                "tone":      tone,
                "signals":   signals,
                "start_msg": msgs[0].index if hasattr(msgs[0], "index") else 0,
                "end_msg":   msgs[-1].index if hasattr(msgs[-1], "index") else 0,
            })

        # ── Detect drift events ────────────────────────────────────
        drift_events: List[Dict] = []
        for i in range(1, len(timeline)):
            prev = timeline[i - 1]
            curr = timeline[i]

            mood_changed = prev["mood"] != curr["mood"]
            tone_changed = prev["tone"] != curr["tone"]

            if mood_changed or tone_changed:
                trigger_info = _detect_trigger(i, timeline, messages)
                drift_events.append({
                    "from_day":   prev["day"],
                    "to_day":     curr["day"],
                    "mood_drift": f"{prev['mood']} → {curr['mood']}" if mood_changed else None,
                    "tone_drift": f"{prev['tone']} → {curr['tone']}" if tone_changed else None,
                    "trigger":    trigger_info["trigger"],
                    "keywords":   trigger_info["keywords"],
                    "evidence_messages": trigger_info["evidence_messages"],
                })

        # ── Compute aggregate statistics ───────────────────────────
        mood_counts = Counter(t["mood"] for t in timeline)
        tone_counts = Counter(t["tone"] for t in timeline)
        dominant_mood = mood_counts.most_common(1)[0][0] if mood_counts else "neutral"
        dominant_tone = tone_counts.most_common(1)[0][0] if tone_counts else "casual"

        all_results[user] = {
            "user":           user,
            "total_sessions": len(timeline),
            "dominant_mood":  dominant_mood,
            "dominant_tone":  dominant_tone,
            "mood_distribution":  dict(mood_counts),
            "tone_distribution":  dict(tone_counts),
            "timeline":       timeline,
            "drift_events":   drift_events,
            "total_drifts":   len(drift_events),
        }

        logger.info(
            "    %s: %d sessions, %d drift events, dominant mood=%s tone=%s",
            user, len(timeline), len(drift_events), dominant_mood, dominant_tone
        )

    # Save to disk
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info("Persona drift saved → %s", output_path)

    return all_results


def load_persona_drift(persona_dir: Path) -> Dict[str, Any]:
    """Load persona_drift.json from disk."""
    path = Path(persona_dir) / "persona_drift.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
