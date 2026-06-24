"""
src/intent_classifier.py — Offline Intent Classifier for ChronoMind v2.

Classifies user messages into one of 5 intent categories using a
fully offline TF-IDF + Logistic Regression pipeline.

No external APIs. No network calls. CPU-only. < 50 MB. < 200ms inference.

Classes:
    reminder          — "Don't forget to...", "Remind me to..."
    emotional_support — "I'm feeling sad...", "I need someone to talk to..."
    action_item       — "Can you help me fix...", "I need to complete..."
    small_talk        — "How are you?", "Nice weather today"
    unknown           — Doesn't fit any of the above
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default model path (relative to project root)
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "intent_model.pkl"

# Human-readable class labels
INTENT_LABELS: Dict[str, str] = {
    "reminder":          "⏰ Reminder",
    "emotional_support": "💙 Emotional Support",
    "action_item":       "✅ Action Item",
    "small_talk":        "💬 Small Talk",
    "unknown":           "❓ Unknown",
}

INTENT_DESCRIPTIONS: Dict[str, str] = {
    "reminder":          "The message is asking to be reminded about something.",
    "emotional_support": "The user is expressing distress, sadness, or asking for emotional connection.",
    "action_item":       "The message requests specific help with a task, problem, or goal.",
    "small_talk":        "Casual, low-stakes conversation with no specific goal.",
    "unknown":           "The message does not clearly fit any of the defined categories.",
}


class IntentClassifier:
    """
    Lightweight offline intent classifier.

    Uses scikit-learn TF-IDF + Logistic Regression for fast, CPU-only inference.
    Model is loaded once and cached in-memory for subsequent calls.
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._pipeline = None
        self._classes = None

    def load(self) -> bool:
        """Load the trained model from disk."""
        if not self.model_path.exists():
            logger.warning("Intent model not found at %s. Run build_intent_model.py first.", self.model_path)
            return False
        try:
            with open(self.model_path, "rb") as f:
                bundle = pickle.load(f)
            self._pipeline = bundle["pipeline"]
            self._classes = bundle["classes"]
            logger.info("Intent classifier loaded from %s", self.model_path)
            return True
        except Exception as e:
            logger.error("Failed to load intent model: %s", e)
            return False

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    def predict(self, text: str) -> Dict[str, object]:
        """
        Classify a single message.

        Returns:
            dict with keys:
              - intent      : str — predicted class label
              - confidence  : float — probability of predicted class
              - all_scores  : Dict[str, float] — probability per class
              - latency_ms  : float — inference time in milliseconds
        """
        if not self.is_ready:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "all_scores": {},
                "latency_ms": 0.0,
            }

        t0 = time.perf_counter()
        proba = self._pipeline.predict_proba([text])[0]
        t1 = time.perf_counter()

        classes = self._pipeline.classes_
        all_scores = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
        best_idx = int(np.argmax(proba))
        intent = classes[best_idx]
        confidence = float(proba[best_idx])
        latency_ms = round((t1 - t0) * 1000, 2)

        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
            "latency_ms": latency_ms,
        }

    def predict_batch(self, texts: list[str]) -> list[Dict[str, object]]:
        """Classify a batch of messages."""
        return [self.predict(t) for t in texts]


# ─── Module-level singleton for app.py reuse ─────────────────────────────────

_singleton: Optional[IntentClassifier] = None


def get_intent_classifier(model_path: Optional[Path] = None) -> IntentClassifier:
    """Return a cached IntentClassifier instance (load once per process)."""
    global _singleton
    if _singleton is None:
        _singleton = IntentClassifier(model_path)
        _singleton.load()
    return _singleton
