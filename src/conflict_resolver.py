"""
src/conflict_resolver.py — Conflict-Aware Retrieval for ChronoMind v2.

Detects and resolves contradictory facts in retrieved evidence.

Algorithm:
    1. Extract entity–attribute pairs from all retrieved chunks
    2. Find entities that appear with conflicting attributes (e.g., location = Delhi vs Bangalore)
    3. Score each piece of evidence: recency + retrieval similarity + emotional weight
    4. Rank evidence and generate a merged, human-readable explanation
    5. Flag contradictions explicitly in the final answer

No external APIs. Fully offline. Rule-based NLP.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Entity–attribute extraction patterns ────────────────────────────────────
# Each pattern: (regex, attribute_name)
# Group 1 = entity value  (e.g., city name, job title)

_LOCATION_PATTERNS = [
    (r"\blive[sd]?\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",           "location"),
    (r"\bmove[sd]?\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",            "location"),
    (r"\bfrom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",                      "location"),
    (r"\bstay(?:ed|ing)?\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",      "location"),
    (r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:now|currently|today)", "location"),
]

_OCCUPATION_PATTERNS = [
    (r"\b(?:I'm|I am|work as|working as)\s+a\s+([a-z][a-z\s]{2,30}?)\b",  "occupation"),
    (r"\bmy\s+job\s+is\s+(?:a\s+)?([a-z][a-z\s]{2,30}?)\b",               "occupation"),
]

_RELATIONSHIP_PATTERNS = [
    (r"\b(?:my\s+)?sister\s+(?:lives?|moved?|works?)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
     "sister_location"),
    (r"\b(?:my\s+)?brother\s+(?:lives?|moved?|works?)\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
     "brother_location"),
]

ALL_PATTERNS = _LOCATION_PATTERNS + _OCCUPATION_PATTERNS + _RELATIONSHIP_PATTERNS


def _extract_facts(text: str) -> List[Tuple[str, str]]:
    """
    Extract (attribute, value) pairs from a text string.
    Returns list of (attribute, value) tuples.
    """
    facts = []
    for pattern, attr in ALL_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(1).strip().title()
            if len(value) >= 2:
                facts.append((attr, value))
    return facts


def _emotional_weight(text: str) -> float:
    """
    Return an emotional weight [0, 1] for a text.
    Higher weight = more emotionally salient content = more trustworthy.
    """
    emotional_words = {
        "family", "sister", "brother", "mother", "father", "married",
        "relationship", "job", "career", "important", "life", "events",
        "definitely", "always", "never", "absolutely", "clearly", "certainly",
        "moved", "now", "currently", "recently", "just", "today", "yesterday",
    }
    words = set(text.lower().split())
    overlap = len(words & emotional_words)
    return min(1.0, overlap / 4.0)


def _parse_message_index(messages_str: str) -> int:
    """Extract the start message index from a '123-456' range string."""
    try:
        return int(str(messages_str).split("-")[0])
    except Exception:
        return 0


def detect_conflicts(
    evidence_items: List[Dict[str, Any]],
    max_total_messages: int = 200_000,
) -> List[Dict[str, Any]]:
    """
    Scan retrieved evidence for contradictory facts.

    Args:
        evidence_items: List of evidence dicts (from chatbot.ask()).
                        Each must have 'text', 'messages', 'score' keys.
        max_total_messages: Used for recency normalization.

    Returns:
        List of conflict dicts, each with:
          - attribute     : what is conflicting (e.g., 'location', 'sister_location')
          - values        : list of distinct conflicting values
          - evidence      : list of evidence items supporting each value
          - resolution    : human-readable resolution text
          - best_evidence : the highest-ranked piece of evidence
    """
    # fact_registry[attribute][value] = list of (evidence_item, score)
    fact_registry: Dict[str, Dict[str, List[Tuple[Dict, float]]]] = defaultdict(lambda: defaultdict(list))

    for item in evidence_items:
        text = item.get("text", "")
        if not text:
            continue

        # Score = retrieval similarity × recency × emotional weight
        raw_score = item.get("score", 0.5)
        msg_idx   = _parse_message_index(item.get("messages", "0"))
        recency   = msg_idx / max(1, max_total_messages)   # higher index = more recent
        emot_w    = _emotional_weight(text)
        composite = raw_score * 0.5 + recency * 0.3 + emot_w * 0.2

        facts = _extract_facts(text)
        for attr, value in facts:
            fact_registry[attr][value].append((item, composite))

    conflicts = []
    for attr, value_map in fact_registry.items():
        if len(value_map) < 2:
            continue  # no conflict

        # Rank values by total composite score
        ranked_values = sorted(
            value_map.items(),
            key=lambda kv: sum(s for _, s in kv[1]),
            reverse=True,
        )

        best_value, best_evidence_list = ranked_values[0]
        other_values = [v for v, _ in ranked_values[1:]]
        best_item = best_evidence_list[0][0]

        # Build human-readable resolution
        attr_display = attr.replace("_", " ").title()
        resolution = _build_resolution(attr, best_value, other_values, best_item)

        # For generate_conflict_aware_answer, we also want the older items
        older_items = []
        for v, evlist in ranked_values[1:]:
            older_items.extend([e for e, _ in evlist])
        older_item = older_items[0] if older_items else None

        conflicts.append({
            "attribute":     attr,
            "display_name":  attr_display,
            "values":        [v for v, _ in ranked_values],
            "best_value":    best_value,
            "other_values":  other_values,
            "evidence":      [
                {"value": v, "items": [e for e, _ in evlist], "total_score": sum(s for _, s in evlist)}
                for v, evlist in ranked_values
            ],
            "best_evidence": best_item,
            "older_evidence": older_item,
            "resolution":    resolution,
        })

    return conflicts


def _build_resolution(attr: str, best: str, others: List[str], best_item: Dict) -> str:
    """Generate a human-readable resolution for a detected conflict."""
    attr_label = attr.replace("_", " ").title()
    others_str = " and ".join(f'**{o}**' for o in others[:3])
    msg_ref    = best_item.get("messages", "unknown")

    lines = [
        f"⚠️ **Conflicting information detected for: {attr_label}**",
        "",
        f"Older conversations mention: {others_str}",
        f"More recent evidence suggests: **{best}** (messages {msg_ref})",
        "",
        "The more recent information is likely current. "
        "Both may be accurate if the situation changed over time.",
    ]
    return "\n".join(lines)


def generate_conflict_aware_answer(query: str, evidence: List[Dict[str, Any]], conflicts: List[Dict[str, Any]]) -> str:
    """Generate a coherent merged answer specifically addressing the conflicts."""
    if not conflicts:
        return "No conflicts detected."
    
    # We will build a unified explanation for the first conflict found for simplicity
    c = conflicts[0]
    attr_label = c["display_name"].lower()
    older_vals = " and ".join(c["other_values"][:3])
    newer_val = c["best_value"]
    
    # Extract contextual snippet if possible
    older_text = c["older_evidence"]["text"] if c.get("older_evidence") else "the user's past situation"
    newer_text = c["best_evidence"]["text"] if c.get("best_evidence") else "a recent update"

    lines = [
        "Based on the retrieved conversations:",
        "",
        f"Earlier messages indicate {older_vals} regarding {attr_label}.",
        "",
        f"More recent messages suggest {newer_val}.",
        "",
        f"The {newer_val} information is likely the most current because it appears in newer conversations and received a higher evidence score.",
        "",
        "Both statements may be correct if the situation occurred over time."
    ]
    return "\n".join(lines)


def resolve(
    evidence_items: List[Dict[str, Any]],
    original_answer: str,
    max_total_messages: int = 200_000,
    query: str = "",
) -> Dict[str, Any]:
    """
    Run full conflict resolution on retrieved evidence.

    Returns:
        dict with:
          - has_conflicts  : bool
          - conflicts      : list of conflict dicts
          - n_conflicts    : int
          - augmented_answer: str — original answer + conflict notices appended
    """
    conflicts = detect_conflicts(evidence_items, max_total_messages)

    augmented = original_answer
    if conflicts:
        augmented += "\n\n---\n\n"
        augmented += f"**Conflicting information detected.**\n\n"
        augmented += generate_conflict_aware_answer(query, evidence_items, conflicts)
        augmented += "\n\n"
        for c in conflicts:
            augmented += "\n" + c["resolution"] + "\n"

    return {
        "has_conflicts":    len(conflicts) > 0,
        "n_conflicts":      len(conflicts),
        "conflicts":        conflicts,
        "augmented_answer": augmented,
    }
