"""
User persona extraction for ChronoMind.

Analyzes messages to extract evidence-based persona traits
across four categories: habits, personal_facts, personality_traits,
and communication_style.

IMPORTANT: In this dataset, "User 1" and "User 2" refer to DIFFERENT
people in each conversation. Persona extraction works per-conversation,
then aggregates corpus-wide patterns. The resulting personas represent
the AGGREGATE profile across all people who appeared as "User 1" or
"User 2" throughout the dataset.

Quality rules:
  - Facts must appear in at least 2 distinct conversations.
  - Captured text must be meaningful (no stopwords/pronouns).
  - Hard cap: 20 facts per category, 50 total.
  - Confidence = f(frequency, evidence breadth).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .config import AppConfig
from .models import Message, PersonaTrait, UserPersona
from .parser import get_conversation_groups

logger = logging.getLogger(__name__)

# ─── Stopwords / junk capture filter ─────────────────────────────────────────
# These tokens, if they are the ENTIRE capture, indicate a meaningless match.
_CAPTURE_BLOCKLIST: Set[str] = {
    # Pronouns
    "it", "them", "that", "this", "those", "these", "he", "she", "they",
    "we", "you", "me", "him", "her", "us", "its", "their",
    # Adverbs / filler
    "too", "so", "very", "really", "quite", "just", "also", "well",
    "good", "great", "nice", "fine", "okay", "ok", "sure", "much",
    "more", "most", "less", "lot", "lots", "little", "bit",
    # Common short verb phrases
    "going", "doing", "trying", "getting", "working",
    "talking", "thinking", "looking", "coming", "seeing",
    # Generic nouns
    "things", "stuff", "people", "person", "way", "ways", "time",
    "times", "day", "days", "year", "years", "place", "places",
    # Articles / conjunctions
    "the", "a", "an", "and", "or", "but", "if", "of", "in", "on",
    "at", "to", "for", "with", "about", "from",
}

# Minimum meaningful words in a capture (after stripping stopwords)
_MIN_CONTENT_WORDS = 1
_MIN_CAPTURE_CHARS = 3
_MAX_CAPTURE_WORDS = 6  # captures longer than this are usually full sentences


def _is_meaningful_capture(text: str) -> bool:
    """
    Return True if the captured text is meaningful enough to be a persona fact.
    Filters out pronouns, stopwords, and generic filler phrases.
    """
    if not text:
        return False

    text = text.strip().rstrip(".,!?;:")
    if len(text) < _MIN_CAPTURE_CHARS:
        return False

    words = text.lower().split()

    # Block captures that are entirely stopwords
    content_words = [w for w in words if w not in _CAPTURE_BLOCKLIST and len(w) > 2]
    if len(content_words) < _MIN_CONTENT_WORDS:
        return False

    # Block captures that are too long (captured a whole sentence)
    if len(words) > _MAX_CAPTURE_WORDS:
        return False

    # Block captures that start with a verb form (suggests it's an activity, not a fact)
    gerund_starters = {"doing", "going", "trying", "making", "having", "being",
                       "working", "getting", "looking", "coming", "watching"}
    if words[0] in gerund_starters:
        return False

    return True


# ─── Habit Patterns ───────────────────────────────────────────────────────────
#
# STRICT FIRST-PERSON ONLY: every pattern requires an explicit "I", "my",
# "I'm", or "I've" subject from the speaker. The (?:i\s+)? optional form is
# intentionally avoided so that third-party mentions ("she goes to the gym",
# "cooking is fun") do NOT match.
#
# Each tuple: (compiled_regex, habit_label, min_occurrences)
# min_occurrences is the minimum number of distinct messages across the
# corpus before the habit is promoted. Higher = stricter.
#
_FP = r"(?:^|(?<=\s)|(?<=\.))"   # position anchor before the "I"

HABIT_PATTERNS: List[Tuple[str, str, int]] = [
    # ── Frequency adverbs: "I always/usually/often/regularly/every day ___" ──
    (r"\bI\s+(?:always|usually|often|regularly|every\s+(?:day|morning|night|week))\s+\w",
     "regular_habit", 5),

    # ── Gym / working out ──
    (r"\bI\s+(?:go|went|go\s+to)\s+(?:the\s+)?gym\b",         "gym",          3),
    (r"\bI\s+(?:work\s*out|workout|exercise|train)\b",          "workout",      5),
    (r"\bmy\s+(?:workout|gym|training)\b",                      "workout",      3),

    # ── Running / jogging ──
    # EXCLUDE "I run my own/a business/shop/farm/gallery" — those are occupation, not sport
    (r"\bI\s+(?:go\s+(?:for\s+)?(?:runs?|jogging)|jog\b|went\s+(?:for\s+a\s+)?(?:run|jog))\b",
     "running", 5),
    (r"\bI(?:'m|\s+am)\s+(?:a\s+)?runner\b",                    "running",      3),
    (r"\bI\s+run\s+(?:every|most|on\s+(?:Mondays?|Tuesdays?|weekdays?|weekends?))\b",
     "running", 3),

    # ── Reading ──
    (r"\bI\s+(?:love|like|enjoy|read)\s+(?:to\s+)?read(?:ing)?\b",             "reading", 5),
    (r"\bI\s+read\s+(?:books?|novels?|a\s+lot|every\s+(?:day|night))\b",       "reading", 3),
    (r"\bmy\s+(?:favourite|favorite)\s+(?:hobby|pastime|activity)\s+is\s+read", "reading", 3),

    # ── Cooking / baking ──
    (r"\bI\s+(?:love|like|enjoy)\s+(?:to\s+)?cook(?:ing)?\b",  "cooking",      5),
    (r"\bI\s+cook(?:ed)?\b",                                    "cooking",      5),
    (r"\bI\s+(?:love|like|enjoy)\s+(?:to\s+)?bak(?:e|ing)\b",  "baking",       5),
    (r"\bI\s+bak(?:e|ed|ing)\b",                                "baking",       5),

    # ── Gaming ──
    (r"\bI\s+(?:love|like|enjoy|play)\s+(?:video\s+)?games?\b", "gaming",       5),
    (r"\bI(?:'m|\s+am)\s+(?:a\s+)?gamer\b",                     "gaming",       3),
    (r"\bI\s+(?:play|playing)\s+(?:\w+\s+)?games?\b",           "gaming",       5),

    # ── Meditation / yoga ──
    (r"\bI\s+(?:meditat|do\s+yoga|practice\s+yoga|do\s+meditation)\w*",
     "meditation_yoga", 3),

    # ── Hiking / camping / outdoors ──
    (r"\bI\s+(?:love|like|enjoy|go)\s+(?:to\s+)?hik(?:e|ing)\b",  "hiking",    5),
    (r"\bI\s+hik(?:e|ed|ing)\b",                                    "hiking",   5),
    (r"\bI\s+(?:love|like|enjoy|go)\s+(?:to\s+)?camp(?:ing)?\b",   "camping",  3),

    # ── Walking ──
    (r"\bI\s+(?:walk|take\s+walks?|go\s+(?:for\s+)?walks?)\b",  "walking",     5),

    # ── Swimming / cycling / biking ──
    (r"\bI\s+(?:swim|swimming|cycle|cycling|bike|biking|ride\s+my\s+bike)\b",
     "cycling_swimming", 5),

    # ── Art / drawing / painting ──
    (r"\bI\s+(?:love|like|enjoy)\s+(?:to\s+)?(?:draw|paint|sketch|doodle)\w*\b",
     "art", 5),
    (r"\bI\s+(?:draw|paint|sketch)\b",                           "art",         5),
    (r"\bI(?:'m|\s+am)\s+(?:an?\s+)?artist\b",                  "art",         3),

    # ── Music (playing instrument / singing) ──
    (r"\bI\s+(?:play|plays?)\s+(?:the\s+)?(?:guitar|piano|drums|violin|bass|ukulele|saxophone)\b",
     "plays_instrument", 3),
    (r"\bI\s+(?:love|like|enjoy)\s+(?:to\s+)?sing(?:ing)?\b",   "singing",     5),
    (r"\bI\s+sing\b",                                             "singing",     5),

    # ── Dancing ──
    (r"\bI\s+(?:love|like|enjoy)\s+(?:to\s+)?danc(?:e|ing)\b",  "dancing",     5),
    (r"\bI\s+danc(?:e|ed)\b",                                    "dancing",     5),

    # ── Volunteering ──
    (r"\bI\s+volunteer\b",                                        "volunteering", 3),
    (r"\bI\s+(?:do|enjoy)\s+volunteer(?:ing|work)\b",            "volunteering", 3),

    # ── Gardening ──
    (r"\bI\s+(?:love|like|enjoy)\s+(?:to\s+)?garden(?:ing)?\b", "gardening",   3),
    (r"\bI\s+garden\b",                                           "gardening",   3),

    # ── Fishing ──
    (r"\bI\s+(?:love|like|enjoy|go)\s+(?:to\s+)?fish(?:ing)?\b","fishing",     3),

    # ── Sleep patterns ──
    (r"\bI\s+(?:sleep|slept)\s+(?:late|early|at\s+\d+|around\s+\d+)\b",
     "sleep_pattern", 3),
    (r"\bI\s+(?:wake|woke)\s+up\s+(?:early|late|at\s+\d+)\b",  "wake_pattern", 3),

    # ── Coffee routine ──
    (r"\bI\s+(?:drink|have|need)\s+coffee\s+(?:every|each|in\s+the)\s+morning\b",
     "coffee_routine", 3),
    (r"\bI(?:'m|\s+am)\s+(?:a\s+)?coffee\s+(?:person|lover|addict)\b",
     "coffee_routine", 3),

    # ── Work from home ──
    (r"\bI\s+work\s+from\s+home\b",                              "works_from_home", 3),

    # ── Studying ──
    (r"\bI\s+(?:study|am\s+studying)\s+",                        "studying",    5),

    # ── Photography ──
    (r"\bI\s+(?:love|like|enjoy|do)\s+(?:to\s+)?photograph(?:y|ing)?\b",
     "photography", 3),
    (r"\bI(?:'m|\s+am)\s+(?:a\s+)?photographer\b",               "photography", 3),

    # ── Swimming ──
    (r"\bI\s+(?:love|like|enjoy|go)\s+(?:to\s+)?swim(?:ming)?\b", "swimming",   3),
    (r"\bI\s+swim\b",                                              "swimming",   5),

    # ── Yoga / meditation ──
    (r"\bI\s+(?:love|like|enjoy|do|practice)\s+(?:to\s+)?yoga\b",  "yoga",      3),

    # ── Martial arts ──
    (r"\bI\s+(?:do|practice|love|like|enjoy)\s+(?:to\s+)?"
     r"(?:karate|judo|boxing|martial\s+arts|kickboxing|taekwondo|jiu.?jitsu)\b",
     "martial_arts", 3),

    # ── Travel ──
    (r"\bI\s+(?:love|like|enjoy)\s+(?:to\s+)?travel(?:ling|ing)?\b", "traveling", 5),
    (r"\bI\s+travel\s+(?:a\s+lot|frequently|often|regularly|every\s+year)\b",
     "traveling", 3),
]


# ─── Personal Fact Patterns ───────────────────────────────────────────────────
# Each entry: (regex, category, has_capture_group)
# Patterns WITHOUT a capture group produce a fixed label.
# Patterns WITH a capture group extract specific content that must pass
# the _is_meaningful_capture filter.

PERSONAL_FACT_PATTERNS: List[Tuple[str, str]] = [
    # Location — capture city/region
    (r"\bi\s+(?:live|am\s+living|stay|moved|moving)\s+(?:in|to|at)\s+([A-Za-z][A-Za-z\s]{2,30})(?:\.|,|!|$)", "location"),
    # Occupation — strict job titles only
    (r"\bi(?:'m|\s+am)\s+(?:a\s+)?(?:nurse|doctor|physician|surgeon|teacher|professor|engineer|"
     r"programmer|developer|designer|firefighter|police\s*officer|librarian|barista|chef|"
     r"veterinarian|pharmacist|accountant|lawyer|attorney|architect|scientist|researcher|"
     r"journalist|writer|artist|musician|therapist|counselor|psychologist|social\s+worker|"
     r"electrician|plumber|carpenter|mechanic|pilot|paramedic|dentist|radiologist|"
     r"muralist|fisherman|manager|director|analyst|consultant)\b", "job_title"),
    # Workplace
    (r"\bi\s+work\s+(?:at|for)\s+([A-Za-z][A-Za-z\s]{2,25})(?:\.|,|!|$)", "workplace"),
    # Age — numeric only
    (r"\bi\s+(?:am\s+)?(\d{1,2})\s+(?:years?\s+old|yo)\b", "age"),
    # Has a pet
    (r"\bi\s+(?:have|got|own)\s+(?:a\s+)?(?:dog|cat|puppy|kitten|pet|bird|hamster|parrot|rabbit)\b", "has_pet"),
    # Has children (numeric)
    (r"\bi\s+(?:have|got)\s+(\d+)\s+(?:kids?|children)\b", "has_n_children"),
    # Is a parent
    (r"\bi(?:'m|\s+am)\s+(?:a\s+)?(?:single\s+)?(?:mom|dad|mother|father|parent)\b", "is_parent"),
    # Has a partner
    (r"\bmy\s+(?:wife|husband|partner|girlfriend|boyfriend|spouse|fiancee?)\b", "has_partner"),
    # Has siblings / family
    (r"\bmy\s+(?:brother|sister|sibling|twin)\b", "has_sibling"),
    # Education — studying specific field
    (r"\bi(?:'m|\s+am)\s+(?:a\s+)?student\s+(?:of\s+|studying\s+)([A-Za-z][A-Za-z\s]{2,30})(?:\.|,|!|$)", "studies"),
    (r"\bi\s+(?:graduated|studied|majored)\s+in\s+([A-Za-z][A-Za-z\s]{2,30})(?:\.|,|!|$)", "studied"),
    # Specific interests (concrete nouns only, not generic "it/them")
    (r"\bi\s+(?:love|am\s+passionate\s+about)\s+((?:photography|cooking|music|travel|"
     r"reading|writing|gaming|sports?|art|science|nature|animals?|history|"
     r"politics|technology|fashion|food|movies?|films?|books?|anime|manga|"
     r"hiking|yoga|fitness|cars?|motorcycles?|gardening|theater|theatre|dance|dancing))\b", "interest"),
]


# ─── Personality keywords ─────────────────────────────────────────────────────
PERSONALITY_KEYWORDS: Dict[str, List[str]] = {
    "analytical": ["analyze", "analysis", "logical", "consider", "evaluate",
                   "data", "evidence", "research", "systematic", "figure out"],
    "humorous": ["haha", "lol", "lmao", "rofl", "funny", "joke", "hilarious",
                 "kidding", "just kidding", "cracking up"],
    "empathetic": ["i understand", "i feel for", "that must be", "sorry to hear",
                   "hope you", "take care", "how are you feeling", "i'm sorry",
                   "i can imagine", "glad to hear", "happy for you"],
    "adventurous": ["adventure", "travel", "explore", "try new", "exciting",
                    "road trip", "hiking", "camping", "bungee", "skydiving"],
    "creative": ["create", "design", "art", "draw", "paint", "write",
                 "creative", "build", "craft", "invent", "mural", "compose"],
    "introverted": ["alone", "quiet", "by myself", "introvert", "recharge",
                    "stay home", "solitude", "prefer staying in"],
    "extroverted": ["party", "socialize", "hang out with friends", "meet people",
                    "love crowds", "get together", "love being around people"],
    "optimistic": ["looking forward", "can't wait", "excited about", "positive",
                   "things will get better", "bright side", "hopeful"],
    "organized": ["plan ahead", "schedule", "organize", "calendar", "priority",
                  "deadline", "structured", "make a list"],
    "passionate": ["i'm passionate", "i'm obsessed", "dedicated to", "devoted to",
                   "i really love", "means everything", "my biggest passion"],
    "family_oriented": ["family is everything", "spend time with my kids",
                        "my children are", "family first", "close to my family"],
    "health_conscious": ["eating healthy", "stay fit", "nutrition", "diet",
                         "work out regularly", "staying healthy", "fitness goals"],
}


# ─── Main extraction function ─────────────────────────────────────────────────

def extract_persona(
    messages: List[Message],
    config: AppConfig,
) -> Dict[str, UserPersona]:
    """
    Extract structured personas for each user label in the conversation corpus.

    Args:
        messages: All conversation messages.
        config: Application configuration.

    Returns:
        Dict mapping user labels to UserPersona objects.
    """
    logger.info("Extracting user personas...")

    user_messages: Dict[str, List[Message]] = defaultdict(list)
    for msg in messages:
        user_messages[msg.user].append(msg)

    personas: Dict[str, UserPersona] = {}

    for user, msgs in user_messages.items():
        n_convos = len(set(m.conversation_id for m in msgs))
        logger.info("Analyzing '%s': %d messages across %d conversations...",
                    user, len(msgs), n_convos)

        habits = _extract_habits(msgs, config)
        facts = _extract_personal_facts(msgs, config)
        traits = _extract_personality_traits(msgs, config)
        style = _extract_communication_style(msgs, config)

        personas[user] = UserPersona(
            user_name=user,
            habits=habits,
            personal_facts=facts,
            personality_traits=traits,
            communication_style=style,
        )

    logger.info("Extracted personas for %d user labels.", len(personas))
    return personas

# ─── Human-readable label mapping ────────────────────────────────────────────
# Maps internal pattern labels (snake_case) to professional display names.
# Any label not listed here falls back to title-cased snake_case.

_HABIT_LABEL_MAP: Dict[str, str] = {
    "regular_habit":      "Routine Activities",
    "gym":                "Goes to the Gym",
    "workout":            "Regular Exercise",
    "running":            "Running / Jogging",
    "reading":            "Regular Reader",
    "studying":           "Academic Study",
    "cooking":            "Home Cooking",
    "baking":             "Baking",
    "gaming":             "Video Gaming",
    "meditation_yoga":    "Mindfulness Practice",
    "hiking":             "Hiking / Trail Walking",
    "camping":            "Camping / Outdoor Activities",
    "walking":            "Daily Walking",
    "cycling_swimming":   "Cycling or Swimming",
    "swimming":           "Swimming",
    "art":                "Visual Arts",
    "plays_instrument":   "Plays a Musical Instrument",
    "singing":            "Singing",
    "dancing":            "Dancing",
    "volunteering":       "Volunteer Work",
    "gardening":          "Gardening",
    "fishing":            "Fishing",
    "sleep_pattern":      "Sleep / Rest Routine",
    "wake_pattern":       "Sleep / Wake Routine",
    "coffee_routine":     "Beverage Preference",
    "works_from_home":    "Remote Work",
    "photography":        "Photography",
    "yoga":               "Yoga Practice",
    "martial_arts":       "Martial Arts",
    "traveling":          "Frequent Traveller",
    "fitness_sport":      "Fitness Activities",
    "outdoor_activities": "Outdoor Activities",
    "music":              "Music",
    "exercise":           "Regular Exercise",
    "coffee":             "Beverage Preference",
}


# ─── Category extractors ──────────────────────────────────────────────────────

def _extract_habits(msgs: List[Message], config: AppConfig) -> List[PersonaTrait]:
    """
    Extract habitual behaviors that are EXPLICITLY stated in first-person.

    Uses strict first-person patterns (all require 'I', 'my', or 'I'm').
    A habit is only promoted when it meets BOTH:
      - min_evidence_convos: appears in N distinct conversations
      - min_occurrences: total pattern matches across all messages

    Each returned trait includes:
      - evidence: list of message indices where it was found
      - snippets: up to 5 actual message texts for display
      - occurrences: total match count
    """
    import math
    pc = config.persona

    # label -> {counts, conv_ids, evidence_indices, snippets}
    accumulator: Dict[str, Dict] = {}

    for msg in msgs:
        text = msg.text  # preserve case for snippet storage; regex uses IGNORECASE

        for pattern, label, min_occ in HABIT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if label not in accumulator:
                    accumulator[label] = {
                        "count": 0,
                        "conv_ids": set(),
                        "indices": [],
                        "snippets": [],
                    }
                acc = accumulator[label]
                acc["count"] += 1
                acc["conv_ids"].add(msg.conversation_id)
                acc["indices"].append(msg.index)
                # Store snippet: trim to 120 chars, keep readable
                if len(acc["snippets"]) < 5:
                    snippet = text.strip()
                    if len(snippet) > 120:
                        snippet = snippet[:117] + "…"
                    if snippet not in acc["snippets"]:
                        acc["snippets"].append(snippet)

    traits = []
    total_convos = max(1, len(set(m.conversation_id for m in msgs)))
    total_msgs = max(1, len(msgs))

    for label, acc in accumulator.items():
        n_convos = len(acc["conv_ids"])
        n_count = acc["count"]

        # Find the min_occurrences requirement for this label
        # (take the lowest threshold among all patterns with this label)
        required_occ = min(
            occ for _, lbl, occ in HABIT_PATTERNS if lbl == label
        )

        # Gate 1: must appear in enough distinct conversations
        if n_convos < pc.min_evidence_convos:
            continue

        # Gate 2: must occur enough times in total
        if n_count < required_occ:
            continue

        # Log-scaled confidence: 3 convos→~0.46, 10→~0.53, 50→~0.67, 500→~0.91
        confidence = min(1.0, 0.30 + math.log1p(n_convos) * 0.10)

        if confidence >= pc.min_confidence:
            # Map internal label to a professional, human-readable display name
            display = _HABIT_LABEL_MAP.get(label, label.replace("_", " ").title())

            traits.append(PersonaTrait(
                trait=display,
                confidence=round(confidence, 2),
                evidence=sorted(set(acc["indices"]))[:10],
                category="habits",
                snippets=acc["snippets"],
                occurrences=n_count,
            ))

    traits.sort(key=lambda t: t.confidence, reverse=True)
    return traits[:pc.max_facts_per_category]


def _extract_personal_facts(msgs: List[Message], config: AppConfig) -> List[PersonaTrait]:
    """
    Extract personal facts using strict patterns + content filtering.

    Two tiers:
      - Per-value facts: specific value (e.g., "Location: Portland") that
        appears in >= min_evidence_convos conversations.
      - Per-category facts: generic signal (e.g., "Has A Job Title") that
        is true if the base category matches in >= min_evidence_convos convos.
    """
    pc = config.persona

    # key -> (display_label, [evidence_indices], {conv_ids})
    per_value: Dict[str, Tuple[str, List[int], set]] = {}
    # base_category -> {conv_ids}
    per_category: Dict[str, Tuple[str, List[int], set]] = {}

    for msg in msgs:
        text = msg.text

        for pattern, category in PERSONAL_FACT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue

            # Aggregate at category level regardless of capture value
            if category not in per_category:
                per_category[category] = (category.replace("_", " ").title(), [], set())
            per_category[category][1].append(msg.index)
            per_category[category][2].add(msg.conversation_id)

            # Try to extract a specific captured value
            if match.lastindex and match.lastindex >= 1:
                captured = match.group(1).strip().rstrip(".,!?;:")
                if not _is_meaningful_capture(captured):
                    continue
                display = f"{category.replace('_', ' ').title()}: {captured.title()}"
                key = f"{category}:{captured.lower()[:40]}"

                if key not in per_value:
                    per_value[key] = (display, [], set())
                per_value[key][1].append(msg.index)
                per_value[key][2].add(msg.conversation_id)

    total_convos = max(1, len(set(m.conversation_id for m in msgs)))
    total_msgs = max(1, len(msgs))
    traits = []

    # --- Tier 1: specific per-value facts ---
    for key, (display, evidence, conv_ids) in per_value.items():
        n_convos = len(conv_ids)
        if n_convos < pc.min_evidence_convos:
            continue
        # Use user-specific conversation count as denominator
        conv_coverage = n_convos / total_convos
        # Log-scaled to give meaningful scores: 2 convos -> ~0.41, 100 -> ~0.67, 500 -> ~0.78
        import math
        confidence = min(1.0, 0.3 + math.log1p(n_convos) * 0.12)
        if confidence >= pc.min_confidence:
            traits.append(PersonaTrait(
                trait=display,
                confidence=round(confidence, 2),
                evidence=sorted(set(evidence))[:10],
                category="personal_facts",
            ))

    # --- Tier 2: category-level facts (fallback when no per-value passed) ---
    for category, (display, evidence, conv_ids) in per_category.items():
        n_convos = len(conv_ids)
        if n_convos < pc.min_evidence_convos:
            continue
        has_specific = any(
            t.trait.lower().startswith(category.replace("_", " ").lower() + ":")
            for t in traits
        )
        if has_specific:
            continue
        import math
        confidence = min(1.0, 0.3 + math.log1p(n_convos) * 0.12)
        if confidence >= pc.min_confidence:
            traits.append(PersonaTrait(
                trait=display,
                confidence=round(confidence, 2),
                evidence=sorted(set(evidence))[:10],
                category="personal_facts",
            ))

    traits.sort(key=lambda t: t.confidence, reverse=True)

    # Keep max 5 per base category to avoid flooding with locations
    category_counts: Dict[str, int] = defaultdict(int)
    MAX_PER_BASE = 5
    filtered = []
    for t in traits:
        base = t.trait.split(":")[0].strip()
        if category_counts[base] < MAX_PER_BASE:
            filtered.append(t)
            category_counts[base] += 1

    return filtered[:pc.max_facts_per_category]


def _extract_personality_traits(msgs: List[Message], config: AppConfig) -> List[PersonaTrait]:
    """Extract personality traits based on keyword frequency across conversations."""
    pc = config.persona
    evidence: Dict[str, List[int]] = defaultdict(list)
    convos: Dict[str, set] = defaultdict(set)

    for msg in msgs:
        text = msg.text.lower()
        for trait, keywords in PERSONALITY_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    evidence[trait].append(msg.index)
                    convos[trait].add(msg.conversation_id)
                    break  # one match per trait per message

    total_msgs = max(1, len(msgs))
    total_convos = max(1, len(set(m.conversation_id for m in msgs)))
    traits = []

    for trait, ev in evidence.items():
        n_convos = len(convos[trait])
        if n_convos < pc.min_evidence_convos:
            continue

        import math
        confidence = min(1.0, 0.25 + math.log1p(n_convos) * 0.11)

        if confidence >= pc.min_confidence:
            traits.append(PersonaTrait(
                trait=trait.replace("_", " ").title(),
                confidence=round(confidence, 2),
                evidence=sorted(set(ev))[:10],
                category="personality_traits",
            ))

    traits.sort(key=lambda t: t.confidence, reverse=True)
    return traits[:pc.max_facts_per_category]


def _extract_communication_style(msgs: List[Message], config: AppConfig) -> List[PersonaTrait]:
    """Analyze statistical communication patterns: length, emoji, questions, tone."""
    if not msgs:
        return []

    pc = config.persona
    traits = []

    # Message length
    lengths = [len(m.text.split()) for m in msgs]
    avg_len = float(np.mean(lengths))

    if avg_len < 6:
        style, conf = "Very short messages (< 6 words avg)", min(1.0, 0.5 + (6 - avg_len) * 0.08)
    elif avg_len < 15:
        style, conf = "Short to medium messages (6-15 words avg)", 0.75
    elif avg_len < 30:
        style, conf = "Medium length messages (15-30 words avg)", 0.70
    else:
        style, conf = "Long detailed messages (30+ words avg)", min(1.0, 0.6 + (avg_len - 30) * 0.01)
    traits.append(PersonaTrait(
        trait=f"{style} (avg {avg_len:.1f}w)",
        confidence=round(conf, 2),
        evidence=[m.index for m in msgs[:5]],
        category="communication_style",
    ))

    # Emoji
    emoji_re = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        r"\U00002702-\U000027B0\U0001F900-\U0001F9FF]+",
        flags=re.UNICODE,
    )
    text_emoji_re = re.compile(r"[:;][-']?[)(DPp/\\|]|<3|xD|XD", re.IGNORECASE)
    emoji_msgs = [m for m in msgs if emoji_re.search(m.text) or text_emoji_re.search(m.text)]
    emoji_ratio = len(emoji_msgs) / len(msgs)

    if emoji_ratio > 0.25:
        traits.append(PersonaTrait(
            trait=f"Heavy emoji user ({emoji_ratio*100:.0f}% of messages)",
            confidence=round(min(1.0, emoji_ratio + 0.3), 2),
            evidence=[m.index for m in emoji_msgs[:10]],
            category="communication_style",
        ))
    elif emoji_ratio > 0.08:
        traits.append(PersonaTrait(
            trait=f"Moderate emoji usage ({emoji_ratio*100:.0f}% of messages)",
            confidence=round(emoji_ratio + 0.45, 2),
            evidence=[m.index for m in emoji_msgs[:10]],
            category="communication_style",
        ))

    # Questions
    q_msgs = [m for m in msgs if "?" in m.text]
    q_ratio = len(q_msgs) / len(msgs)
    if q_ratio > 0.25:
        traits.append(PersonaTrait(
            trait=f"Frequent question asker ({q_ratio*100:.0f}% of messages)",
            confidence=round(min(1.0, q_ratio + 0.3), 2),
            evidence=[m.index for m in q_msgs[:10]],
            category="communication_style",
        ))
    elif q_ratio > 0.12:
        traits.append(PersonaTrait(
            trait=f"Regularly asks questions ({q_ratio*100:.0f}% of messages)",
            confidence=round(q_ratio + 0.42, 2),
            evidence=[m.index for m in q_msgs[:10]],
            category="communication_style",
        ))

    # Exclamation
    exclaim_msgs = [m for m in msgs if "!" in m.text]
    exclaim_ratio = len(exclaim_msgs) / len(msgs)
    if exclaim_ratio > 0.20:
        traits.append(PersonaTrait(
            trait=f"Enthusiastic tone — frequent '!' ({exclaim_ratio*100:.0f}%)",
            confidence=round(min(1.0, exclaim_ratio + 0.3), 2),
            evidence=[m.index for m in exclaim_msgs[:10]],
            category="communication_style",
        ))

    # Supportive language
    support_re = re.compile(
        r"that'?s?\s+(?:great|awesome|cool|amazing|wonderful)|"
        r"good for you|i'?m happy\s+for|i'?m glad|keep it up|proud of you|"
        r"you(?:'re| are) doing great|well done|congrats",
        re.IGNORECASE,
    )
    support_msgs = [m for m in msgs if support_re.search(m.text)]
    support_ratio = len(support_msgs) / len(msgs)
    if support_ratio > 0.04:
        traits.append(PersonaTrait(
            trait=f"Supportive and encouraging ({support_ratio*100:.0f}% of messages)",
            confidence=round(min(1.0, support_ratio * 4 + 0.3), 2),
            evidence=[m.index for m in support_msgs[:10]],
            category="communication_style",
        ))

    # Vocabulary diversity (Type-Token Ratio)
    all_words = [w.lower() for m in msgs for w in m.text.split() if w.isalpha()]
    if len(all_words) > 50:
        ttr = len(set(all_words)) / len(all_words)
        if ttr > 0.5:
            traits.append(PersonaTrait(
                trait=f"Varied vocabulary (TTR={ttr:.2f})",
                confidence=round(min(1.0, ttr), 2),
                evidence=[],
                category="communication_style",
            ))

    return sorted(traits, key=lambda t: t.confidence, reverse=True)


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_personas(
    personas: Dict[str, UserPersona],
    output_dir: str | Path,
) -> None:
    """Save extracted personas to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_personas = {user: persona.to_dict() for user, persona in personas.items()}
    combined_path = output_dir / "personas.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_personas, f, indent=2, ensure_ascii=False)

    for user, persona in personas.items():
        safe_name = re.sub(r'[^\w\-]', '_', user.lower())
        user_path = output_dir / f"persona_{safe_name}.json"
        with open(user_path, "w", encoding="utf-8") as f:
            json.dump(persona.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("Saved personas to %s", output_dir)


def load_personas(persona_dir: str | Path) -> Dict[str, UserPersona]:
    """Load personas from the combined JSON file."""
    path = Path(persona_dir) / "personas.json"
    if not path.exists():
        logger.warning("Personas file not found: %s", path)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    personas = {}
    for user, pdata in data.items():
        persona = UserPersona(user_name=pdata["user_name"])
        for category in ["habits", "personal_facts", "personality_traits", "communication_style"]:
            items = []
            for item in pdata.get(category, []):
                items.append(PersonaTrait(
                    trait=item["trait"],
                    confidence=item["confidence"],
                    evidence=item.get("evidence", []),
                    category=category,
                    snippets=item.get("snippets", []),       # new field, backward-compat
                    occurrences=item.get("occurrences", 0),  # new field, backward-compat
                ))
            setattr(persona, category, items)
        personas[user] = persona

    logger.info("Loaded personas for %d users.", len(personas))
    return personas
