"""
answer_synthesizer.py — Natural-language answer synthesis for ChronoMind.

Takes a RetrievalResult (structured evidence) and produces a human-readable
answer WITHOUT dumping raw retrieval data. Evidence is returned separately
for the UI to render in a collapsible section.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .models import RetrievalResult, UserPersona


# ─── Query intent classification ─────────────────────────────────────────────

def _classify_intent(query: str) -> str:
    """
    Classify the query intent into one of several categories.
    Returns: 'persona', 'habits', 'communication', 'topics',
             'location', 'occupation', 'interests', 'relationship', 'general'
    """
    q = query.lower()

    if any(k in q for k in ["who is", "what kind of person", "what type of person",
                              "personality", "character", "describe"]):
        return "persona"

    if any(k in q for k in ["habit", "routine", "regularly", "every day",
                              "always", "wake up", "sleep", "workout", "exercise"]):
        return "habits"

    if any(k in q for k in ["communicate", "communication", "style",
                              "emoji", "message length", "questions", "tone"]):
        return "communication"

    if any(k in q for k in ["job", "work", "occupation", "profession",
                              "career", "employed", "do for a living"]):
        return "occupation"

    if any(k in q for k in ["live", "location", "city", "where", "place",
                              "area", "region", "state"]):
        return "location"

    if any(k in q for k in ["interest", "hobby", "hobbies", "like", "enjoy",
                              "passion", "love doing", "favourite", "favorite"]):
        return "interests"

    if any(k in q for k in ["relationship", "partner", "family", "children",
                              "married", "girlfriend", "boyfriend", "kids"]):
        return "relationship"

    if any(k in q for k in ["topic", "discuss", "talked about",
                              "conversation about", "subjects"]):
        return "topics"

    return "general"


# ─── Persona synthesizers ─────────────────────────────────────────────────────

def _extract_persona_facts(
    personas: Dict[str, UserPersona],
    category_filter: Optional[str] = None,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract human-readable trait lists per user, optionally filtered by category.
    Returns: {user: {category: [trait_string, ...]}}
    """
    result = {}
    cats = ["habits", "personal_facts", "personality_traits", "communication_style"]
    if category_filter:
        cats = [c for c in cats if category_filter in c]

    for user, persona in personas.items():
        pd = persona.to_dict()
        result[user] = {}
        for cat in cats:
            items = pd.get(cat, [])
            if items:
                result[user][cat] = [item["trait"] for item in items]

    return result


def _synthesize_persona_answer(
    query: str,
    rr: RetrievalResult,
    personas: Dict[str, UserPersona],
    intent: str,
) -> str:
    """Build a natural-language persona summary."""

    if not personas:
        return _synthesize_from_topics(query, rr, intent)

    sections = []

    for user, persona in personas.items():
        pd = persona.to_dict()

        # Personality paragraph
        personality = [t["trait"] for t in pd.get("personality_traits", [])]
        habits = [t["trait"] for t in pd.get("habits", [])[:6]]
        facts = [t["trait"] for t in pd.get("personal_facts", [])[:8]]
        comms = [t["trait"] for t in pd.get("communication_style", [])]

        # Build a flowing description
        lines = [f"**{user}**"]

        if personality:
            p_str = ", ".join(personality[:4]).lower()
            lines.append(
                f"Across the conversations, {user} comes across as **{p_str}**."
            )

        if facts:
            # Separate interests from other facts
            interests = [f for f in facts if f.lower().startswith("interest:")]
            other_facts = [f for f in facts if not f.lower().startswith("interest:")]

            if other_facts:
                lines.append("**Personal background:** " + " · ".join(other_facts[:5]))

            if interests:
                interest_names = [i.split(":")[-1].strip() for i in interests[:5]]
                lines.append(
                    f"**Interests & passions:** {', '.join(interest_names)}"
                )

        if habits:
            hab_str = ", ".join(h.lower() for h in habits[:5])
            lines.append(f"**Regular habits:** {hab_str}")

        if comms:
            lines.append("**Communication style:** " + " · ".join(comms[:3]))

        sections.append("\n".join(lines))

    return "\n\n---\n\n".join(sections)


def _synthesize_habits_answer(
    rr: RetrievalResult,
    personas: Dict[str, UserPersona],
) -> str:
    if not personas:
        return _synthesize_from_topics_raw(rr)

    lines = []
    for user, persona in personas.items():
        pd = persona.to_dict()
        habits = [t["trait"] for t in pd.get("habits", [])]
        if habits:
            lines.append(f"**{user}** — {len(habits)} habits identified:")
            for h in habits:
                lines.append(f"  • {h}")

    if not lines:
        return "No specific habits were consistently detected across the conversations."

    return (
        "Based on patterns across all conversations, here are the regularly "
        "occurring habits:\n\n" + "\n".join(lines)
    )


def _synthesize_communication_answer(
    rr: RetrievalResult,
    personas: Dict[str, UserPersona],
) -> str:
    if not personas:
        return _synthesize_from_topics_raw(rr)

    sections = []
    for user, persona in personas.items():
        pd = persona.to_dict()
        comms = pd.get("communication_style", [])
        if comms:
            traits = [t["trait"] for t in comms]
            sections.append(f"**{user}:** " + " · ".join(traits))

    if not sections:
        return "No specific communication style patterns were detected."

    return (
        "Here is how each user communicates based on statistical analysis "
        "of all messages:\n\n" + "\n\n".join(sections)
    )


def _synthesize_occupation_answer(
    rr: RetrievalResult,
    personas: Dict[str, UserPersona],
) -> str:
    if not personas:
        return _synthesize_from_topics_raw(rr)

    lines = []
    for user, persona in personas.items():
        pd = persona.to_dict()
        facts = pd.get("personal_facts", [])
        # Only use proper job/workplace facts — exclude Preference: entries
        jobs = [
            t["trait"] for t in facts
            if any(kw in t["trait"].lower()
                   for kw in ["job title", "workplace", "work for", "occupation"])
            and not t["trait"].lower().startswith("preference:")
            and not t["trait"].lower().startswith("occupation: ")
        ]
        # Also include raw job_title category-level fact
        job_generics = [t["trait"] for t in facts
                        if t["trait"].lower() in ("job title", "workplace")]
        jobs = jobs + [j for j in job_generics if j not in jobs]

        if jobs:
            lines.append(f"**{user}:** " + ", ".join(jobs[:5]))

    if not lines:
        topic_text = _synthesize_from_topics_raw(rr)
        if topic_text:
            return (
                "No single dominant occupation was detected consistently. "
                "Here are relevant excerpts from conversations:\n\n" + topic_text
            )
        return "No occupation information was consistently detected."

    return (
        "Based on what was said across the conversations:\n\n"
        + "\n".join(lines)
        + "\n\n*Note: Since User 1 and User 2 represent different people in each "
        "conversation, these reflect aggregate patterns across many people.*"
    )


def _synthesize_location_answer(
    rr: RetrievalResult,
    personas: Dict[str, UserPersona],
) -> str:
    if not personas:
        return _synthesize_from_topics_raw(rr)

    lines = []
    for user, persona in personas.items():
        pd = persona.to_dict()
        facts = pd.get("personal_facts", [])
        locations = [t["trait"] for t in facts if "location" in t["trait"].lower()]
        if locations:
            lines.append(f"**{user}:** " + ", ".join(locations))

    # Also check raw chunks for location mentions
    chunk_locations = []
    for rc in rr.raw_chunks[:3]:
        text = rc.get("text", "")
        # Find "I live in X" patterns
        matches = re.findall(
            r"i\s+(?:live|moved?|am\s+living)\s+(?:in|to)\s+([A-Za-z][A-Za-z\s,]{2,30})",
            text, re.IGNORECASE
        )
        chunk_locations.extend(m.strip() for m in matches if m.strip())

    if lines or chunk_locations:
        result = "Locations mentioned across the conversations:\n\n"
        result += "\n".join(lines) if lines else ""
        if chunk_locations:
            unique_locs = list(dict.fromkeys(chunk_locations))[:8]
            result += f"\n\nFrom conversation excerpts: {', '.join(unique_locs)}"
        return result

    return "No specific location information was detected with enough frequency."


def _synthesize_interests_answer(
    rr: RetrievalResult,
    personas: Dict[str, UserPersona],
) -> str:
    if not personas:
        return _synthesize_from_topics_raw(rr)

    lines = []
    for user, persona in personas.items():
        pd = persona.to_dict()
        facts = pd.get("personal_facts", [])
        interests = [
            t["trait"].split(":")[-1].strip()
            for t in facts
            if t["trait"].lower().startswith("interest:")
        ]
        habits = [t["trait"] for t in pd.get("habits", [])[:5]]

        user_lines = []
        if interests:
            user_lines.append(f"  **Stated interests:** {', '.join(interests)}")
        if habits:
            user_lines.append(f"  **Regular activities:** {', '.join(h.lower() for h in habits)}")

        if user_lines:
            lines.append(f"**{user}**\n" + "\n".join(user_lines))

    if not lines:
        return "No specific interests or hobbies were consistently detected."

    return (
        "Here are the hobbies and interests identified from the conversations:\n\n"
        + "\n\n".join(lines)
    )


def _synthesize_relationship_answer(
    rr: RetrievalResult,
    personas: Dict[str, UserPersona],
) -> str:
    if not personas:
        return _synthesize_from_topics_raw(rr)

    lines = []
    for user, persona in personas.items():
        pd = persona.to_dict()
        facts = pd.get("personal_facts", [])
        # Only proper relationship/family facts — exclude Preference: entries
        rel_keywords = ["partner", "sibling", "parent", "children", "family", "has "]
        rels = [
            t["trait"] for t in facts
            if any(kw in t["trait"].lower() for kw in rel_keywords)
            and not t["trait"].lower().startswith("preference:")
        ]
        if rels:
            lines.append(f"**{user}:** " + ", ".join(rels[:6]))

    if not lines:
        return _synthesize_from_topics_raw(rr) or "No relationship information was consistently detected."

    return "Based on conversations, here is what we know about relationships:\n\n" + "\n".join(lines)


def _synthesize_topics_answer(rr: RetrievalResult) -> str:
    """Synthesize answer about discussion topics from topic/checkpoint summaries only."""
    if not rr.topic_summaries and not rr.checkpoint_summaries:
        return "No matching topic discussions were found for your query."

    # Use ONLY topic and checkpoint summaries — NOT raw chunk text
    summaries = []
    for ts in rr.topic_summaries[:5]:
        s = ts.get("summary", "").strip()
        # Skip entries that look like raw dialogue (contain ":" speaker markers)
        if s and len(s) > 30 and not re.search(r'^User [12]:', s):
            summaries.append(s)

    for cs in rr.checkpoint_summaries[:2]:
        s = cs.get("summary", "").strip()
        if s and len(s) > 30 and not re.search(r'^User [12]:', s):
            summaries.append(s)

    if not summaries:
        # Fall back but still filter to sentence fragments
        raw = _synthesize_from_topics_raw(rr)
        if not raw:
            return "Relevant topics were found but summaries are unavailable."
        # Extract first 2 meaningful sentences from raw
        sentences = re.split(r'(?<=[.!?])\s+', raw)
        clean = [s for s in sentences if len(s) > 20][:3]
        return "Based on retrieved conversations:\n\n" + " ".join(clean)

    # Deduplicate sentences across summaries
    seen: set = set()
    output: list = []
    for summary in summaries:
        for sent in re.split(r'(?<=[.!?])\s+', summary):
            s = sent.strip()
            if len(s) > 15 and s.lower() not in seen:
                seen.add(s.lower())
                output.append(s)

    combined = " ".join(output[:10])
    return (
        "Here are the most relevant discussion topics found:\n\n"
        + combined
    )


def _synthesize_from_topics(query: str, rr: RetrievalResult, intent: str) -> str:
    """Fallback: synthesize answer from topic and checkpoint summaries only."""
    texts = []

    for ts in rr.topic_summaries[:4]:
        s = ts.get("summary", "").strip()
        if s:
            texts.append(s)

    for cs in rr.checkpoint_summaries[:2]:
        s = cs.get("summary", "").strip()
        if s:
            texts.append(s)

    if not texts:
        return (
            "I couldn't find specific information for your query. "
            "Try asking about personality traits, habits, communication style, "
            "or specific topics discussed in the conversations."
        )

    combined = " ".join(texts)
    # Trim to a reasonable length
    if len(combined) > 800:
        combined = combined[:800].rsplit(".", 1)[0] + "."

    return f"Based on the retrieved conversation summaries:\n\n{combined}"


def _synthesize_from_topics_raw(rr: RetrievalResult) -> str:
    """Return concatenated summaries without extra prose."""
    texts = []
    for ts in rr.topic_summaries[:3]:
        s = ts.get("summary", "").strip()
        if s:
            texts.append(s)
    return " ".join(texts)


# ─── Main synthesis entry point ───────────────────────────────────────────────

def synthesize_answer(
    query: str,
    rr: RetrievalResult,
    personas: Optional[Dict[str, UserPersona]] = None,
) -> Tuple[str, List[Dict]]:
    """
    Synthesize a human-readable answer from retrieved evidence.

    Args:
        query: The user's question.
        rr: RetrievalResult with structured evidence.
        personas: Loaded persona data (optional).

    Returns:
        Tuple of (answer_text, evidence_list)
        - answer_text: Natural language answer (no raw retrieval dump)
        - evidence_list: Structured list of evidence items for the "Show Evidence" section
    """
    intent = _classify_intent(query)
    has_personas = bool(personas)

    # ── Generate natural language answer ──────────────────────────────────────
    if intent == "persona":
        answer = _synthesize_persona_answer(query, rr, personas or {}, intent)

    elif intent == "habits":
        answer = _synthesize_habits_answer(rr, personas or {})

    elif intent == "communication":
        answer = _synthesize_communication_answer(rr, personas or {})

    elif intent == "occupation":
        answer = _synthesize_occupation_answer(rr, personas or {})

    elif intent == "location":
        answer = _synthesize_location_answer(rr, personas or {})

    elif intent == "interests":
        answer = _synthesize_interests_answer(rr, personas or {})

    elif intent == "relationship":
        answer = _synthesize_relationship_answer(rr, personas or {})

    elif intent == "topics":
        answer = _synthesize_topics_answer(rr)

    else:
        # General: try persona first, then topic summaries
        if has_personas and _looks_like_persona_question(query):
            answer = _synthesize_persona_answer(query, rr, personas or {}, intent)
        else:
            answer = _synthesize_from_topics(query, rr, intent)

    # ── Build evidence list (for "Show Evidence" expander) ───────────────────
    evidence = []

    for ts in rr.topic_summaries:
        evidence.append({
            "type": "topic",
            "label": f"Topic {ts['topic_id']}",
            "conv_id": ts.get("conversation_id", "?"),
            "messages": ts["messages"],
            "score": ts["score"],
            "text": ts.get("summary", ""),
            "sub_label": ts.get("label", ""),
        })

    for rc in rr.raw_chunks:
        evidence.append({
            "type": "chunk",
            "label": f"Conversation #{rc.get('conversation_id', '?')}",
            "conv_id": rc.get("conversation_id", "?"),
            "messages": rc["messages"],
            "score": rc["score"],
            "text": rc.get("text", ""),
            "sub_label": "",
        })

    for cs in rr.checkpoint_summaries:
        evidence.append({
            "type": "checkpoint",
            "label": f"Checkpoint {cs['checkpoint_id']}",
            "conv_id": "—",
            "messages": cs["messages"],
            "score": cs["score"],
            "text": cs.get("summary", ""),
            "sub_label": "",
        })

    # Sort by score descending
    evidence.sort(key=lambda e: e["score"], reverse=True)

    return answer, evidence


def _looks_like_persona_question(query: str) -> bool:
    """Secondary check: does this query want persona info?"""
    keywords = [
        "person", "they", "their", "user", "people", "who",
        "what do", "how are", "behavior", "trait",
    ]
    q = query.lower()
    return any(k in q for k in keywords)
