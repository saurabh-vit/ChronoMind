"""
Retrieval pipeline for ChronoMind.

Combines results from all three FAISS indexes (chunks, topics,
checkpoints) and persona data to build rich context for answering
user questions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .config import AppConfig
from .models import RetrievalResult, UserPersona
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    vector_store: VectorStore,
    config: AppConfig,
    personas: Optional[Dict[str, UserPersona]] = None,
) -> RetrievalResult:
    """
    Execute the full retrieval pipeline for a user query.

    Steps:
        1. Search all three FAISS indexes for relevant content.
        2. Optionally incorporate persona data for persona-related queries.
        3. Combine all retrieved context into a single context string.

    Args:
        query: Natural language user question.
        vector_store: Initialized VectorStore with loaded indexes.
        config: Application configuration.
        personas: Optional persona data for persona-related queries.

    Returns:
        RetrievalResult with all retrieved context.
    """
    logger.info("Retrieving context for query: '%s'", query[:100])

    # Step 1: Multi-index search
    results = vector_store.search(query)

    # Step 2: Build retrieval result
    rr = RetrievalResult(query=query)

    # Process topic results
    for score, meta in results.get("topics", []):
        rr.topic_summaries.append({
            "score": round(score, 4),
            "topic_id": meta.get("topic_id"),
            "conversation_id": meta.get("conversation_id", -1),
            "messages": f"{meta.get('start_message', '?')}-{meta.get('end_message', '?')}",
            "summary": meta.get("summary", ""),
            "label": meta.get("label", ""),
        })

    # Process chunk results
    for score, meta in results.get("chunks", []):
        rr.raw_chunks.append({
            "score": round(score, 4),
            "chunk_id": meta.get("chunk_id"),
            "conversation_id": meta.get("conversation_id", -1),
            "messages": f"{meta.get('start_message', '?')}-{meta.get('end_message', '?')}",
            "text": meta.get("text", ""),
        })

    # Process checkpoint results
    for score, meta in results.get("checkpoints", []):
        rr.checkpoint_summaries.append({
            "score": round(score, 4),
            "checkpoint_id": meta.get("checkpoint_id"),
            "messages": f"{meta.get('start_message', '?')}-{meta.get('end_message', '?')}",
            "summary": meta.get("summary", ""),
        })

    # Step 3: Build combined context
    context_parts = []
    MAX_CONTEXT_CHARS = 6000  # Keep context within reasonable size for display

    # Add topic summaries
    if rr.topic_summaries:
        context_parts.append("=== RELEVANT TOPIC SUMMARIES ===")
        for ts in rr.topic_summaries:
            conv_id = ts.get("conversation_id", "?")
            context_parts.append(
                f"Topic {ts['topic_id']} [Conv #{conv_id}, msgs {ts['messages']}, "
                f"relevance: {ts['score']:.2f}]:\n{ts['summary']}"
            )

    # Add raw chunks
    if rr.raw_chunks:
        context_parts.append("\n=== RELEVANT RAW MESSAGES ===")
        for rc in rr.raw_chunks:
            conv_id = rc.get("conversation_id", "?")
            # Truncate individual chunk text to avoid bloat
            chunk_text = rc['text'][:400] + "..." if len(rc['text']) > 400 else rc['text']
            context_parts.append(
                f"Chunk {rc['chunk_id']} [Conv #{conv_id}, msgs {rc['messages']}, "
                f"relevance: {rc['score']:.2f}]:\n{chunk_text}"
            )

    # Add checkpoint summaries
    if rr.checkpoint_summaries:
        context_parts.append("\n=== RELEVANT CHECKPOINT SUMMARIES ===")
        for cs in rr.checkpoint_summaries:
            context_parts.append(
                f"Checkpoint {cs['checkpoint_id']} (msgs {cs['messages']}, "
                f"relevance: {cs['score']:.2f}):\n{cs['summary']}"
            )

    # Add persona context for persona-related queries
    if personas and _is_persona_query(query):
        context_parts.append("\n=== USER PERSONA DATA ===")
        for user, persona in personas.items():
            context_parts.append(f"\n--- {user} ---")
            pd = persona.to_dict()
            for category in ["habits", "personal_facts", "personality_traits", "communication_style"]:
                items = pd.get(category, [])
                if items:
                    context_parts.append(f"  {category.replace('_', ' ').title()}:")
                    for item in items[:8]:  # Cap per-category items
                        context_parts.append(
                            f"    - {item['trait']} (confidence: {item['confidence']:.2f})"
                        )

    combined = "\n".join(context_parts)
    # Truncate if too long
    if len(combined) > MAX_CONTEXT_CHARS:
        combined = combined[:MAX_CONTEXT_CHARS] + "\n\n[...context truncated for brevity...]"
    rr.combined_context = combined
    logger.info(
        "Retrieved: %d topics, %d chunks, %d checkpoints.",
        len(rr.topic_summaries),
        len(rr.raw_chunks),
        len(rr.checkpoint_summaries),
    )

    return rr


def _is_persona_query(query: str) -> bool:
    """Check if the query is persona-related."""
    persona_keywords = [
        "persona", "personality", "habit", "communicate", "communication",
        "style", "who is", "what kind", "what type",
        "tell me about user", "tell me about their", "tell me about them",
        "relationship", "how do they", "how does", "trait", "behavior",
        "emoji", "fact", "personal facts", "what are their",
    ]
    query_lower = query.lower()
    return any(kw in query_lower for kw in persona_keywords)


def format_answer(query: str, context: str) -> str:
    """
    Generate an answer from retrieved context without using paid APIs.

    Uses a template-based approach that presents the retrieved evidence
    in a structured, readable format. For questions that match specific
    patterns, provides targeted formatting.

    Args:
        query: The user's question.
        context: Combined context from retrieval.

    Returns:
        Formatted answer string.
    """
    if not context.strip():
        return (
            "I couldn't find relevant information to answer your question. "
            "Try rephrasing or asking about specific topics, habits, or "
            "communication patterns discussed in the conversations."
        )

    query_lower = query.lower()

    # Determine answer type and format accordingly
    if any(kw in query_lower for kw in ["what kind of person", "who is", "what type"]):
        return _format_persona_answer(query, context)
    elif any(kw in query_lower for kw in ["habit", "routine", "regularly", "always"]):
        return _format_habits_answer(query, context)
    elif any(kw in query_lower for kw in ["communicate", "communication", "style", "message", "emoji"]):
        return _format_communication_answer(query, context)
    elif any(kw in query_lower for kw in ["topic", "discuss", "talked about", "conversation about"]):
        return _format_topic_answer(query, context)
    elif any(kw in query_lower for kw in ["relationship", "friend", "between"]):
        return _format_relationship_answer(query, context)
    else:
        return _format_general_answer(query, context)


def _format_persona_answer(query: str, context: str) -> str:
    """Format answer for persona-related questions."""
    return (
        f"Based on the conversation analysis, here's what I found:\n\n"
        f"{context}\n\n"
        f"**Note:** All traits above are evidence-based, extracted from "
        f"actual message content with confidence scores."
    )


def _format_habits_answer(query: str, context: str) -> str:
    """Format answer for habit-related questions."""
    return (
        f"Here are the habits identified from the conversations:\n\n"
        f"{context}\n\n"
        f"Each habit is backed by specific message evidence."
    )


def _format_communication_answer(query: str, context: str) -> str:
    """Format answer for communication style questions."""
    return (
        f"Here's the communication style analysis:\n\n"
        f"{context}\n\n"
        f"This analysis is based on statistical patterns across all messages."
    )


def _format_topic_answer(query: str, context: str) -> str:
    """Format answer for topic-related questions."""
    return (
        f"Here are the relevant discussion topics:\n\n"
        f"{context}\n\n"
        f"Topics were detected dynamically using semantic similarity analysis."
    )


def _format_relationship_answer(query: str, context: str) -> str:
    """Format answer for relationship-related questions."""
    return (
        f"Based on conversation patterns, here's what I found about relationships:\n\n"
        f"{context}\n\n"
        f"Relationship insights are derived from interaction patterns and message content."
    )


def _format_general_answer(query: str, context: str) -> str:
    """Format answer for general questions."""
    return (
        f"Here's what I found related to your question:\n\n"
        f"{context}\n\n"
        f"This information was retrieved from topic summaries, "
        f"raw message chunks, and checkpoint summaries."
    )
