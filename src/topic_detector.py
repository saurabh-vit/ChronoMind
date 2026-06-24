"""
Dynamic topic detection for ChronoMind.

Processes messages chronologically WITHIN each conversation,
respecting conversation boundaries as hard topic breaks.
Uses embedding cosine similarity to detect topic shifts
within longer conversations.

Each topic segment is then summarized and stored with its embedding.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from .config import AppConfig
from .embeddings import cosine_similarity, encode_texts
from .models import Message, Topic
from .parser import get_conversation_groups
from .summarizer import summarize_messages

logger = logging.getLogger(__name__)


def detect_topics(
    messages: List[Message],
    config: AppConfig,
) -> List[Topic]:
    """
    Detect topic segments in conversations.

    Each conversation is an independent dialogue. Conversation boundaries
    are ALWAYS treated as topic boundaries. Within a single conversation,
    dynamic topic detection is applied using cosine similarity on embeddings.

    Algorithm per conversation:
        1. Encode each message into a dense embedding.
        2. Maintain a running centroid for the current topic (windowed average).
        3. Compare each new message's embedding to the centroid.
        4. If cosine similarity drops below threshold, start a new topic.
        5. Summarize each completed topic segment.

    Args:
        messages: Chronologically ordered list of Message objects.
        config: Application configuration with topic detection parameters.

    Returns:
        List of Topic objects with summaries and embeddings.
    """
    if not messages:
        return []

    tc = config.topic_detection
    sc = config.summarization

    logger.info(
        "Detecting topics (threshold=%.2f, window=%d, min_messages=%d)",
        tc.similarity_threshold,
        tc.window_size,
        tc.min_topic_messages,
    )

    # Group messages by conversation
    conv_groups = get_conversation_groups(messages)
    logger.info("Processing %d conversations for topic detection.", len(conv_groups))

    # Step 1: Encode ALL messages in one batch for efficiency
    texts = [f"{m.user}: {m.text}" for m in messages]
    all_embeddings = encode_texts(
        texts,
        model_name=config.embedding_model,
        show_progress=True,
    )

    # Build index from global message index -> embedding position
    msg_idx_to_pos = {m.index: i for i, m in enumerate(messages)}

    # Step 2: Process each conversation independently
    all_topics: List[Topic] = []
    topic_id_counter = 0

    sorted_conv_ids = sorted(conv_groups.keys())

    for conv_id in sorted_conv_ids:
        conv_msgs = conv_groups[conv_id]

        if len(conv_msgs) < 2:
            # Single-message conversation is its own topic
            emb_pos = msg_idx_to_pos[conv_msgs[0].index]
            summary = f"{conv_msgs[0].user}: {conv_msgs[0].text}"
            topic = Topic(
                topic_id=topic_id_counter,
                start_message=conv_msgs[0].index,
                end_message=conv_msgs[0].index,
                conversation_id=conv_id,
                messages=conv_msgs,
                summary=summary[:200],
                label=_generate_topic_label(conv_msgs, summary),
                embedding=all_embeddings[emb_pos].tolist(),
            )
            all_topics.append(topic)
            topic_id_counter += 1
            continue

        # Get embeddings for this conversation's messages
        conv_emb_positions = [msg_idx_to_pos[m.index] for m in conv_msgs]
        conv_embeddings = all_embeddings[conv_emb_positions]

        # Dynamic topic detection within this conversation
        segments = _segment_conversation(
            conv_embeddings,
            threshold=tc.similarity_threshold,
            window_size=tc.window_size,
            min_topic_messages=tc.min_topic_messages,
        )

        # Create Topic objects for each segment
        for seg_local_indices in segments:
            seg_messages = [conv_msgs[idx] for idx in seg_local_indices]

            # Summarize the topic
            summary = summarize_messages(
                seg_messages,
                method=sc.method,
                sentences_count=sc.sentences_count,
            )

            # Compute topic embedding as mean of message embeddings
            seg_emb_positions = [conv_emb_positions[idx] for idx in seg_local_indices]
            topic_embedding = np.mean(all_embeddings[seg_emb_positions], axis=0)

            topic = Topic(
                topic_id=topic_id_counter,
                start_message=seg_messages[0].index,
                end_message=seg_messages[-1].index,
                conversation_id=conv_id,
                messages=seg_messages,
                summary=summary,
                label=_generate_topic_label(seg_messages, summary),
                embedding=topic_embedding.tolist(),
            )
            all_topics.append(topic)
            topic_id_counter += 1

    logger.info(
        "Detected %d topics across %d messages in %d conversations.",
        len(all_topics), len(messages), len(conv_groups),
    )
    return all_topics


def _segment_conversation(
    embeddings: np.ndarray,
    threshold: float,
    window_size: int,
    min_topic_messages: int,
) -> List[List[int]]:
    """
    Segment a single conversation into topic segments using cosine similarity.

    A new topic is only started when:
      1. The current segment has >= min_topic_messages messages, AND
      2. Cosine similarity to the running centroid drops below threshold.

    Trailing segments smaller than min_topic_messages // 2 are merged
    into the previous segment to avoid orphan micro-topics.

    Args:
        embeddings: Embeddings for messages in this conversation (n, dim).
        threshold: Cosine similarity threshold for topic breaks.
        window_size: Number of recent messages for centroid calculation.
        min_topic_messages: Minimum messages before allowing a topic break.

    Returns:
        List of segments, each a list of local (0-based) message indices.
    """
    n = len(embeddings)

    # Short conversation: always a single topic
    if n <= min_topic_messages:
        return [list(range(n))]

    segments: List[List[int]] = []
    current_segment: List[int] = [0]

    for i in range(1, n):
        emb = embeddings[i]

        # Compute windowed centroid from recent messages in current segment
        window_start = max(0, len(current_segment) - window_size)
        window_indices = current_segment[window_start:]
        centroid = np.mean(embeddings[window_indices], axis=0)

        sim = cosine_similarity(emb, centroid)

        # Only split if current segment is large enough
        if sim < threshold and len(current_segment) >= min_topic_messages:
            segments.append(current_segment)
            current_segment = [i]
        else:
            current_segment.append(i)

    if current_segment:
        segments.append(current_segment)

    # Merge trailing micro-segments back into the previous one
    merge_threshold = max(2, min_topic_messages // 2)
    merged = []
    for seg in segments:
        if merged and len(seg) < merge_threshold:
            # Too small — absorb into the previous segment
            merged[-1] = merged[-1] + seg
        else:
            merged.append(seg)

    # Final safety: if we somehow ended with 0 segments, return all
    return merged if merged else [list(range(n))]


def _generate_topic_label(messages: List[Message], summary: str) -> str:
    """
    Generate a short label for a topic based on its content.

    Uses the first few significant words from the summary or first message.
    """
    source = summary if summary else (messages[0].text if messages else "Unknown")

    # Extract first meaningful phrase
    words = source.split()
    label_words = []
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "i", "you", "we",
                  "they", "he", "she", "it", "to", "and", "or", "but", "in", "on",
                  "at", "for", "of", "with", "that", "this", "my", "your", "user",
                  "1:", "2:", "hi", "hello", "hey", "how", "hi!", "hello!"}

    for w in words:
        clean = w.strip(".,!?:;\"'").lower()
        if clean and clean not in stop_words and len(clean) > 1:
            label_words.append(clean.capitalize())
        if len(label_words) >= 4:
            break

    return " ".join(label_words) if label_words else f"Topic {messages[0].index}"
