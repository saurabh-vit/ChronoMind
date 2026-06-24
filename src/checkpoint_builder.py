"""
Checkpoint builder for ChronoMind.

Creates independent fixed-interval checkpoints (every N messages)
with summaries and embeddings for coarse-grained retrieval.

Checkpoints respect conversation boundaries — a checkpoint will
never blend messages from different conversations.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from .config import AppConfig
from .embeddings import encode_texts
from .models import Checkpoint, Message
from .summarizer import summarize_messages

logger = logging.getLogger(__name__)


def build_checkpoints(
    messages: List[Message],
    config: AppConfig,
) -> List[Checkpoint]:
    """
    Create fixed-interval checkpoints from a message stream.

    Each checkpoint covers up to `config.checkpoints.interval` messages.
    Checkpoints are built sequentially and respect conversation boundaries:
    a checkpoint boundary is inserted whenever the conversation_id changes,
    even if the interval hasn't been reached yet.

    Args:
        messages: Chronologically ordered messages.
        config: Application configuration.

    Returns:
        List of Checkpoint objects.
    """
    if not messages:
        return []

    interval = config.checkpoints.interval
    sc = config.summarization

    logger.info("Building checkpoints (interval=%d messages).", interval)

    checkpoints: List[Checkpoint] = []
    cp_id = 0
    current_segment: List[Message] = []
    current_start = 0

    for msg in messages:
        # Check if we need to start a new checkpoint
        should_break = False

        if current_segment:
            # Break at conversation boundary
            if msg.conversation_id != current_segment[-1].conversation_id:
                should_break = True
            # Break at interval
            elif len(current_segment) >= interval:
                should_break = True

        if should_break and current_segment:
            # Finalize current checkpoint
            summary = summarize_messages(
                current_segment,
                method=sc.method,
                sentences_count=sc.sentences_count,
            )
            checkpoint = Checkpoint(
                checkpoint_id=cp_id,
                start_message=current_segment[0].index,
                end_message=current_segment[-1].index,
                messages=current_segment,
                summary=summary,
            )
            checkpoints.append(checkpoint)
            cp_id += 1
            current_segment = []

        current_segment.append(msg)

    # Finalize last segment
    if current_segment:
        summary = summarize_messages(
            current_segment,
            method=sc.method,
            sentences_count=sc.sentences_count,
        )
        checkpoint = Checkpoint(
            checkpoint_id=cp_id,
            start_message=current_segment[0].index,
            end_message=current_segment[-1].index,
            messages=current_segment,
            summary=summary,
        )
        checkpoints.append(checkpoint)

    # Compute embeddings for all checkpoint summaries in batch
    summary_texts = [cp.summary for cp in checkpoints]
    if summary_texts:
        embeddings = encode_texts(
            summary_texts,
            model_name=config.embedding_model,
        )
        for cp, emb in zip(checkpoints, embeddings):
            cp.embedding = emb.tolist()

    logger.info("Created %d checkpoints.", len(checkpoints))
    return checkpoints
