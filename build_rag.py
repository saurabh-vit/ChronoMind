"""
build_rag.py — Build the RAG system for ChronoMind.

This script:
  1. Parses conversations from the data file.
  2. Detects topic segments dynamically (per-conversation).
  3. Creates fixed-interval checkpoints (conversation-aware).
  4. Chunks raw messages (conversation-aware).
  5. Builds and persists FAISS indexes.
  6. Saves topic and checkpoint data as JSON.

Usage:
    python build_rag.py [--config path/to/config.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.parser import parse_conversations, get_user_names, get_conversation_count
from src.topic_detector import detect_topics
from src.checkpoint_builder import build_checkpoints
from src.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_rag")


def main(config_path: str | None = None) -> None:
    """Run the full RAG build pipeline."""

    logger.info("=" * 60)
    logger.info("ChronoMind — RAG Build Pipeline")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(config_path)

    # Step 1: Parse conversations
    logger.info("[1/5] Parsing conversations...")
    messages = parse_conversations(config.paths.conversations_file)
    users = get_user_names(messages)
    n_convos = get_conversation_count(messages)
    logger.info("  Found %d messages in %d conversations from users: %s",
                len(messages), n_convos, ", ".join(users))

    if not messages:
        logger.error("No messages found. Check your data file.")
        sys.exit(1)

    # Step 2: Detect topics
    logger.info("[2/5] Detecting topics...")
    topics = detect_topics(messages, config)
    logger.info("  Detected %d topics.", len(topics))

    # Save topics data
    topics_path = Path(config.paths.checkpoints_dir) / "topics.json"
    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in topics], f, indent=2, ensure_ascii=False)
    logger.info("  Topics saved to %s", topics_path)

    # Step 3: Build checkpoints
    logger.info("[3/5] Building checkpoints...")
    checkpoints = build_checkpoints(messages, config)
    logger.info("  Created %d checkpoints.", len(checkpoints))

    # Save checkpoints data
    cp_path = Path(config.paths.checkpoints_dir) / "checkpoints.json"
    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump([cp.to_dict() for cp in checkpoints], f, indent=2, ensure_ascii=False)
    logger.info("  Checkpoints saved to %s", cp_path)

    # Step 4: Build vector indexes
    logger.info("[4/5] Building FAISS indexes...")
    store = VectorStore(config)

    # Build chunks index
    chunks = store.build_chunks_index(messages)
    logger.info("  Chunks index: %d chunks.", len(chunks))

    # Build topics index
    store.build_topics_index(topics)
    logger.info("  Topics index: %d topics.", len(topics))

    # Build checkpoints index
    store.build_checkpoints_index(checkpoints)
    logger.info("  Checkpoints index: %d checkpoints.", len(checkpoints))

    # Step 5: Save indexes
    logger.info("[5/5] Persisting indexes...")
    store.save()

    # Save message data for reference
    messages_path = Path(config.paths.data_dir) / "parsed_messages.json"
    with open(messages_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"index": m.index, "user": m.user, "text": m.text,
              "conversation_id": m.conversation_id}
             for m in messages],
            f, indent=2, ensure_ascii=False,
        )
    logger.info("  Messages saved to %s", messages_path)

    logger.info("=" * 60)
    logger.info("RAG build complete!")
    logger.info("  Messages: %d", len(messages))
    logger.info("  Conversations: %d", n_convos)
    logger.info("  Topics: %d", len(topics))
    logger.info("  Checkpoints: %d", len(checkpoints))
    logger.info("  Chunks: %d", len(chunks))
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChronoMind RAG system.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    args = parser.parse_args()
    main(args.config)
