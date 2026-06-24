"""
build_persona.py — Extract user personas from conversations.

This script:
  1. Parses conversations from the data file.
  2. Extracts evidence-based persona traits per user label.
  3. Saves structured persona JSON files.

Usage:
    python build_persona.py [--config path/to/config.json]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.parser import parse_conversations, get_user_names, get_conversation_count
from src.persona_extractor import extract_persona, save_personas

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_persona")


def main(config_path: str | None = None) -> None:
    """Run the persona extraction pipeline."""

    logger.info("=" * 60)
    logger.info("ChronoMind — Persona Extraction Pipeline")
    logger.info("=" * 60)

    config = load_config(config_path)

    # Step 1: Parse conversations
    logger.info("[1/2] Parsing conversations...")
    messages = parse_conversations(config.paths.conversations_file)
    users = get_user_names(messages)
    n_convos = get_conversation_count(messages)
    logger.info("  Found %d messages in %d conversations from users: %s",
                len(messages), n_convos, ", ".join(users))

    if not messages:
        logger.error("No messages found. Check your data file.")
        sys.exit(1)

    # Step 2: Extract personas
    logger.info("[2/2] Extracting personas...")
    personas = extract_persona(messages, config)

    for user, persona in personas.items():
        pd = persona.to_dict()
        logger.info("  %s:", user)
        logger.info("    Habits: %d", len(pd["habits"]))
        logger.info("    Personal Facts: %d", len(pd["personal_facts"]))
        logger.info("    Personality Traits: %d", len(pd["personality_traits"]))
        logger.info("    Communication Style: %d", len(pd["communication_style"]))

    # Save
    save_personas(personas, config.paths.persona_dir)

    logger.info("=" * 60)
    logger.info("Persona extraction complete!")
    logger.info("  Users analyzed: %d", len(personas))
    logger.info("  Conversations covered: %d", n_convos)
    logger.info("  Output: %s", config.paths.persona_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract user personas.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    args = parser.parse_args()
    main(args.config)
