"""
build_persona_drift.py — Run the Adaptive Persona Engine.

Usage:
    python build_persona_drift.py

Reads parsed conversations and outputs persona/persona_drift.json.
"""

from __future__ import annotations

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_persona_drift")

BASE_DIR = Path(__file__).resolve().parent


def main():
    logger.info("=" * 60)
    logger.info("ChronoMind v2 — Adaptive Persona Engine")
    logger.info("=" * 60)

    from src.config import load_config
    from src.parser import parse_conversations
    from src.persona_drift import analyze_persona_drift

    config = load_config()

    logger.info("[1/2] Parsing conversations...")
    data_path = Path(config.paths.data_dir) / "conversations.csv"
    messages = parse_conversations(data_path)
    logger.info("  Parsed %d messages", len(messages))

    logger.info("[2/2] Analyzing persona drift...")
    output_path = Path(config.paths.persona_dir) / "persona_drift.json"
    results = analyze_persona_drift(messages, output_path=output_path)

    logger.info("=" * 60)
    logger.info("Persona drift complete!")
    for user, data in results.items():
        logger.info(
            "  %s: %d sessions, %d drift events, mood=%s, tone=%s",
            user,
            data["total_sessions"],
            data["total_drifts"],
            data["dominant_mood"],
            data["dominant_tone"],
        )
    logger.info("  Output: %s", output_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
