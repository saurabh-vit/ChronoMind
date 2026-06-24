"""
Configuration loader for ChronoMind.

Loads settings from config.json and provides typed access
to all configuration parameters used across the system.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TopicDetectionConfig:
    """Parameters controlling dynamic topic segmentation."""
    similarity_threshold: float = 0.35
    min_topic_messages: int = 3
    window_size: int = 5


@dataclass
class CheckpointConfig:
    """Parameters for fixed-interval checkpoint creation."""
    interval: int = 100


@dataclass
class ChunkingConfig:
    """Parameters for raw message chunking."""
    chunk_size: int = 20
    chunk_overlap: int = 5


@dataclass
class RetrievalConfig:
    """Parameters for the retrieval pipeline."""
    top_k_chunks: int = 5
    top_k_topics: int = 3
    top_k_checkpoints: int = 2


@dataclass
class SummarizationConfig:
    """Parameters for text summarization."""
    method: str = "sumy_lsa"
    sentences_count: int = 3


@dataclass
class PersonaConfig:
    """Parameters for persona extraction."""
    min_confidence: float = 0.4
    evidence_window: int = 5
    max_facts_per_category: int = 20
    max_total_facts: int = 50
    min_evidence_convos: int = 2  # must appear in at least N conversations


@dataclass
class PathsConfig:
    """Filesystem paths used throughout the project."""
    data_dir: str = "data"
    vector_db_dir: str = "vector_db"
    checkpoints_dir: str = "checkpoints"
    persona_dir: str = "persona"
    conversations_file: str = "data/conversations.csv"

    def resolve(self, root: Path) -> "PathsConfig":
        """Resolve all paths relative to project root."""
        self.data_dir = str(root / self.data_dir)
        self.vector_db_dir = str(root / self.vector_db_dir)
        self.checkpoints_dir = str(root / self.checkpoints_dir)
        self.persona_dir = str(root / self.persona_dir)
        self.conversations_file = str(root / self.conversations_file)
        return self


@dataclass
class AppConfig:
    """Top-level application configuration."""
    project_name: str = "ChronoMind"
    version: str = "1.0.0"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    topic_detection: TopicDetectionConfig = field(default_factory=TopicDetectionConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def ensure_directories(self) -> None:
        """Create all required output directories."""
        for dir_path in [
            self.paths.data_dir,
            self.paths.vector_db_dir,
            self.paths.checkpoints_dir,
            self.paths.persona_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


def _dict_to_dataclass(cls: type, data: Dict[str, Any]) -> Any:
    """Recursively convert a dict to a nested dataclass."""
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    for key, value in data.items():
        if key in field_types and isinstance(value, dict):
            # Resolve the actual class from type annotation string
            nested_cls = globals().get(field_types[key])
            if nested_cls is not None:
                kwargs[key] = _dict_to_dataclass(nested_cls, value)
            else:
                kwargs[key] = value
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """
    Load configuration from a JSON file.

    Args:
        config_path: Path to config.json. Defaults to project root.

    Returns:
        Fully initialized AppConfig with resolved paths.
    """
    if config_path is None:
        config_path = _PROJECT_ROOT / "config.json"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning("Config file not found at %s, using defaults.", config_path)
        cfg = AppConfig()
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cfg = _dict_to_dataclass(AppConfig, raw)

    cfg.paths.resolve(_PROJECT_ROOT)
    cfg.ensure_directories()

    logger.info("Configuration loaded: %s v%s", cfg.project_name, cfg.version)
    return cfg
