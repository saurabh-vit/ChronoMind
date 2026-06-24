"""
Embedding engine for ChronoMind.

Wraps SentenceTransformers to provide a singleton embedding model
with batch encoding and caching support.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Module-level singleton
_model = None
_model_name: Optional[str] = None


def get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get or initialize the singleton SentenceTransformer model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded SentenceTransformer model.
    """
    global _model, _model_name

    if _model is None or _model_name != model_name:
        logger.info("Loading embedding model: %s", model_name)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name)
        _model_name = model_name
        logger.info("Embedding model loaded successfully.")

    return _model


def encode_texts(
    texts: Union[str, List[str]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """
    Encode one or more texts into dense vector embeddings.

    Args:
        texts: Single string or list of strings to encode.
        model_name: HuggingFace model identifier.
        batch_size: Batch size for encoding.
        show_progress: Whether to show a progress bar.
        normalize: Whether to L2-normalize the output vectors.

    Returns:
        numpy array of shape (n, embedding_dim).
    """
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return np.array([])

    model = get_model(model_name)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )

    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Assumes vectors may or may not be normalized.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity score in [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))
