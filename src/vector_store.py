"""
FAISS vector store for ChronoMind.

Manages three separate FAISS indexes:
  1. Raw message chunks (conversation-aware)
  2. Topic summaries
  3. Checkpoint summaries

Supports persistence (save/load) and similarity search.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from .config import AppConfig
from .embeddings import encode_texts
from .models import Checkpoint, Chunk, Message, Topic

logger = logging.getLogger(__name__)


@dataclass
class VectorIndex:
    """Wrapper around a single FAISS index with metadata."""
    name: str
    index: Optional[faiss.Index] = None
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    dimension: int = 384

    def build(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Build the FAISS index from embeddings and metadata."""
        if len(embeddings) == 0:
            logger.warning("No embeddings provided for index '%s'.", self.name)
            return

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine on normalized vecs)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata

        logger.info("Built index '%s' with %d vectors (dim=%d).", self.name, len(metadata), self.dimension)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Search the index for nearest neighbors.

        Args:
            query_embedding: Query vector (1, dim) or (dim,).
            top_k: Number of results to return.

        Returns:
            List of (score, metadata_dict) tuples, sorted by relevance.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append((float(score), self.metadata[idx]))

        return results

    def save(self, directory: str | Path) -> None:
        """Persist index and metadata to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.index is not None:
            faiss.write_index(self.index, str(directory / f"{self.name}.faiss"))

        with open(directory / f"{self.name}_meta.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        logger.info("Saved index '%s' to %s", self.name, directory)

    def load(self, directory: str | Path) -> bool:
        """Load index and metadata from disk."""
        directory = Path(directory)
        faiss_path = directory / f"{self.name}.faiss"
        meta_path = directory / f"{self.name}_meta.json"

        if not faiss_path.exists() or not meta_path.exists():
            logger.warning("Index files not found for '%s' in %s", self.name, directory)
            return False

        self.index = faiss.read_index(str(faiss_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        logger.info("Loaded index '%s' (%d vectors) from %s", self.name, self.index.ntotal, directory)
        return True


class VectorStore:
    """
    Manages the three FAISS indexes used by ChronoMind.

    Indexes:
        chunks: Raw message chunks (fine-grained, conversation-aware)
        topics: Topic summary embeddings (topic-level)
        checkpoints: Checkpoint summary embeddings (coarse-grained)
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.chunks_index = VectorIndex(name="chunks", dimension=config.embedding_dimension)
        self.topics_index = VectorIndex(name="topics", dimension=config.embedding_dimension)
        self.checkpoints_index = VectorIndex(name="checkpoints", dimension=config.embedding_dimension)

    def build_chunks_index(self, messages: List[Message]) -> List[Chunk]:
        """
        Chunk messages and build the raw chunks FAISS index.

        Chunks respect conversation boundaries. Each chunk's text is formatted
        as a readable dialogue block for high-quality retrieval context.

        Args:
            messages: All messages.

        Returns:
            List of Chunk objects.
        """
        chunk_size = self.config.chunking.chunk_size
        chunk_overlap = self.config.chunking.chunk_overlap

        chunks: List[Chunk] = []
        chunk_id = 0

        from .parser import get_conversation_groups
        conv_groups = get_conversation_groups(messages)

        for conv_id in sorted(conv_groups.keys()):
            conv_msgs = conv_groups[conv_id]

            start = 0
            while start < len(conv_msgs):
                end = min(start + chunk_size, len(conv_msgs))
                segment = conv_msgs[start:end]

                # Rich text formatting for retrieval quality
                lines = [f"[Conversation #{conv_id}]"]
                for m in segment:
                    lines.append(f"{m.user}: {m.text}")
                text = "\n".join(lines)

                chunk = Chunk(
                    chunk_id=chunk_id,
                    start_message=segment[0].index,
                    end_message=segment[-1].index,
                    conversation_id=conv_id,
                    text=text,
                )
                chunks.append(chunk)
                chunk_id += 1

                step = max(1, chunk_size - chunk_overlap)
                start += step

        # Encode all chunk texts
        texts = [c.text for c in chunks]
        embeddings = encode_texts(texts, model_name=self.config.embedding_model, show_progress=True)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb.tolist()

        metadata = [c.to_dict() for c in chunks]
        self.chunks_index.build(embeddings, metadata)

        logger.info("Built chunks index: %d chunks.", len(chunks))
        return chunks

    def build_topics_index(self, topics: List[Topic]) -> None:
        """Build the topics FAISS index from detected topics."""
        if not topics:
            return

        embeddings = np.array([t.embedding for t in topics], dtype=np.float32)
        metadata = [t.to_dict() for t in topics]
        self.topics_index.build(embeddings, metadata)

    def build_checkpoints_index(self, checkpoints: List[Checkpoint]) -> None:
        """Build the checkpoints FAISS index."""
        if not checkpoints:
            return

        embeddings = np.array([cp.embedding for cp in checkpoints], dtype=np.float32)
        metadata = [cp.to_dict() for cp in checkpoints]
        self.checkpoints_index.build(embeddings, metadata)

    def save(self) -> None:
        """Persist all indexes to disk."""
        db_dir = self.config.paths.vector_db_dir
        self.chunks_index.save(db_dir)
        self.topics_index.save(db_dir)
        self.checkpoints_index.save(db_dir)
        logger.info("All indexes saved to %s", db_dir)

    def load(self) -> bool:
        """Load all indexes from disk."""
        db_dir = self.config.paths.vector_db_dir
        c = self.chunks_index.load(db_dir)
        t = self.topics_index.load(db_dir)
        k = self.checkpoints_index.load(db_dir)
        success = c and t and k
        if success:
            logger.info("All indexes loaded from %s", db_dir)
        return success

    def search(
        self,
        query: str,
        top_k_chunks: Optional[int] = None,
        top_k_topics: Optional[int] = None,
        top_k_checkpoints: Optional[int] = None,
    ) -> Dict[str, List[Tuple[float, Dict[str, Any]]]]:
        """
        Search all three indexes for a given query.

        Args:
            query: Natural language query.
            top_k_chunks: Override for number of chunk results.
            top_k_topics: Override for number of topic results.
            top_k_checkpoints: Override for number of checkpoint results.

        Returns:
            Dict with keys 'chunks', 'topics', 'checkpoints',
            each mapping to a list of (score, metadata) tuples.
        """
        rc = self.config.retrieval

        query_emb = encode_texts(query, model_name=self.config.embedding_model)

        return {
            "chunks": self.chunks_index.search(
                query_emb, top_k=top_k_chunks or rc.top_k_chunks
            ),
            "topics": self.topics_index.search(
                query_emb, top_k=top_k_topics or rc.top_k_topics
            ),
            "checkpoints": self.checkpoints_index.search(
                query_emb, top_k=top_k_checkpoints or rc.top_k_checkpoints
            ),
        }
