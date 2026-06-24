"""
Data models for ChronoMind.

Provides typed dataclasses for messages, topics, checkpoints,
chunks, and persona artifacts used throughout the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    """A single parsed message from a conversation."""
    index: int
    user: str
    text: str
    conversation_id: int  # which conversation block this message belongs to
    raw_line: str = ""

    def __str__(self) -> str:
        return f"{self.user}: {self.text}"


@dataclass
class Topic:
    """A detected topic segment spanning a range of messages."""
    topic_id: int
    start_message: int
    end_message: int
    conversation_id: int = -1  # -1 means spans multiple conversations
    messages: List[Message] = field(default_factory=list)
    summary: str = ""
    label: str = ""
    embedding: Optional[List[float]] = None

    @property
    def message_count(self) -> int:
        return self.end_message - self.start_message + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "start_message": self.start_message,
            "end_message": self.end_message,
            "conversation_id": self.conversation_id,
            "message_count": self.message_count,
            "summary": self.summary,
            "label": self.label,
        }


@dataclass
class Checkpoint:
    """A fixed-interval checkpoint summarizing a block of messages."""
    checkpoint_id: int
    start_message: int
    end_message: int
    messages: List[Message] = field(default_factory=list)
    summary: str = ""
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "start_message": self.start_message,
            "end_message": self.end_message,
            "summary": self.summary,
        }


@dataclass
class Chunk:
    """A raw message chunk for fine-grained retrieval."""
    chunk_id: int
    start_message: int
    end_message: int
    conversation_id: int = -1
    text: str = ""
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "start_message": self.start_message,
            "end_message": self.end_message,
            "conversation_id": self.conversation_id,
            "text": self.text,
        }


@dataclass
class PersonaTrait:
    """A single extracted persona trait with evidence."""
    trait: str
    confidence: float
    evidence: List[int] = field(default_factory=list)      # message indices
    category: str = ""                                      # habits / personal_facts / etc.
    snippets: List[str] = field(default_factory=list)       # actual matched message text
    occurrences: int = 0                                    # total match count across all messages

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trait": self.trait,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "snippets": self.snippets,
            "occurrences": self.occurrences,
        }


@dataclass
class UserPersona:
    """Complete persona profile for a single user."""
    user_name: str
    habits: List[PersonaTrait] = field(default_factory=list)
    personal_facts: List[PersonaTrait] = field(default_factory=list)
    personality_traits: List[PersonaTrait] = field(default_factory=list)
    communication_style: List[PersonaTrait] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_name": self.user_name,
            "habits": [t.to_dict() for t in self.habits],
            "personal_facts": [t.to_dict() for t in self.personal_facts],
            "personality_traits": [t.to_dict() for t in self.personality_traits],
            "communication_style": [t.to_dict() for t in self.communication_style],
        }


@dataclass
class RetrievalResult:
    """Container for multi-source retrieval results."""
    query: str
    topic_summaries: List[Dict[str, Any]] = field(default_factory=list)
    raw_chunks: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint_summaries: List[Dict[str, Any]] = field(default_factory=list)
    combined_context: str = ""
