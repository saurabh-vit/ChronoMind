"""
chatbot.py — ChronoMind v2 Chatbot Engine.

Extends v1 with:
  - Offline intent classification (TF-IDF + Logistic Regression)
  - Conflict-aware retrieval (contradiction detection + resolution)

Pipeline:
    Query
    → Intent Detection (IntentClassifier)
    → RAG Retrieval (FAISS × 3 + Persona)
    → Conflict Resolution (ConflictResolver)
    → Answer Synthesis (AnswerSynthesizer)
    → Response with metadata
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import AppConfig, load_config
from src.persona_extractor import load_personas
from src.retrieval import retrieve
from src.answer_synthesizer import synthesize_answer
from src.models import RetrievalResult, UserPersona
from src.vector_store import VectorStore
from src.intent_classifier import get_intent_classifier, INTENT_LABELS, INTENT_DESCRIPTIONS
from src.conflict_resolver import resolve as resolve_conflicts

logger = logging.getLogger(__name__)


class ChronoBot:
    """
    ChronoMind v2 Chatbot.

    Adds intent classification and conflict resolution on top of the v1
    RAG + Persona pipeline. All components are fully offline.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.vector_store: Optional[VectorStore] = None
        self.personas: Dict[str, UserPersona] = {}
        self._initialized = False
        self._intent_clf = get_intent_classifier()

    def initialize(self) -> bool:
        """Load vector indexes and persona data."""
        logger.info("Initializing ChronoBot...")

        self.vector_store = VectorStore(self.config)
        indexes_loaded = self.vector_store.load()

        if not indexes_loaded:
            logger.warning("FAISS indexes not found. Run 'python build_rag.py' first.")
            return False

        self.personas = load_personas(self.config.paths.persona_dir)
        if not self.personas:
            logger.warning("Personas not found. Run 'python build_persona.py' first.")

        self._initialized = True
        logger.info("ChronoBot v2 initialized. %d personas available.", len(self.personas))
        return True

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.vector_store is not None

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask the chatbot a question.

        Returns a dict with:
          - 'answer'         : str — synthesized natural-language answer (conflict-resolved)
          - 'evidence'       : list — structured evidence items for "Show Evidence" UI
          - 'retrieval'      : RetrievalResult — raw retrieval result
          - 'intent'         : dict — intent classification result
          - 'conflicts'      : dict — conflict resolution result
        """
        if not self.is_ready:
            return {
                "answer": "The chatbot is not initialized. Please build the RAG system first.",
                "evidence": [],
                "retrieval": None,
                "intent": {"intent": "unknown", "confidence": 0.0, "all_scores": {}},
                "conflicts": {"has_conflicts": False, "n_conflicts": 0, "conflicts": []},
            }

        if not question.strip():
            return {
                "answer": "Please ask a question about the conversation participants.",
                "evidence": [],
                "retrieval": None,
                "intent": {"intent": "unknown", "confidence": 0.0, "all_scores": {}},
                "conflicts": {"has_conflicts": False, "n_conflicts": 0, "conflicts": []},
            }

        # Step 1: Intent classification
        intent_result = self._intent_clf.predict(question)

        # Step 2: Retrieve relevant evidence from all FAISS indexes
        rr = retrieve(
            query=question,
            vector_store=self.vector_store,
            config=self.config,
            personas=self.personas,
        )

        # Step 3: Synthesize a natural-language answer
        answer, evidence = synthesize_answer(
            query=question,
            rr=rr,
            personas=self.personas if self.personas else None,
        )

        # Step 4: Conflict resolution on raw retrieved chunks
        conflict_result = resolve_conflicts(
            evidence_items=evidence,
            original_answer=answer,
            query=question,
        )

        # Use conflict-augmented answer if conflicts were found
        final_answer = conflict_result["augmented_answer"]

        return {
            "answer":    final_answer,
            "evidence":  evidence,
            "retrieval": rr,
            "intent":    intent_result,
            "conflicts": conflict_result,
        }

    def get_suggested_questions(self) -> List[str]:
        """Return a list of suggested questions for the user."""
        return [
            "What kind of person is User 1?",
            "What hobbies and habits do they have?",
            "What jobs or occupations are mentioned?",
            "How do they communicate — short or detailed messages?",
            "What cities or locations are discussed?",
            "Do they use emojis frequently?",
            "What personality traits come through in conversations?",
            "Are they introverted or extroverted?",
            "What topics come up most often?",
            "Tell me about their relationships and family.",
        ]
