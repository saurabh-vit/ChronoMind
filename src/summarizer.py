"""
Text summarization for ChronoMind.

Provides extractive summarization using Sumy (LSA method)
as the lightweight, local-only summarization engine.
"""

from __future__ import annotations

import logging
from typing import List

from .models import Message

logger = logging.getLogger(__name__)

# Lazy initialization flag
_nltk_ready = False


def _ensure_nltk() -> None:
    """Download required NLTK data if not already present."""
    global _nltk_ready
    if _nltk_ready:
        return

    import nltk
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
    _nltk_ready = True


def summarize_text(
    text: str,
    method: str = "sumy_lsa",
    sentences_count: int = 3,
) -> str:
    """
    Generate an extractive summary of the given text.

    Adaptively reduces sentences_count if the text is too short for
    the requested number of sentences (avoids Sumy LSA warnings/failures
    on short conversation blocks).

    Args:
        text: Input text to summarize.
        method: Summarization method. Supported: 'sumy_lsa', 'sumy_text_rank'.
        sentences_count: Number of sentences in the summary (capped to text length).

    Returns:
        Summary string. Returns truncated text if summarization fails.
    """
    text = text.strip()
    if not text:
        return ""

    # Very short text — return as-is (no point summarizing)
    if len(text) < 80:
        return text[:500]

    _ensure_nltk()

    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.nlp.stemmers import Stemmer
        from sumy.utils import get_stop_words

        language = "english"
        stemmer = Stemmer(language)
        tokenizer = Tokenizer(language)

        parser = PlaintextParser.from_string(text, tokenizer)

        # Adapt sentence count: don't request more sentences than available
        doc_sentence_count = len(list(parser.document.sentences))
        if doc_sentence_count == 0:
            return _fallback_summary(text, sentences_count)

        # Must have at least 2x the words as sentences for LSA to work reliably
        word_count = len(text.split())
        # Cap sentences to what the document actually has
        effective_count = min(sentences_count, max(1, doc_sentence_count // 2))
        effective_count = min(effective_count, max(1, word_count // 8))
        effective_count = max(1, effective_count)

        if method == "sumy_text_rank":
            from sumy.summarizers.text_rank import TextRankSummarizer
            summarizer = TextRankSummarizer(stemmer)
        else:
            from sumy.summarizers.lsa import LsaSummarizer
            summarizer = LsaSummarizer(stemmer)

        summarizer.stop_words = get_stop_words(language)

        sentences = summarizer(parser.document, effective_count)
        summary = " ".join(str(s) for s in sentences)

        if summary.strip():
            return summary.strip()

    except Exception as e:
        logger.debug("Summarization failed (%s): %s", method, e)

    # Fallback: return first N sentences by splitting on sentence boundaries
    return _fallback_summary(text, sentences_count)


def summarize_messages(
    messages: List[Message],
    method: str = "sumy_lsa",
    sentences_count: int = 3,
) -> str:
    """
    Summarize a list of messages.

    Args:
        messages: List of Message objects.
        method: Summarization method.
        sentences_count: Number of sentences in output.

    Returns:
        Summary string.
    """
    if not messages:
        return ""

    text = "\n".join(f"{m.user}: {m.text}" for m in messages)
    return summarize_text(text, method=method, sentences_count=sentences_count)


def _fallback_summary(text: str, n_sentences: int = 3) -> str:
    """Simple fallback: return first N sentences."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    selected = sentences[:n_sentences]
    result = " ".join(selected)
    if len(result) > 500:
        result = result[:497] + "..."
    return result
