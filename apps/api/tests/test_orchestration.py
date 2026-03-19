"""Tests for orchestration decision layer and strategy selection."""

import pytest

from orchestration.decision import (
    classify_query,
    decide_strategy,
    has_relevant_context,
)


# --- classify_query ---


def test_classify_query_general() -> None:
    """General messages classify as 'general'."""
    assert classify_query("Hello, how are you?") == "general"
    assert classify_query("What is 2 + 2?") == "general"
    assert classify_query("") == "general"


def test_classify_query_retrieval() -> None:
    """Messages with retrieval phrases classify as 'retrieval'."""
    assert classify_query("What does the document say about X?") == "retrieval"
    assert classify_query("According to the document, ...") == "retrieval"
    assert classify_query("Find in the documents") == "retrieval"


# --- has_relevant_context ---


def test_has_relevant_context_empty() -> None:
    """Empty chunks list has no relevant context."""
    assert has_relevant_context([]) is False


def test_has_relevant_context_with_content() -> None:
    """Chunks with content provide relevant context."""
    chunks = [{"content": "Paris is the capital of France.", "similarity": 0.8}]
    assert has_relevant_context(chunks) is True


def test_has_relevant_context_with_similarity_threshold() -> None:
    """Chunks below min_similarity are filtered out."""
    chunks = [{"content": "Some text", "similarity": 0.2}]
    assert has_relevant_context(chunks, min_similarity=0.5) is False
    assert has_relevant_context(chunks, min_similarity=0.1) is True


def test_has_relevant_context_empty_content_ignored() -> None:
    """Chunks with empty content do not count."""
    chunks = [{"content": "   ", "similarity": 0.9}]
    assert has_relevant_context(chunks) is False


def test_has_relevant_context_no_similarity_field() -> None:
    """Chunks without similarity field count (backward compat)."""
    chunks = [{"content": "Valid content"}]
    assert has_relevant_context(chunks) is True


# --- decide_strategy ---


def test_decide_strategy_direct() -> None:
    """Direct hint always returns direct."""
    assert decide_strategy("direct") == "direct"
    assert decide_strategy("direct", chunks=[{"content": "x", "similarity": 0.9}]) == "direct"


def test_decide_strategy_rag() -> None:
    """Rag hint always returns rag."""
    assert decide_strategy("rag") == "rag"
    assert decide_strategy("rag", chunks=[]) == "rag"


def test_decide_strategy_smart_with_context() -> None:
    """Smart with relevant chunks returns rag."""
    chunks = [{"content": "Paris is the capital.", "similarity": 0.85}]
    assert decide_strategy("smart", chunks=chunks) == "rag"


def test_decide_strategy_smart_fallback_no_context() -> None:
    """Smart with no chunks falls back to direct."""
    assert decide_strategy("smart", chunks=[]) == "direct"


def test_decide_strategy_smart_fallback_low_similarity() -> None:
    """Smart with low-similarity chunks falls back to direct."""
    chunks = [{"content": "Barely relevant", "similarity": 0.1}]
    assert decide_strategy("smart", chunks=chunks, min_similarity=0.5) == "direct"


def test_decide_strategy_smart_rag_when_above_threshold() -> None:
    """Smart uses RAG when similarity above threshold."""
    chunks = [{"content": "Relevant", "similarity": 0.8}]
    assert decide_strategy("smart", chunks=chunks, min_similarity=0.5) == "rag"


def test_decide_strategy_tools_fallback() -> None:
    """Tools hint falls back to direct (future extension)."""
    assert decide_strategy("tools") == "direct"
