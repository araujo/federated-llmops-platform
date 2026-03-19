"""Tests for evals evaluation logic."""

import sys
from pathlib import Path

# Add repo root so evals can be imported
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pytest

from evals.evaluation import (
    compute_aggregates,
    evaluate_single,
    fuzzy_similarity,
    normalize,
)


# --- normalize ---


def test_normalize_lowercase() -> None:
    """Normalization lowercases."""
    assert normalize("Paris") == "paris"
    assert normalize("BLUE") == "blue"


def test_normalize_strip_spaces() -> None:
    """Normalization strips leading/trailing spaces."""
    assert normalize("  Paris  ") == "paris"
    assert normalize("\n\t4\t\n") == "4"


def test_normalize_collapse_whitespace() -> None:
    """Normalization collapses repeated whitespace."""
    assert normalize("Paris   is   great") == "paris is great"


def test_normalize_removes_punctuation() -> None:
    """Normalization removes simple punctuation."""
    assert normalize("Paris.") == "paris"
    assert normalize("The answer is 4!") == "the answer is 4"
    assert normalize("Blue?") == "blue"


def test_normalize_empty() -> None:
    """Empty or invalid input returns empty string."""
    assert normalize("") == ""
    assert normalize("   ") == ""
    assert normalize(None) == ""  # type: ignore[arg-type]


# --- fuzzy_similarity ---


def test_fuzzy_similarity_exact() -> None:
    """Exact match gives 1.0."""
    assert fuzzy_similarity("paris", "paris") == 1.0


def test_fuzzy_similarity_empty() -> None:
    """Both empty gives 1.0; one empty gives 0.0."""
    assert fuzzy_similarity("", "") == 1.0
    assert fuzzy_similarity("paris", "") == 0.0
    assert fuzzy_similarity("", "paris") == 0.0


def test_fuzzy_similarity_near_match() -> None:
    """Near matches get high similarity."""
    assert fuzzy_similarity("paris", "pares") > 0.7
    assert fuzzy_similarity("blue", "bleu") > 0.6


# --- evaluate_single ---


def test_evaluate_exact_match() -> None:
    """Exact match after normalization is correct."""
    r = evaluate_single("Paris", "Paris")
    assert r["correct"] is True
    assert r["evaluation_method_used"] == "exact"
    assert r["similarity_score"] == 1.0


def test_evaluate_case_insensitive() -> None:
    """Case-insensitive match is correct (exact after normalize)."""
    r = evaluate_single("PARIS", "paris")
    assert r["correct"] is True
    assert r["evaluation_method_used"] == "exact"


def test_evaluate_punctuation_differences() -> None:
    """Punctuation differences are normalized away."""
    r = evaluate_single("Paris.", "Paris")
    assert r["correct"] is True
    assert r["evaluation_method_used"] == "exact"


def test_evaluate_substring_match() -> None:
    """Expected substring in actual is correct."""
    r = evaluate_single("The capital of France is Paris.", "Paris")
    assert r["correct"] is True
    assert r["evaluation_method_used"] == "substring"


def test_evaluate_fuzzy_near_match() -> None:
    """Fuzzy near-match above threshold is correct (typo in short answer)."""
    # "Pares" vs "Paris" - similar length, ratio ~0.8
    r = evaluate_single("Pares", "Paris", fuzzy_threshold=0.5)
    assert r["correct"] is True
    assert r["evaluation_method_used"] == "fuzzy"


def test_evaluate_empty_response() -> None:
    """Empty actual is always wrong."""
    r = evaluate_single("", "Paris")
    assert r["correct"] is False
    assert r["evaluation_method_used"] == "none"
    assert r["normalized_actual"] == ""


def test_evaluate_empty_expected() -> None:
    """Empty expected is always wrong."""
    r = evaluate_single("Paris", "")
    assert r["correct"] is False
    assert r["evaluation_method_used"] == "none"


def test_evaluate_stores_all_fields() -> None:
    """Result includes normalized_expected, normalized_actual, similarity_score, evaluation_method_used."""
    r = evaluate_single("The answer is 4.", "4")
    assert "normalized_expected" in r
    assert "normalized_actual" in r
    assert "similarity_score" in r
    assert "evaluation_method_used" in r
    assert r["normalized_expected"] == "4"
    assert "4" in r["normalized_actual"]


# --- compute_aggregates ---


def test_compute_aggregates_empty() -> None:
    """Empty results give zero aggregates."""
    agg = compute_aggregates([])
    assert agg["accuracy"] == 0.0
    assert agg["avg_latency_ms"] == 0.0
    assert agg["avg_similarity_score"] == 0.0


def test_compute_aggregates() -> None:
    """Aggregates compute accuracy, avg latency, avg similarity."""
    results = [
        {"correct": True, "latency_ms": 100, "similarity_score": 1.0},
        {"correct": False, "latency_ms": 200, "similarity_score": 0.3},
    ]
    agg = compute_aggregates(results)
    assert agg["accuracy"] == 0.5
    assert agg["avg_latency_ms"] == 150.0
    assert agg["avg_similarity_score"] == 0.65
