"""Evaluation logic - normalization, correctness, and similarity scoring."""

import re
from difflib import SequenceMatcher
from typing import Literal

EvaluationMethod = Literal["exact", "substring", "fuzzy", "none"]


def normalize(s: str) -> str:
    """Normalize string for comparison.

    - lowercase
    - strip leading/trailing spaces
    - collapse repeated whitespace
    - remove simple punctuation (.,!?;:'\")
    """
    if not s or not isinstance(s, str):
        return ""
    t = s.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[.,!?;:'\"]+", "", t)
    return t.strip()


def fuzzy_similarity(a: str, b: str) -> float:
    """Compute similarity ratio using difflib (0.0 to 1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def evaluate_single(
    actual: str,
    expected: str,
    fuzzy_threshold: float = 0.6,
) -> dict:
    """Evaluate a single response against expected answer.

    Supports: exact match, normalized substring, fuzzy similarity threshold.

    Returns dict with:
        - correct: bool
        - normalized_expected: str
        - normalized_actual: str
        - similarity_score: float (0.0 to 1.0)
        - evaluation_method_used: "exact" | "substring" | "fuzzy" | "none"
    """
    norm_expected = normalize(expected)
    norm_actual = normalize(actual)
    similarity = fuzzy_similarity(norm_expected, norm_actual)

    if not norm_expected:
        return {
            "correct": False,
            "normalized_expected": norm_expected,
            "normalized_actual": norm_actual,
            "similarity_score": round(similarity, 4),
            "evaluation_method_used": "none",
        }

    if not norm_actual:
        return {
            "correct": False,
            "normalized_expected": norm_expected,
            "normalized_actual": norm_actual,
            "similarity_score": round(similarity, 4),
            "evaluation_method_used": "none",
        }

    # 1. Exact match
    if norm_actual == norm_expected:
        return {
            "correct": True,
            "normalized_expected": norm_expected,
            "normalized_actual": norm_actual,
            "similarity_score": round(similarity, 4),
            "evaluation_method_used": "exact",
        }

    # 2. Normalized substring: expected contained in actual
    if norm_expected in norm_actual:
        return {
            "correct": True,
            "normalized_expected": norm_expected,
            "normalized_actual": norm_actual,
            "similarity_score": round(similarity, 4),
            "evaluation_method_used": "substring",
        }

    # 3. Fuzzy similarity threshold
    if similarity >= fuzzy_threshold:
        return {
            "correct": True,
            "normalized_expected": norm_expected,
            "normalized_actual": norm_actual,
            "similarity_score": round(similarity, 4),
            "evaluation_method_used": "fuzzy",
        }

    return {
        "correct": False,
        "normalized_expected": norm_expected,
        "normalized_actual": norm_actual,
        "similarity_score": round(similarity, 4),
        "evaluation_method_used": "none",
    }


def compute_aggregates(results: list[dict]) -> dict:
    """Compute aggregate metrics from evaluation results."""
    total = len(results)
    if total == 0:
        return {
            "accuracy": 0.0,
            "avg_latency_ms": 0.0,
            "avg_similarity_score": 0.0,
        }
    correct = sum(1 for r in results if r.get("correct", False))
    total_latency = sum(r.get("latency_ms", 0) for r in results)
    total_similarity = sum(r.get("similarity_score", 0) for r in results)
    return {
        "accuracy": round(correct / total, 4),
        "avg_latency_ms": round(total_latency / total, 2),
        "avg_similarity_score": round(total_similarity / total, 4),
    }
