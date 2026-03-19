"""Decision layer - separates strategy selection from execution."""

from typing import Literal

Strategy = Literal["direct", "rag", "tools"]
QueryType = Literal["general", "retrieval", "tool"]
StrategyHint = Literal["direct", "rag", "smart", "tools"]


def classify_query(message: str) -> QueryType:
    """Classify the query type for routing decisions.

    Returns:
        "general" - general conversation, no retrieval needed
        "retrieval" - likely needs document context
        "tool" - may need tool invocation (future)

    For now uses simple heuristics. Can be extended with LLM-based
    classification for LangGraph integration.
    """
    if not message or not isinstance(message, str):
        return "general"
    msg = message.strip().lower()
    # Heuristics for retrieval intent (extensible)
    retrieval_phrases = (
        "what does the document",
        "what is in the document",
        "according to the document",
        "in the document",
        "from the documents",
        "search for",
        "find in",
    )
    if any(p in msg for p in retrieval_phrases):
        return "retrieval"
    # Future: tool-related patterns (calculator, etc.)
    return "general"


def has_relevant_context(
    chunks: list[dict],
    min_similarity: float = 0.0,
) -> bool:
    """Determine if retrieved chunks provide relevant context.

    Args:
        chunks: List of chunk dicts (may include 'similarity' key).
        min_similarity: Minimum similarity score for a chunk to count.
                        0.0 = any chunk with content counts.

    Returns:
        True if we have usable context for RAG.
    """
    if not chunks:
        return False
    for c in chunks:
        content = c.get("content", "") or ""
        if not content.strip():
            continue
        sim = c.get("similarity")
        if sim is not None:
            if float(sim) >= min_similarity:
                return True
        else:
            # No similarity field - any content counts (backward compat)
            return True
    return False


def decide_strategy(
    strategy_hint: StrategyHint,
    chunks: list[dict] | None = None,
    min_similarity: float = 0.0,
) -> Strategy:
    """Decide which execution strategy to use.

    Args:
        strategy_hint: Requested strategy from API route.
            - "direct": always direct LLM
            - "rag": always RAG (even with empty context)
            - "smart": RAG if relevant context exists, else direct
            - "tools": future tool-based (for now falls back to direct)
        chunks: Retrieved chunks (for "smart" decision). None if not retrieved.
        min_similarity: Threshold for "relevant" context in smart mode.

    Returns:
        Strategy to execute: "direct", "rag", or "tools".
    """
    if strategy_hint == "direct":
        return "direct"
    if strategy_hint == "rag":
        return "rag"
    if strategy_hint == "tools":
        # Future: tool-based strategy; for now direct
        return "direct"
    if strategy_hint == "smart":
        if chunks is not None and has_relevant_context(chunks, min_similarity):
            return "rag"
        return "direct"
    # Unknown hint -> direct
    return "direct"
