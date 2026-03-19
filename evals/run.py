"""Evaluation runner - executes against the real chat pipeline via API."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from evals.evaluation import compute_aggregates, evaluate_single

# Default API base URL (override with EVALS_API_URL env if needed)
DEFAULT_API_URL = "http://localhost:8000"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _fetch_prompt_metadata(
    api_url: str,
    api_key: str | None,
    prompt_name: str = "rag_chat",
    prompt_version: str | None = None,
) -> tuple[str, str, str]:
    """Fetch prompt_name, prompt_version, model from /prompts API."""
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key

    with httpx.Client(timeout=10.0) as client:
        r = client.get(
            f"{api_url}/prompts/{prompt_name}",
            headers=headers or None,
        )
        r.raise_for_status()
        data = r.json()
        versions = data.get("versions", [])
        if not versions:
            return prompt_name, "unknown", "unknown"
        v_info = versions[0]
        if prompt_version:
            v_info = next(
                (x for x in versions if x.get("version") == prompt_version),
                versions[0],
            )
        version = v_info.get("version") or "unknown"
        model = v_info.get("model") or "unknown"

        r2 = client.get(
            f"{api_url}/prompts/{prompt_name}/{version}",
            headers=headers or None,
        )
        r2.raise_for_status()
        meta = r2.json().get("metadata", {})
        model = meta.get("model") or model

    return prompt_name, version, model


def _run_single_version(
    items: list[dict],
    api_url: str,
    api_key: str | None,
    prompt_name: str,
    prompt_version: str,
    fuzzy_threshold: float,
) -> tuple[list[dict], dict]:
    """Run evals for one prompt version. Returns (results, aggregates)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        _, resolved_version, model = _fetch_prompt_metadata(
            api_url, api_key, prompt_name, prompt_version
        )
    except Exception:
        resolved_version = prompt_version or "latest"
        model = "unknown"

    results: list[dict] = []

    with httpx.Client(timeout=60.0) as client:
        for i, item in enumerate(items):
            question = item.get("question", "")
            expected = item.get("expected_answer", "")

            payload: dict[str, Any] = {"message": question}
            if prompt_version and prompt_version != "latest":
                payload["prompt_version"] = prompt_version

            start = datetime.now(timezone.utc)
            try:
                r = client.post(
                    f"{api_url}/chat/rag",
                    json=payload,
                    headers=headers,
                )
                r.raise_for_status()
                actual = r.json().get("response", "")
            except Exception as e:
                actual = ""
                print(
                    f"Error on item {i + 1} ({resolved_version}): {e}",
                    file=sys.stderr,
                )

            end = datetime.now(timezone.utc)
            latency_ms = (end - start).total_seconds() * 1000

            eval_result = evaluate_single(
                actual, expected, fuzzy_threshold=fuzzy_threshold
            )

            record = {
                "prompt_name": prompt_name,
                "prompt_version": resolved_version,
                "model": model,
                "question": question,
                "expected": expected,
                "actual": actual,
                "correct": eval_result["correct"],
                "latency_ms": round(latency_ms, 2),
                "similarity_score": eval_result["similarity_score"],
                "normalized_expected": eval_result["normalized_expected"],
                "normalized_actual": eval_result["normalized_actual"],
                "evaluation_method_used": eval_result["evaluation_method_used"],
            }
            results.append(record)

    aggregates = compute_aggregates(results)
    return results, aggregates


def run_evals(
    dataset_path: str,
    api_url: str = DEFAULT_API_URL,
    api_key: str | None = None,
    fuzzy_threshold: float = 0.6,
    prompt_name: str = "rag_chat",
    prompt_versions: list[str] | None = None,
) -> dict:
    """Run evaluations against the real chat pipeline via POST /chat/rag.

    Uses the live API - does NOT call the LLM directly. Requires the API running.

    Args:
        dataset_path: Path to JSON dataset file.
        api_url: Base URL of the API (default: http://localhost:8000).
        api_key: Optional API key for X-API-Key header.
        fuzzy_threshold: Min similarity (0.0-1.0) for fuzzy match. Default 0.6.
        prompt_name: Prompt template name. Default rag_chat.
        prompt_versions: List of versions to compare (e.g. ["v1", "v2"]).
            None = single run with latest version.

    Returns:
        Summary. In single mode: accuracy, avg_latency_ms, avg_similarity_score,
        results. In compare mode: comparison_summary, per_version results.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with path.open(encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError("Dataset must be a JSON array")

    if prompt_versions and len(prompt_versions) > 1:
        return _run_compare_mode(
            items=items,
            api_url=api_url,
            api_key=api_key,
            fuzzy_threshold=fuzzy_threshold,
            prompt_name=prompt_name,
            prompt_versions=prompt_versions,
        )

    # Single-version mode (current behavior)
    version = (prompt_versions[0] if prompt_versions else None) or "latest"
    api_version = None if version == "latest" else version

    results, aggregates = _run_single_version(
        items=items,
        api_url=api_url,
        api_key=api_key,
        prompt_name=prompt_name,
        prompt_version=api_version,
        fuzzy_threshold=fuzzy_threshold,
    )

    summary = {
        "accuracy": aggregates["accuracy"],
        "avg_latency_ms": aggregates["avg_latency_ms"],
        "avg_similarity_score": aggregates["avg_similarity_score"],
        "total": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"results_{timestamp}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {out_path}")
    return summary


def _run_compare_mode(
    items: list[dict],
    api_url: str,
    api_key: str | None,
    fuzzy_threshold: float,
    prompt_name: str,
    prompt_versions: list[str],
) -> dict:
    """Run evals for multiple prompt versions and produce comparison summary."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    version_results: dict[str, list[dict]] = {}
    version_aggregates: dict[str, dict] = {}

    for version in prompt_versions:
        print(f"Running evals for {prompt_name}/{version}...", file=sys.stderr)
        results, aggregates = _run_single_version(
            items=items,
            api_url=api_url,
            api_key=api_key,
            prompt_name=prompt_name,
            prompt_version=version,
            fuzzy_threshold=fuzzy_threshold,
        )
        version_results[version] = results
        version_aggregates[version] = aggregates

        out_path = run_dir / f"{version}_results.json"
        summary = {
            "prompt_name": prompt_name,
            "prompt_version": version,
            "accuracy": aggregates["accuracy"],
            "avg_latency_ms": aggregates["avg_latency_ms"],
            "avg_similarity_score": aggregates["avg_similarity_score"],
            "total": len(results),
            "correct": sum(1 for r in results if r["correct"]),
            "results": results,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  Saved {out_path}", file=sys.stderr)

    comparison = {
        "prompt_name": prompt_name,
        "prompt_versions": prompt_versions,
        "dataset_size": len(items),
        "by_version": {
            v: {
                "accuracy": version_aggregates[v]["accuracy"],
                "avg_latency_ms": version_aggregates[v]["avg_latency_ms"],
                "avg_similarity_score": version_aggregates[v]["avg_similarity_score"],
                "correct": sum(1 for r in version_results[v] if r["correct"]),
                "total": len(version_results[v]),
            }
            for v in prompt_versions
        },
        "timestamp": timestamp,
    }

    comparison_path = run_dir / "comparison_summary.json"
    with comparison_path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"Comparison saved to {comparison_path}")
    return {
        "mode": "compare",
        "run_dir": str(run_dir),
        "comparison_summary": comparison,
        "version_results": version_results,
    }


def main() -> None:
    """CLI: python -m evals.run [dataset] [--api-url URL] [--api-key KEY]
    [--fuzzy-threshold F] [--prompt-name NAME] [--compare-versions v1,v2]."""
    import os

    dataset_path = "evals/datasets/rag_eval_dataset.json"
    api_url = os.environ.get("EVALS_API_URL", DEFAULT_API_URL)
    api_key = os.environ.get("API_KEY") or None
    fuzzy_threshold = 0.6
    prompt_name = "rag_chat"
    prompt_versions: list[str] | None = None

    args = sys.argv[1:]
    if args and not args[0].startswith("--"):
        dataset_path = args[0]
        args = args[1:]

    i = 0
    while i < len(args):
        if args[i] == "--api-url" and i + 1 < len(args):
            api_url = args[i + 1]
            i += 2
        elif args[i] == "--api-key" and i + 1 < len(args):
            api_key = args[i + 1]
            i += 2
        elif args[i] == "--fuzzy-threshold" and i + 1 < len(args):
            try:
                fuzzy_threshold = float(args[i + 1])
            except ValueError:
                fuzzy_threshold = 0.6
            i += 2
        elif args[i] == "--prompt-name" and i + 1 < len(args):
            prompt_name = args[i + 1]
            i += 2
        elif args[i] == "--compare-versions" and i + 1 < len(args):
            prompt_versions = [v.strip() for v in args[i + 1].split(",") if v.strip()]
            i += 2
        else:
            i += 1

    if not Path(dataset_path).is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / dataset_path
        if candidate.exists():
            dataset_path = str(candidate)
        elif Path(dataset_path).exists():
            dataset_path = str(Path(dataset_path).resolve())

    if prompt_versions:
        print(f"Comparing versions {prompt_versions} of {prompt_name} on {dataset_path}...")
    else:
        print(f"Running evals on {dataset_path} against {api_url}...")

    summary = run_evals(
        dataset_path,
        api_url=api_url,
        api_key=api_key,
        fuzzy_threshold=fuzzy_threshold,
        prompt_name=prompt_name,
        prompt_versions=prompt_versions,
    )

    if summary.get("mode") == "compare":
        comp = summary["comparison_summary"]
        print("\n--- Comparison Summary ---")
        for v, m in comp["by_version"].items():
            print(f"  {v}: accuracy={m['accuracy']:.2%}, "
                  f"latency={m['avg_latency_ms']:.0f}ms, "
                  f"similarity={m['avg_similarity_score']:.4f}")
    else:
        print("\n--- Summary ---")
        print(f"Accuracy: {summary['accuracy']:.2%} "
              f"({summary['correct']}/{summary['total']})")
        print(f"Avg latency: {summary['avg_latency_ms']:.2f} ms")
        print(f"Avg similarity score: {summary['avg_similarity_score']:.4f}")


if __name__ == "__main__":
    main()
