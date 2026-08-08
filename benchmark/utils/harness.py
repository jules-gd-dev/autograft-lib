"""Benchmark harness: litellm token/cost tally, embeddings, and result assembly."""

from benchmark.data.catalog import BenchmarkDoc
from benchmark.embed import embed_batch
from benchmark.utils.cost import cost_from_tokens
from benchmark.utils.metrics import (
    ResolutionReport,
    compute_accuracy,
    layer_breakdown,
)

ARBITER_MODEL = "groq/llama-3.1-8b-instant"

# Real per-call token accounting, populated by the litellm success callback.
TALLY: dict[str, int | float] = {"prompt": 0, "completion": 0, "calls": 0, "cost": 0.0}


def litellm_callback(kwargs, completion_response, start_time, end_time) -> None:
    """Accumulate real token counts and cost from each API response."""
    u = getattr(completion_response, "usage", None)
    pt = getattr(u, "prompt_tokens", 0) or 0 if u else 0
    ct = getattr(u, "completion_tokens", 0) or 0 if u else 0
    TALLY["prompt"] += pt
    TALLY["completion"] += ct
    TALLY["calls"] += 1
    TALLY["cost"] += cost_from_tokens(pt, ct, ARBITER_MODEL)


def build_embeddings(docs: list[BenchmarkDoc]) -> dict[str, list[float]]:
    """Pre-compute real MiniLM embeddings for every unique entity name."""
    names = sorted({m.name for d in docs for m in d.entities})
    print(f"Embedding {len(names)} unique entity names with all-MiniLM-L6-v2 ...")
    vecs = embed_batch(names)
    return dict(zip(names, vecs))


def assemble_results(report: ResolutionReport) -> dict:
    """Build the full results dict incl. AutoGraft / Naive / Full-LLM baselines."""
    total = len(report.mentions)
    uniq = len({f"{m.true_canonical}|{m.entity_type}" for m in report.mentions})
    acc = compute_accuracy(report)
    per_call = TALLY["calls"] or 1
    avg_prompt = TALLY["prompt"] / per_call
    avg_comp = TALLY["completion"] / per_call
    full_tokens = int((avg_prompt + avg_comp) * total)
    full_cost = cost_from_tokens(int(avg_prompt * total), int(avg_comp * total), ARBITER_MODEL)
    latency = [m.latency_ms for m in report.mentions]
    return {
        "model": ARBITER_MODEL,
        "total_mentions": total,
        "unique_identities": uniq,
        "autograft": {
            "llm_calls": TALLY["calls"],
            "prompt_tokens": TALLY["prompt"],
            "completion_tokens": TALLY["completion"],
            "total_tokens": TALLY["prompt"] + TALLY["completion"],
            "cost_usd": round(TALLY["cost"], 8),
            "final_nodes": len(report.node_canon),
            "layer_breakdown": layer_breakdown(report),
            "latency_mean_ms": round(sum(latency) / len(latency), 3) if latency else 0,
            "latency_max_ms": round(max(latency), 3) if latency else 0,
        },
        "naive": {"final_nodes": total, "duplicates": total - uniq, "cost_usd": 0.0},
        "full_llm_er": {
            "llm_calls": total,
            "total_tokens": full_tokens,
            "cost_usd": round(full_cost, 8),
            "per_call_prompt": round(avg_prompt, 1),
            "per_call_completion": round(avg_comp, 1),
        },
        "accuracy": acc,
        "mentions": [_dump(m) for m in report.mentions],
    }


def _dump(m) -> dict:
    return {
        "name": m.name,
        "type": m.entity_type,
        "true": m.true_canonical,
        "matched": m.matched_canonical,
        "correct": (m.matched_canonical == m.true_canonical) if m.is_match else None,
        "layer": m.layer,
        "is_match": m.is_match,
        "tokens": m.tokens,
        "ms": round(m.latency_ms, 2),
    }
