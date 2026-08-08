"""Accuracy metrics computed against ground-truth canonical identities."""

from dataclasses import dataclass, field


@dataclass
class MentionResult:
    """Outcome of resolving a single entity mention through the pipeline."""

    name: str
    entity_type: str
    true_canonical: str
    matched_node_id: str | None
    matched_canonical: str | None
    layer: str
    is_match: bool
    tokens: int
    latency_ms: float


@dataclass
class ResolutionReport:
    """Aggregated results of a full benchmark run."""

    mentions: list[MentionResult] = field(default_factory=list)
    node_canon: dict[str, str] = field(default_factory=dict)

    def add(self, m: MentionResult) -> None:
        self.mentions.append(m)


def compute_accuracy(report: ResolutionReport) -> dict[str, float]:
    """Compute merge precision / recall / F1 against ground truth.

    A merge is correct iff the matched node shares the mention's canonical_id.
    """
    true_pos = 0
    false_pos = 0
    # True merges = total mentions minus first occurrence of each (id, type).
    seen: set[str] = set()
    total_true_merges = 0
    for m in report.mentions:
        key = f"{m.true_canonical}|{m.entity_type}"
        if key in seen:
            total_true_merges += 1
        seen.add(key)

        if m.is_match:
            if m.matched_canonical == m.true_canonical:
                true_pos += 1
            else:
                false_pos += 1

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 1.0
    recall = true_pos / total_true_merges if total_true_merges else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "true_merges_possible": float(total_true_merges),
        "correct_merges": float(true_pos),
        "wrong_merges": float(false_pos),
        "declined_merges": float(total_true_merges - true_pos),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


_CATEGORY = {
    ("deterministic", True): "deterministic_match",
    ("semantic", True): "semantic_match",
    ("llm_arbiter", True): "llm_merge",
    ("llm_arbiter", False): "llm_declined",
    ("llm_arbiter_error", False): "llm_error",
}


def layer_breakdown(report: ResolutionReport) -> dict[str, int]:
    """Count mentions per outcome, distinguishing matches from declines.

    Note: a non-match falls through with the default layer='deterministic',
    so we classify by (layer, is_match) to avoid conflating true deterministic
    matches with no-candidate declines.
    """
    counts: dict[str, int] = {}
    for m in report.mentions:
        cat = _CATEGORY.get((m.layer, m.is_match), "no_match_declined")
        counts[cat] = counts.get(cat, 0) + 1
    return counts
