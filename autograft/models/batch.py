"""Pydantic models for batch entity resolution results."""

from pydantic import BaseModel

from autograft.models.entities import MatchResult


class ResolutionReport(BaseModel):
    """Aggregated stats for a batch resolution run."""

    input_count: int = 0
    cluster_count: int = 0
    intra_merges: int = 0
    intra_merges_by_layer: dict[str, int] = {}
    graph_merges_by_layer: dict[str, int] = {}
    new_nodes: int = 0
    total_tokens: int = 0


class BatchResult(BaseModel):
    """Result of resolving a batch: per-input fates plus aggregate report."""

    results: list[MatchResult] = []
    report: ResolutionReport = ResolutionReport()
    # Cluster representative id per input (aligned with `results`): same value
    # means the inputs were intra-batch duplicates of one another.
    rep_node_ids: list[str] = []
