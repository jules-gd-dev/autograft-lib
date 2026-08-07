"""Layer 1: Deterministic Entity Resolution using string matching."""

from rapidfuzz import fuzz

from autograft.config import AutoGraftConfig
from autograft.models.entities import Entity, ExistingNode, MatchResult

MATCH_THRESHOLD = 95.0


def find_exact_match(
    new_entity: Entity,
    existing_nodes: list[ExistingNode],
    config: AutoGraftConfig | None = None,
) -> MatchResult:
    """Checks if new_entity matches any existing node based on exact name or alias."""
    cfg = config or AutoGraftConfig()
    matching_func = getattr(fuzz, cfg.matching_algorithm, fuzz.ratio)

    best_score = 0.0
    best_node_id = None

    for existing in existing_nodes:
        candidate_names = [existing.canonical_name] + existing.aliases
        for name in candidate_names:
            score = float(matching_func(new_entity.canonical_name, name))
            if score > best_score:
                best_score = score
                best_node_id = existing.node_id

    if best_score >= MATCH_THRESHOLD and best_node_id is not None:
        return MatchResult(
            is_match=True,
            matched_node_id=best_node_id,
            score=best_score,
            layer="deterministic",
        )

    return MatchResult(is_match=False)
