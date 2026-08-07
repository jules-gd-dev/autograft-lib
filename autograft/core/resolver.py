import logging

from autograft.config import AutoGraftConfig
from autograft.db.client import GraphDatabaseClient, ListDatabaseClient
from autograft.layers.deterministic import find_exact_match
from autograft.layers.llm_arbiter import arbitrate_match
from autograft.layers.semantic import find_semantic_match
from autograft.models.entities import Entity, ExistingNode, MatchResult

logger = logging.getLogger("autograft.resolver")


def resolve_entity(
    new_entity: Entity,
    db_client: GraphDatabaseClient | list[ExistingNode],
    config: AutoGraftConfig | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> MatchResult:
    """Orchestrates 3-layer ER pipeline to determine entity match."""
    if isinstance(db_client, list):
        db_client = ListDatabaseClient(db_client)

    cfg = config or AutoGraftConfig()
    if model:
        cfg.model = model
    if api_key:
        cfg.api_key = api_key
    if api_base:
        cfg.api_base = api_base

    # Layer 1: Deterministic
    exact_candidates = db_client.find_exact_candidates(new_entity)
    if not exact_candidates:
        logger.debug(
            f"Declined merge: No existing nodes match type '{new_entity.type}' for '{new_entity.canonical_name}'"
        )
        return MatchResult(is_match=False)

    exact_result = find_exact_match(new_entity, exact_candidates, config=cfg)
    if exact_result.is_match:
        return exact_result

    # Layer 2: Semantic Vector Blocking
    semantic_candidates = db_client.find_semantic_candidates(new_entity, limit=5)
    semantic_result = find_semantic_match(
        new_entity,
        semantic_candidates,
        match_threshold=cfg.match_threshold,
        uncertainty_threshold=cfg.uncertainty_threshold,
    )
    if semantic_result.is_match:
        return semantic_result

    # Layer 3: LLM Arbitration for uncertain matches
    if (
        semantic_result.layer == "semantic_uncertain"
        and semantic_result.matched_node_id is not None
    ):
        matched_node = next(
            (n for n in semantic_candidates if n.node_id == semantic_result.matched_node_id),
            None,
        )
        if matched_node is not None:
            return arbitrate_match(new_entity, matched_node, config=cfg)

    logger.debug(
        f"Declined merge: No candidate found for '{new_entity.canonical_name}'"
    )
    return MatchResult(is_match=False)
