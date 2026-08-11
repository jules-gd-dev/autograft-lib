import logging

from autograft.config import AutoGraftConfig, alias_key
from autograft.db.client import GraphDatabaseClient, ListDatabaseClient
from autograft.layers.deterministic import find_exact_match
from autograft.layers.lexical import find_lexical_match
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
    enable_arbiter: bool = True,
) -> MatchResult:
    """Orchestrates the multi-layer ER pipeline to determine an entity match.

    `enable_arbiter=False` short-circuits Layer 3 (LLM) for 0-token resolution;
    ambiguous semantic results then surface as `semantic_uncertain` no-matches.
    """
    if isinstance(db_client, list):
        db_client = ListDatabaseClient(db_client)

    cfg = config or AutoGraftConfig()
    if model:
        cfg.model = model
    if api_key:
        cfg.api_key = api_key
    if api_base:
        cfg.api_base = api_base

    # V3: alias_map normalization before L1/lexical matching. The incoming name
    # is always recorded as new_alias (V2); matching uses the mapped canonical.
    incoming_name = new_entity.canonical_name
    mapped = cfg.alias_map.get(alias_key(incoming_name))
    match_entity = (
        new_entity.model_copy(update={"canonical_name": mapped})
        if mapped
        else new_entity
    )

    # Layer 1: Deterministic
    exact_candidates = db_client.find_exact_candidates(match_entity)
    exact_result = find_exact_match(match_entity, exact_candidates, config=cfg)
    if exact_result.is_match:
        exact_result.new_alias = incoming_name
        return exact_result

    # Layer 1.5: Lexical (suffix-strip + acronym), 0 token
    lexical_result = find_lexical_match(match_entity, exact_candidates, config=cfg)
    if lexical_result.is_match:
        lexical_result.new_alias = incoming_name
        return lexical_result

    # Layer 2: Semantic Vector Blocking
    semantic_candidates = db_client.find_semantic_candidates(match_entity, limit=5)
    semantic_result = find_semantic_match(
        match_entity,
        semantic_candidates,
        match_threshold=cfg.match_threshold,
        uncertainty_threshold=cfg.uncertainty_threshold,
    )
    if semantic_result.is_match:
        semantic_result.new_alias = incoming_name
        return semantic_result

    # Layer 3: LLM Arbitration for uncertain matches
    if (
        semantic_result.layer == "semantic_uncertain"
        and semantic_result.matched_node_id is not None
    ):
        if not enable_arbiter:
            return semantic_result
        matched_node = next(
            (
                n
                for n in semantic_candidates
                if n.node_id == semantic_result.matched_node_id
            ),
            None,
        )
        if matched_node is not None:
            llm_result = arbitrate_match(match_entity, matched_node, config=cfg)
            if llm_result.is_match:
                llm_result.new_alias = incoming_name
            return llm_result

    logger.debug(f"Declined merge: No candidate found for '{incoming_name}'")
    return MatchResult(is_match=False)
