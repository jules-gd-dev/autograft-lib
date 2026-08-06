"""Core Entity Resolver orchestrating Layer 1, Layer 2, and Layer 3 matching."""
from autograft.layers.deterministic import find_exact_match
from autograft.layers.llm_arbiter import arbitrate_match
from autograft.layers.semantic import find_semantic_match
from autograft.models.entities import Entity, ExistingNode, MatchResult


def resolve_entity(
    new_entity: Entity, existing_nodes: list[ExistingNode]
) -> MatchResult:
    """Orchestrates 3-layer ER pipeline to determine entity match."""
    # Layer 1: Deterministic
    exact_result = find_exact_match(new_entity, existing_nodes)
    if exact_result.is_match:
        return exact_result

    # Layer 2: Semantic Vector Blocking
    semantic_result = find_semantic_match(new_entity, existing_nodes)
    if semantic_result.is_match:
        return semantic_result

    # Layer 3: LLM Arbitration for uncertain matches
    if (
        semantic_result.layer == "semantic_uncertain"
        and semantic_result.matched_node_id is not None
    ):
        matched_node = next(
            (
                n
                for n in existing_nodes
                if n.node_id == semantic_result.matched_node_id
            ),
            None,
        )
        if matched_node is not None:
            return arbitrate_match(new_entity, matched_node)

    return MatchResult(is_match=False)
