"""Layer 2: Semantic Entity Resolution using vector embeddings."""
from typing import Optional
import numpy as np
from autograft.models.entities import Entity, ExistingNode, MatchResult


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Computes the cosine similarity between two vector embeddings."""
    a = np.array(vec1, dtype=float)
    b = np.array(vec2, dtype=float)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def find_semantic_match(
    new_entity: Entity,
    existing_nodes: list[ExistingNode],
    match_threshold: float = 0.85,
    uncertainty_threshold: float = 0.75,
) -> MatchResult:
    """Finds semantic entity match based on vector embedding similarity."""
    if new_entity.embedding is None:
        return MatchResult(is_match=False)

    best_score: float = -1.0
    best_node_id: Optional[str] = None

    for node in existing_nodes:
        if node.embedding is None:
            continue
        sim = cosine_similarity(new_entity.embedding, node.embedding)
        if sim > best_score:
            best_score = sim
            best_node_id = node.node_id

    if best_node_id is None or best_score < uncertainty_threshold:
        return MatchResult(is_match=False)

    if best_score >= match_threshold:
        return MatchResult(
            is_match=True,
            matched_node_id=best_node_id,
            score=best_score,
            layer="semantic",
        )

    return MatchResult(
        is_match=False,
        matched_node_id=best_node_id,
        score=best_score,
        layer="semantic_uncertain",
    )
