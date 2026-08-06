"""Unit tests for Layer 2 Semantic ER."""

from autograft.layers.semantic import cosine_similarity, find_semantic_match
from autograft.models.entities import Entity, ExistingNode


def test_cosine_similarity() -> None:
    """Test helper cosine_similarity function."""
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_strong_semantic_match() -> None:
    """Test semantic match when similarity is > 0.85."""
    new_entity = Entity(
        canonical_name="Apple",
        type="Company",
        embedding=[1.0, 0.0, 0.0],
    )
    existing_nodes = [
        ExistingNode(
            node_id="node_1",
            canonical_name="Apple Inc.",
            type="Company",
            embedding=[1.0, 0.0, 0.0],
        )
    ]

    result = find_semantic_match(new_entity, existing_nodes)
    assert result.is_match is True
    assert result.matched_node_id == "node_1"
    assert result.score == 1.0
    assert result.layer == "semantic"


def test_uncertain_semantic_match() -> None:
    """Test uncertain match when similarity is between 0.75 and 0.85."""
    new_entity = Entity(
        canonical_name="Apple Corp",
        type="Company",
        embedding=[1.0, 0.0, 0.0],
    )
    existing_nodes = [
        ExistingNode(
            node_id="node_uncertain",
            canonical_name="Pear Inc.",
            type="Company",
            embedding=[0.8, 0.6, 0.0],
        )
    ]

    result = find_semantic_match(new_entity, existing_nodes)
    assert result.is_match is False
    assert result.matched_node_id == "node_uncertain"
    assert 0.75 <= result.score < 0.85
    assert result.layer == "semantic_uncertain"


def test_no_match_low_similarity() -> None:
    """Test low similarity (< 0.75) returns no match and None node_id."""
    new_entity = Entity(
        canonical_name="Apple",
        type="Company",
        embedding=[1.0, 0.0, 0.0],
    )
    existing_nodes = [
        ExistingNode(
            node_id="node_diff",
            canonical_name="Banana",
            type="Fruit",
            embedding=[0.0, 1.0, 0.0],
        )
    ]

    result = find_semantic_match(new_entity, existing_nodes)
    assert result.is_match is False
    assert result.matched_node_id is None


def test_missing_embedding() -> None:
    """Test when new_entity.embedding is None."""
    new_entity = Entity(
        canonical_name="Apple",
        type="Company",
        embedding=None,
    )
    existing_nodes = [
        ExistingNode(
            node_id="node_1",
            canonical_name="Apple Inc.",
            type="Company",
            embedding=[1.0, 0.0, 0.0],
        )
    ]

    result = find_semantic_match(new_entity, existing_nodes)
    assert result.is_match is False
    assert result.matched_node_id is None
