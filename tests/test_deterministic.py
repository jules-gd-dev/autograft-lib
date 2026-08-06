"""Unit tests for Layer 1 Deterministic ER."""

from autograft.layers.deterministic import find_exact_match
from autograft.models.entities import Entity, ExistingNode


def test_exact_name_match() -> None:
    """Test exact match on canonical_name."""
    new_entity = Entity(canonical_name="Apple Inc.", type="Company")
    existing_nodes = [
        ExistingNode(
            node_id="node_1",
            canonical_name="Apple Inc.",
            type="Company",
            aliases=["Apple"],
        ),
        ExistingNode(
            node_id="node_2",
            canonical_name="Microsoft Corp.",
            type="Company",
        ),
    ]

    result = find_exact_match(new_entity, existing_nodes)
    assert result.is_match is True
    assert result.matched_node_id == "node_1"
    assert result.score == 100.0
    assert result.layer == "deterministic"


def test_alias_match() -> None:
    """Test match when new_entity name matches an alias in ExistingNode."""
    new_entity = Entity(canonical_name="AAPL", type="Company")
    existing_nodes = [
        ExistingNode(
            node_id="node_1",
            canonical_name="Apple Inc.",
            type="Company",
            aliases=["AAPL", "Apple"],
        )
    ]

    result = find_exact_match(new_entity, existing_nodes)
    assert result.is_match is True
    assert result.matched_node_id == "node_1"
    assert result.score == 100.0


def test_no_match() -> None:
    """Test completely different names resulting in no match."""
    new_entity = Entity(canonical_name="Google", type="Company")
    existing_nodes = [
        ExistingNode(
            node_id="node_1",
            canonical_name="Apple Inc.",
            type="Company",
            aliases=["AAPL"],
        )
    ]

    result = find_exact_match(new_entity, existing_nodes)
    assert result.is_match is False
    assert result.matched_node_id is None


def test_near_match_below_threshold() -> None:
    """Test similar names where similarity score is below the 95.0 threshold."""
    new_entity = Entity(canonical_name="John Doe", type="Person")
    existing_nodes = [
        ExistingNode(
            node_id="node_1",
            canonical_name="Jon Doe",
            type="Person",
        )
    ]

    result = find_exact_match(new_entity, existing_nodes)
    assert result.is_match is False
    assert result.matched_node_id is None
