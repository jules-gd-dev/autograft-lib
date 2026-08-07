"""Unit tests for Core Entity Resolver orchestrator."""

from unittest.mock import patch

from autograft.core.resolver import resolve_entity
from autograft.models.entities import Entity, ExistingNode, MatchResult


@patch("autograft.core.resolver.find_semantic_match")
def test_resolver_layer_1_hit(mock_semantic) -> None:
    """Test exact match in Layer 1 short-circuits before Layer 2."""
    new_entity = Entity(canonical_name="Apple Inc.", type="Company")
    existing_nodes = [
        ExistingNode(node_id="node_1", canonical_name="Apple Inc.", type="Company")
    ]

    result = resolve_entity(new_entity, existing_nodes)

    assert result.is_match is True
    assert result.matched_node_id == "node_1"
    assert result.layer == "deterministic"
    mock_semantic.assert_not_called()


@patch("autograft.core.resolver.arbitrate_match")
def test_resolver_layer_3_uncertain_hit(mock_arbitrate) -> None:
    """Test uncertain Layer 2 result triggers Layer 3 LLM arbitration."""
    mock_arbitrate.return_value = MatchResult(
        is_match=True,
        matched_node_id="node_uncertain",
        score=1.0,
        layer="llm_arbiter",
    )

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

    result = resolve_entity(new_entity, existing_nodes)

    assert result.is_match is True
    assert result.matched_node_id == "node_uncertain"
    assert result.layer == "llm_arbiter"
    mock_arbitrate.assert_called_once()


def test_resolver_layer_2_strong_hit() -> None:
    """Test strong semantic match in Layer 2 returns semantic result directly."""
    new_entity = Entity(
        canonical_name="Apple Corp",
        type="Company",
        embedding=[1.0, 0.0, 0.0],
    )
    existing_nodes = [
        ExistingNode(
            node_id="node_apple",
            canonical_name="Apple Inc.",
            type="Company",
            embedding=[0.99, 0.0, 0.0],
        )
    ]

    result = resolve_entity(new_entity, existing_nodes)

    assert result.is_match is True
    assert result.matched_node_id == "node_apple"
    assert result.layer == "semantic"


def test_resolver_no_match() -> None:
    """Test when no layer finds a match."""
    new_entity = Entity(canonical_name="Google", type="Company")
    existing_nodes = [
        ExistingNode(node_id="node_1", canonical_name="Apple", type="Company")
    ]

    result = resolve_entity(new_entity, existing_nodes)

    assert result.is_match is False
    assert result.matched_node_id is None
