"""Unit tests for Layer 3 LLM Arbiter."""

from unittest.mock import patch

from autograft.layers.llm_arbiter import arbitrate_match
from autograft.models.entities import Entity, ExistingNode


@patch("autograft.layers.llm_arbiter._ask_llm")
def test_llm_confirms_match(mock_ask_llm) -> None:
    """Test LLM arbitration confirming match when LLM responds with OUI."""
    # Arrange
    mock_ask_llm.return_value = "OUI"
    new_entity = Entity(canonical_name="J. Dupont", type="Person")
    existing_node = ExistingNode(
        node_id="123", canonical_name="Jean Dupont", type="Person"
    )

    # Act
    result = arbitrate_match(new_entity, existing_node)

    # Assert
    assert result.is_match is True
    assert result.matched_node_id == "123"
    assert result.score == 1.0
    assert result.layer == "llm_arbiter"


@patch("autograft.layers.llm_arbiter._ask_llm")
def test_llm_rejects_match(mock_ask_llm) -> None:
    """Test LLM arbitration rejecting match when LLM responds with NON."""
    # Arrange
    mock_ask_llm.return_value = "NON"
    new_entity = Entity(canonical_name="Apple", type="Fruit")
    existing_node = ExistingNode(
        node_id="456", canonical_name="Apple Inc.", type="Company"
    )

    # Act
    result = arbitrate_match(new_entity, existing_node)

    # Assert
    assert result.is_match is False
    assert result.matched_node_id is None
    assert result.layer == "llm_arbiter"
