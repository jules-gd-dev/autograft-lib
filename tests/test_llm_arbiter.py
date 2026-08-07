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
    mock_ask_llm.return_value = ("NON", 15)
    new_entity = Entity(canonical_name="Apple", type="Fruit", aliases=["Crisp Apple"])
    existing_node = ExistingNode(
        node_id="456",
        canonical_name="Apple Inc.",
        type="Company",
        aliases=["Apple Computer"],
    )

    # Act
    result = arbitrate_match(new_entity, existing_node)

    # Assert
    assert result.is_match is False
    assert result.matched_node_id is None
    assert result.layer == "llm_arbiter"
    assert result.tokens_used == 15


@patch("autograft.layers.llm_arbiter._ask_llm")
def test_llm_arbiter_exception_handling(mock_ask_llm) -> None:
    """Test arbitrate_match handles exceptions from _ask_llm gracefully."""
    mock_ask_llm.side_effect = RuntimeError("API down")
    new_entity = Entity(canonical_name="A", type="T")
    existing_node = ExistingNode(node_id="1", canonical_name="B", type="T")

    result = arbitrate_match(new_entity, existing_node)

    assert result.is_match is False
    assert result.score == 0.0
    assert result.layer == "llm_arbiter_error"


from unittest.mock import MagicMock

import pytest

from autograft.layers.llm_arbiter import _ask_llm


@patch("litellm.completion")
def test_ask_llm_success(mock_completion) -> None:
    """Test _ask_llm returns content and token count."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "YES"
    mock_response.usage.total_tokens = 42
    mock_completion.return_value = mock_response

    content, tokens = _ask_llm("test prompt", model="mock-model")
    assert content == "YES"
    assert tokens == 42


@patch("time.sleep")
@patch("litellm.completion")
def test_ask_llm_rate_limit_retry_then_success(mock_completion, mock_sleep) -> None:
    """Test _ask_llm retries on rate limit error and succeeds."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "YES"
    mock_response.usage = None

    class RateLimitError(Exception):
        pass

    mock_completion.side_effect = [RateLimitError("429 RateLimit"), mock_response]

    content, tokens = _ask_llm("test prompt", model="mock-model")
    assert content == "YES"
    assert tokens == 0
    mock_sleep.assert_called_once_with(3)


@patch("time.sleep")
@patch("litellm.completion")
def test_ask_llm_max_retries_exceeded(mock_completion, mock_sleep) -> None:
    """Test _ask_llm raises RuntimeError after 5 failed rate limit retries."""
    mock_completion.side_effect = Exception("429 RateLimit")

    with pytest.raises(RuntimeError, match="failed after 5 retry attempts"):
        _ask_llm("test prompt", model="mock-model")

    assert mock_sleep.call_count == 5


@patch("litellm.completion")
def test_ask_llm_unexpected_exception(mock_completion) -> None:
    """Test _ask_llm re-raises non-rate-limit exceptions immediately."""
    mock_completion.side_effect = ValueError("Invalid argument")

    with pytest.raises(ValueError, match="Invalid argument"):
        _ask_llm("test prompt", model="mock-model")
