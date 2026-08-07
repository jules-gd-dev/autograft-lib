"""Unit tests for AutoGraftConfig and explicit credentials initialization."""

from unittest.mock import MagicMock, patch

from autograft.config import AutoGraftConfig
from autograft.core.resolver import resolve_entity
from autograft.integrations.langchain import AutoGraftNeo4jMiddleware
from autograft.integrations.llamaindex import AutoGraftLlamaIndexMiddleware
from autograft.layers.llm_arbiter import _ask_llm
from autograft.models.entities import Entity, ExistingNode


def test_autograft_config_defaults() -> None:
    """Test AutoGraftConfig default values."""
    cfg = AutoGraftConfig()
    assert cfg.model == "groq/llama-3.3-70b-versatile"
    assert cfg.api_key is None
    assert cfg.api_base is None
    assert cfg.match_threshold == 0.85
    assert cfg.uncertainty_threshold == 0.75


def test_autograft_config_custom_values() -> None:
    """Test AutoGraftConfig explicit custom values."""
    cfg = AutoGraftConfig(
        model="openai/gpt-4o",
        api_key="sk-test-key",
        api_base="https://custom.api.base/v1",
        match_threshold=0.9,
        uncertainty_threshold=0.8,
    )
    assert cfg.model == "openai/gpt-4o"
    assert cfg.api_key == "sk-test-key"
    assert cfg.api_base == "https://custom.api.base/v1"
    assert cfg.match_threshold == 0.9
    assert cfg.uncertainty_threshold == 0.8


@patch("litellm.completion")
def test_ask_llm_with_explicit_config_and_api_key(mock_completion) -> None:
    """Test _ask_llm passes explicit api_key and api_base to litellm."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "YES"
    mock_response.usage.total_tokens = 10
    mock_completion.return_value = mock_response

    cfg = AutoGraftConfig(
        model="anthropic/claude-3-5-sonnet",
        api_key="custom-key-123",
        api_base="https://custom.endpoint/v1",
    )

    content, tokens = _ask_llm("test prompt", config=cfg)
    assert content == "YES"
    assert tokens == 10
    mock_completion.assert_called_once_with(
        model="anthropic/claude-3-5-sonnet",
        messages=[{"role": "user", "content": "test prompt"}],
        api_key="custom-key-123",
        api_base="https://custom.endpoint/v1",
    )


@patch("autograft.core.resolver.arbitrate_match")
def test_resolver_passes_config(mock_arbitrate) -> None:
    """Test resolve_entity passes config and explicit arguments down to arbiter."""
    new_entity = Entity(
        canonical_name="Apple Corp", type="Company", embedding=[1.0, 0.0, 0.0]
    )
    existing_nodes = [
        ExistingNode(
            node_id="n1",
            canonical_name="Pear Inc.",
            type="Company",
            embedding=[0.8, 0.6, 0.0],
        )
    ]

    cfg = AutoGraftConfig(model="openai/gpt-4o", api_key="sk-explicit")
    resolve_entity(new_entity, existing_nodes, config=cfg)
    mock_arbitrate.assert_called_once_with(new_entity, existing_nodes[0], config=cfg)

    mock_arbitrate.reset_mock()
    resolve_entity(
        new_entity,
        existing_nodes,
        model="groq/llama-3.3-70b-versatile",
        api_key="sk-kwarg-key",
        api_base="https://kwarg.endpoint/v1",
    )
    assert mock_arbitrate.call_count == 1
    call_cfg = mock_arbitrate.call_args.kwargs["config"]
    assert call_cfg.model == "groq/llama-3.3-70b-versatile"
    assert call_cfg.api_key == "sk-kwarg-key"
    assert call_cfg.api_base == "https://kwarg.endpoint/v1"


def test_middlewares_support_explicit_credentials() -> None:
    """Test LangChain and LlamaIndex middlewares accept explicit config and credentials."""
    mock_neo4j = MagicMock()
    middleware_lc = AutoGraftNeo4jMiddleware(
        mock_neo4j,
        model="openai/gpt-4o",
        api_key="sk-langchain-key",
        api_base="https://lc.endpoint/v1",
    )
    assert middleware_lc.config.model == "openai/gpt-4o"
    assert middleware_lc.config.api_key == "sk-langchain-key"
    assert middleware_lc.config.api_base == "https://lc.endpoint/v1"

    mock_store = MagicMock()
    middleware_li = AutoGraftLlamaIndexMiddleware(
        mock_store,
        model="anthropic/claude-3-5-sonnet",
        api_key="sk-llamaindex-key",
        api_base="https://li.endpoint/v1",
    )
    assert middleware_li.config.model == "anthropic/claude-3-5-sonnet"
    assert middleware_li.config.api_key == "sk-llamaindex-key"
    assert middleware_li.config.api_base == "https://li.endpoint/v1"
