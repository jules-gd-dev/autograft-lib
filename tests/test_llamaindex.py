"""Tests for AutoGraft LlamaIndex integration."""

from unittest.mock import MagicMock

from autograft.integrations.llamaindex import AutoGraftLlamaIndexMiddleware


def test_llamaindex_middleware_deduplication() -> None:
    """Test that AutoGraftLlamaIndexMiddleware deduplicates nodes."""
    mock_store = MagicMock()
    # Mock Neo4j query returning an existing canonical node
    mock_store.structured_query.return_value = (
        [{"id": "Apple Inc.", "aliases": ["Apple"]}],
        None,
    )

    middleware = AutoGraftLlamaIndexMiddleware(mock_store)

    # Create dummy EntityNode
    # We must allow the middleware to modify the name attribute, so we need a mock object that is mutable
    class MutableMockEntityNode:
        def __init__(self, name, label):
            self.name = name
            self.label = label

    mut_node_apple = MutableMockEntityNode(name="Apple", label="Company")
    mut_node_iphone = MutableMockEntityNode(name="iPhone", label="Product")

    middleware.upsert_nodes([mut_node_apple, mut_node_iphone])

    # Assert Neo4j query was called
    assert mock_store.structured_query.call_count == 3

    # Assert nodes were updated in-place (canonicalized)
    assert mut_node_apple.name == "Apple Inc."  # Deduplicated!
    assert mut_node_iphone.name == "iPhone"  # Untouched (new)

    # Assert underlying store was called
    mock_store.upsert_nodes.assert_called_once()


def test_llamaindex_middleware_relations_and_getattr() -> None:
    """Test upsert_relations, uncached label, and store method delegation via __getattr__."""
    mock_store = MagicMock()
    mock_store.custom_store_method.return_value = "delegated_result"
    mock_store.structured_query.return_value = ([], None)

    middleware = AutoGraftLlamaIndexMiddleware(mock_store)

    class MutableMockEntityNode:
        def __init__(self, name, label):
            self.name = name
            self.label = label

    # Upsert node with uncached label
    middleware.upsert_nodes([MutableMockEntityNode(name="BrandNew", label="NewLabel")])
    assert mock_store.structured_query.call_count == 1

    # Test upsert_relations
    relations = [MagicMock()]
    middleware.upsert_relations(relations)
    mock_store.upsert_relations.assert_called_once_with(relations)

    # Test __getattr__ delegation
    result = middleware.custom_store_method("arg1")
    assert result == "delegated_result"
    mock_store.custom_store_method.assert_called_once_with("arg1")


def test_llamaindex_middleware_semantic_match() -> None:
    """Test that LlamaIndex middleware uses vector search for semantic candidates and handles errors."""
    mock_store = MagicMock()

    def mock_structured_query(query: str, **kwargs) -> tuple[list[dict], None]:
        if "WHERE n.id = $name" in query:
            return [], None  # No exact match
        if "db.index.vector.queryNodes" in query:
            return [
                {
                    "id": "Apple Inc.",
                    "aliases": [],
                    "embedding": [1.0, 0.0, 0.0],
                    "score": 0.95,
                }
            ], None
        return [], None

    mock_store.structured_query.side_effect = mock_structured_query

    middleware = AutoGraftLlamaIndexMiddleware(mock_store)

    class MutableMockEntityNode:
        def __init__(self, name, label, properties=None):
            self.name = name
            self.label = label
            self.properties = properties or {}

    mut_node_apple = MutableMockEntityNode(
        name="Apple", label="Company", properties={"embedding": [1.0, 0.0, 0.0]}
    )
    middleware.upsert_nodes([mut_node_apple])

    assert mut_node_apple.name == "Apple Inc."
    assert (
        mock_store.structured_query.call_count == 4
    )  # exact + index + semantic + persist

    # Test Exception Handling
    def mock_structured_query_err(query: str, **kwargs) -> tuple[list[dict], None]:
        if "db.index.vector.queryNodes" in query:
            raise ValueError("Index missing")
        return [], None

    mock_store.structured_query.side_effect = mock_structured_query_err

    mut_node_err = MutableMockEntityNode(
        name="Apple", label="Company", properties={"embedding": [1.0, 0.0, 0.0]}
    )
    middleware.upsert_nodes([mut_node_err])
    assert mut_node_err.name == "Apple"
    assert (
        mock_store.structured_query.call_count == 6
    )  # first doc(4) + second exact(1) + vector error(1)
