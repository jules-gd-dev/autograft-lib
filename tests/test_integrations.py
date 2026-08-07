"""Tests for AutoGraft external integrations."""

from unittest.mock import MagicMock

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

from autograft.integrations.langchain import AutoGraftNeo4jMiddleware


def test_langchain_middleware_deduplication() -> None:
    """Test that AutoGraftNeo4jMiddleware deduplicates nodes and updates relationships."""
    mock_neo4j = MagicMock()
    # Mock Neo4j query returning an existing canonical node
    mock_neo4j.query.return_value = [{"id": "Apple Inc.", "aliases": ["Apple"]}]

    middleware = AutoGraftNeo4jMiddleware(mock_neo4j)

    # Create dummy graph document with duplicate node "Apple"
    node_apple = Node(id="Apple", type="Company")
    node_iphone = Node(id="iPhone", type="Product")
    rel = Relationship(source=node_apple, target=node_iphone, type="PRODUCES")
    doc = GraphDocument(
        nodes=[node_apple, node_iphone],
        relationships=[rel],
        source=Document(page_content="Apple produces the iPhone."),
    )

    middleware.add_graph_documents([doc])

    # Assert Neo4j query was called for 'Company' and 'Product'
    assert mock_neo4j.query.call_count == 2

    # Assert nodes were updated in-place (canonicalized)
    assert doc.nodes[0].id == "Apple Inc."  # Deduplicated!
    assert doc.nodes[1].id == "iPhone"  # Untouched (new)

    # Assert relationships were remapped
    assert doc.relationships[0].source.id == "Apple Inc."
    assert doc.relationships[0].target.id == "iPhone"

    # Assert underlying Neo4jGraph was called
    mock_neo4j.add_graph_documents.assert_called_once()


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
    assert mock_store.structured_query.call_count == 2

    # Assert nodes were updated in-place (canonicalized)
    assert mut_node_apple.name == "Apple Inc."  # Deduplicated!
    assert mut_node_iphone.name == "iPhone"  # Untouched (new)

    # Assert underlying store was called
    mock_store.upsert_nodes.assert_called_once()


def test_langchain_middleware_relationship_remapping_and_cache() -> None:
    """Test source and target node remapping and cache initialization for new node types."""
    mock_neo4j = MagicMock()
    # Mock Neo4j query returning existing canonical nodes for both types
    mock_neo4j.query.side_effect = [
        [{"id": "Apple Inc.", "aliases": ["Apple"]}],
        [{"id": "iPhone 15", "aliases": ["iPhone"]}],
    ]

    middleware = AutoGraftNeo4jMiddleware(mock_neo4j)

    node_apple = Node(id="Apple", type="Company")
    node_iphone = Node(id="iPhone", type="Product")
    rel = Relationship(source=node_apple, target=node_iphone, type="PRODUCES")
    doc = GraphDocument(
        nodes=[node_apple, node_iphone],
        relationships=[rel],
        source=Document(page_content="Apple produces iPhone."),
    )

    middleware.add_graph_documents([doc])

    # Assert both source and target relationships remapped
    assert doc.relationships[0].source.id == "Apple Inc."
    assert doc.relationships[0].target.id == "iPhone 15"

    # Now add document with an uncached node type to hit self._node_cache[node.type] = []
    node_new = Node(id="UncachedEntity", type="UncachedType")
    doc_uncached = GraphDocument(nodes=[node_new], relationships=[], source=Document(page_content=""))
    middleware.add_graph_documents([doc_uncached])
    assert "UncachedType" in middleware._node_cache


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
    assert "NewLabel" in middleware._node_cache

    # Test upsert_relations
    relations = [MagicMock()]
    middleware.upsert_relations(relations)
    mock_store.upsert_relations.assert_called_once_with(relations)

    # Test __getattr__ delegation
    result = middleware.custom_store_method("arg1")
    assert result == "delegated_result"
    mock_store.custom_store_method.assert_called_once_with("arg1")


