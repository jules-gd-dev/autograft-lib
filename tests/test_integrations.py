"""Tests for AutoGraft external integrations."""
from unittest.mock import MagicMock
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

from autograft.integrations.langchain import AutoGraftNeo4jMiddleware
from autograft.models.entities import ExistingNode


from langchain_core.documents import Document

def test_langchain_middleware_deduplication() -> None:
    """Test that AutoGraftNeo4jMiddleware deduplicates nodes and updates relationships."""
    mock_neo4j = MagicMock()
    # Mock Neo4j query returning an existing canonical node
    mock_neo4j.query.return_value = [
        {"id": "Apple Inc.", "aliases": ["Apple"]}
    ]

    middleware = AutoGraftNeo4jMiddleware(mock_neo4j)

    # Create dummy graph document with duplicate node "Apple"
    node_apple = Node(id="Apple", type="Company")
    node_iphone = Node(id="iPhone", type="Product")
    rel = Relationship(source=node_apple, target=node_iphone, type="PRODUCES")
    doc = GraphDocument(
        nodes=[node_apple, node_iphone],
        relationships=[rel],
        source=Document(page_content="Apple produces the iPhone.")
    )

    middleware.add_graph_documents([doc])

    # Assert Neo4j query was called for 'Company' and 'Product'
    assert mock_neo4j.query.call_count == 2
    
    # Assert nodes were updated in-place (canonicalized)
    assert doc.nodes[0].id == "Apple Inc."  # Deduplicated!
    assert doc.nodes[1].id == "iPhone"      # Untouched (new)

    # Assert relationships were remapped
    assert doc.relationships[0].source.id == "Apple Inc."
    assert doc.relationships[0].target.id == "iPhone"

    # Assert underlying Neo4jGraph was called
    mock_neo4j.add_graph_documents.assert_called_once()

from autograft.integrations.llamaindex import AutoGraftLlamaIndexMiddleware
from collections import namedtuple

def test_llamaindex_middleware_deduplication() -> None:
    """Test that AutoGraftLlamaIndexMiddleware deduplicates nodes."""
    mock_store = MagicMock()
    # Mock Neo4j query returning an existing canonical node
    mock_store.structured_query.return_value = (
        [{"id": "Apple Inc.", "aliases": ["Apple"]}],
        None
    )

    middleware = AutoGraftLlamaIndexMiddleware(mock_store)

    # Create dummy EntityNode
    MockEntityNode = namedtuple('MockEntityNode', ['name', 'label'])
    node_apple = MockEntityNode(name="Apple", label="Company")
    node_iphone = MockEntityNode(name="iPhone", label="Product")
    
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
    assert mut_node_iphone.name == "iPhone"     # Untouched (new)

    # Assert underlying store was called
    mock_store.upsert_nodes.assert_called_once()
