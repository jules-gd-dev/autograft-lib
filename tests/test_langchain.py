"""Tests for AutoGraft LangChain integration."""

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

    assert mock_neo4j.query.call_count == 3

    # Assert nodes were updated in-place (canonicalized)
    assert doc.nodes[0].id == "Apple Inc."  # Deduplicated!
    assert doc.nodes[1].id == "iPhone"  # Untouched (new)

    # Assert relationships were remapped
    assert doc.relationships[0].source.id == "Apple Inc."
    assert doc.relationships[0].target.id == "iPhone"

    # Assert underlying Neo4jGraph was called
    mock_neo4j.add_graph_documents.assert_called_once()


def test_langchain_middleware_relationship_remapping_and_cache() -> None:
    """Test source and target node remapping and cache initialization for new node types."""
    mock_neo4j = MagicMock()

    def mock_query(query: str, **kwargs) -> list[dict]:
        if "Company" in query:
            return [{"id": "Apple Inc.", "aliases": ["Apple"]}]
        if "Product" in query:
            return [{"id": "iPhone 15", "aliases": ["iPhone"]}]
        return []

    mock_neo4j.query.side_effect = mock_query

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
    assert mock_neo4j.query.call_count == 4

    # Now add document with an uncached node type
    node_new = Node(id="UncachedEntity", type="UncachedType")
    doc_uncached = GraphDocument(
        nodes=[node_new], relationships=[], source=Document(page_content="")
    )
    middleware.add_graph_documents([doc_uncached])
    assert mock_neo4j.query.call_count == 5


def test_langchain_middleware_semantic_match() -> None:
    """Test that LangChain middleware uses vector search for semantic candidates and handles errors."""
    mock_neo4j = MagicMock()

    def mock_query(query: str, **kwargs) -> list[dict]:
        if "WHERE n.id = $name" in query:
            return []  # No exact match
        if "db.index.vector.queryNodes" in query:
            return [
                {
                    "id": "Apple Inc.",
                    "aliases": [],
                    "embedding": [1.0, 0.0, 0.0],
                    "score": 0.95,
                }
            ]
        return []

    mock_neo4j.query.side_effect = mock_query

    middleware = AutoGraftNeo4jMiddleware(mock_neo4j)
    node_apple = Node(
        id="Apple", type="Company", properties={"embedding": [1.0, 0.0, 0.0]}
    )
    doc = GraphDocument(
        nodes=[node_apple], relationships=[], source=Document(page_content="")
    )
    middleware.add_graph_documents([doc])

    # Assert nodes were updated in-place (canonicalized via semantic match)
    assert doc.nodes[0].id == "Apple Inc."
    assert mock_neo4j.query.call_count == 4  # exact + vector index + semantic + persist

    # Test Exception Handling in semantic candidates
    def mock_query_err(query: str, **kwargs) -> list[dict]:
        if "db.index.vector.queryNodes" in query:
            raise ValueError("Index missing")
        return []

    mock_neo4j.query.side_effect = mock_query_err

    node_err = Node(
        id="Apple", type="Company", properties={"embedding": [1.0, 0.0, 0.0]}
    )
    doc_err = GraphDocument(
        nodes=[node_err], relationships=[], source=Document(page_content="")
    )
    middleware.add_graph_documents([doc_err])
    # Fails gracefully
    assert doc_err.nodes[0].id == "Apple"
    assert (
        mock_neo4j.query.call_count == 6
    )  # first doc(4) + second doc(2), index cached
