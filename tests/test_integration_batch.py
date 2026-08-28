"""Integration tests for intra-batch dedup in the LangChain/LlamaIndex paths.

The unit suite covers layers in isolation; these tests exercise the full
document path (extraction-shaped input -> middleware -> store) that the
audit found untested: two variants of the same entity inside one batch.
"""

from unittest.mock import MagicMock

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

from autograft.integrations.langchain import AutoGraftNeo4jMiddleware
from autograft.integrations.llamaindex import AutoGraftLlamaIndexMiddleware


def test_langchain_two_variants_same_doc_collapse_onto_one_new_node() -> None:
    """Two unseen variants in one doc must not be written as two nodes."""
    mock_neo4j = MagicMock()
    mock_neo4j.query.return_value = []  # graph knows neither variant

    middleware = AutoGraftNeo4jMiddleware(mock_neo4j)
    node_a = Node(id="Apple", type="Company")
    node_b = Node(id="Apple Inc.", type="Company")
    rel = Relationship(source=node_b, target=node_a, type="SAME_AS")
    doc = GraphDocument(
        nodes=[node_a, node_b],
        relationships=[rel],
        source=Document(page_content="Apple. Apple Inc."),
    )

    middleware.add_graph_documents([doc])

    # Both nodes collapse onto the cluster representative (first variant).
    assert node_a.id == "Apple"
    assert node_b.id == "Apple"
    assert doc.relationships[0].source.id == "Apple"
    assert doc.relationships[0].target.id == "Apple"
    mock_neo4j.add_graph_documents.assert_called_once()


def test_langchain_two_variants_same_doc_merge_onto_existing_graph_node() -> None:
    """If the graph already knows the canonical, every variant maps onto it."""
    mock_neo4j = MagicMock()
    mock_neo4j.query.return_value = [{"id": "Apple Inc.", "aliases": ["Apple"]}]

    middleware = AutoGraftNeo4jMiddleware(mock_neo4j)
    node_a = Node(id="Apple", type="Company")
    node_b = Node(id="Apple Inc.", type="Company")
    doc = GraphDocument(
        nodes=[node_a, node_b],
        relationships=[],
        source=Document(page_content="Apple. Apple Inc."),
    )

    middleware.add_graph_documents([doc])

    assert node_a.id == "Apple Inc."
    assert node_b.id == "Apple Inc."
    # The representative's alias was persisted on the matched graph node.
    persist_calls = [
        c for c in mock_neo4j.query.call_args_list if "SET" in str(c.args[0])
    ]
    assert persist_calls


def test_llamaindex_two_variants_same_upsert_collapse_onto_one_new_node() -> None:
    """Two unseen variants in one LlamaIndex upsert are written as one node."""
    mock_store = MagicMock()
    mock_store.structured_query.return_value = ([], None)

    middleware = AutoGraftLlamaIndexMiddleware(mock_store)

    class MutableMockEntityNode:
        def __init__(self, name: str, label: str) -> None:
            self.name = name
            self.label = label

    node_a = MutableMockEntityNode(name="Apple", label="Company")
    node_b = MutableMockEntityNode(name="Apple Inc.", label="Company")

    middleware.upsert_nodes([node_a, node_b])

    assert node_a.name == "Apple"
    assert node_b.name == "Apple"
    mock_store.upsert_nodes.assert_called_once()
