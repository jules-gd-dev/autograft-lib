import time

from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node

from autograft.integrations import AutoGraftNeo4jMiddleware

# Wait for Neo4j to be ready
time.sleep(10)

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
autograft_graph = AutoGraftNeo4jMiddleware(graph)

# Create 2 documents with duplicates
doc1 = GraphDocument(
    nodes=[Node(id="Apple Inc.", type="Company")],
    relationships=[],
    source=None
)
doc2 = GraphDocument(
    nodes=[Node(id="Apple", type="Company")],
    relationships=[],
    source=None
)

autograft_graph.add_graph_documents([doc1, doc2])

# Query neo4j to verify
res = graph.query("MATCH (n:Company) RETURN n.id AS id")
print("Neo4j Results:")
for r in res:
    print(r)
assert len(res) == 1, "Duplicate was not resolved!"
assert res[0]["id"] == "Apple Inc.", "ID was not canonicalized"
print("SUCCESS!")
