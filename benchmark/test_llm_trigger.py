import os
import time
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from autograft.integrations.langchain import AutoGraftNeo4jMiddleware
from autograft.config import AutoGraftConfig

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")

# We design the embeddings so their cosine similarity is EXACTLY 0.85
# SpaceX = [1.0, 0.0, 0.0]
# Space Exploration Tech = [0.85, 0.52678, 0.0]  # cos(theta) = 0.85, sin(theta) = 0.52678
emb_spacex_1 = [1.0, 0.0, 0.0]
emb_spacex_2 = [0.85, 0.52678, 0.0]
emb_musk = [0.0, 1.0, 0.0]

doc1 = GraphDocument(
    nodes=[Node(id="SpaceX", type="Company", properties={"embedding": emb_spacex_1}), Node(id="Elon Musk", type="Person", properties={"embedding": emb_musk})],
    relationships=[Relationship(source=Node(id="Elon Musk", type="Person"), target=Node(id="SpaceX", type="Company"), type="FOUNDER_OF")],
    source=Document(page_content="Elon Musk is the founder of SpaceX, the leading aerospace manufacturer.")
)

doc2 = GraphDocument(
    nodes=[Node(id="Space Exploration Tech", type="Company", properties={"embedding": emb_spacex_2}), Node(id="E. Musk", type="Person", properties={"embedding": emb_musk})],
    relationships=[Relationship(source=Node(id="E. Musk", type="Person"), target=Node(id="Space Exploration Tech", type="Company"), type="CEO_OF")],
    source=Document(page_content="E. Musk serves as the CEO of Space Exploration Tech, known for building Starship.")
)

docs = [doc1, doc2]

graph.query("MATCH (n) DETACH DELETE n")

# Configure AutoGraft to trigger Groq LLM Arbiter!
config = AutoGraftConfig(
    model=os.environ.get("AUTOGRRAFT_LLM_MODEL", "groq/llama-3.1-8b-instant"),
    api_key=os.environ.get("GROQ_API_KEY", ""),
    embedding_dimension=3,
    match_threshold=0.90,
    uncertainty_threshold=0.80
)

# Enable debug logging to see the LLM call
import logging
logging.basicConfig(level=logging.DEBUG)

print("Starting AutoGraft ingestion...")
autograft_graph = AutoGraftNeo4jMiddleware(graph, config=config)
for i, doc in enumerate(docs):
    print(f"Processing doc {i+1}...")
    autograft_graph.add_graph_documents([doc])
    time.sleep(2) # Allow Neo4j vector index to sync

records = graph.query("MATCH (n:Company) RETURN n.id AS id")
print("Nodes in DB:", [r['id'] for r in records])
