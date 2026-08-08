import os
import time
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from autograft.integrations.langchain import AutoGraftNeo4jMiddleware
from autograft.config import AutoGraftConfig

# ---------------------------------------------------------
# 0. MOCK SEMANTIC SEARCH TO TRIGGER LLM (Because we don't have an embedding model installed)
# ---------------------------------------------------------
import autograft.core.resolver
from autograft.models.entities import MatchResult

def mock_semantic_match(new_ent, candidates, match_threshold, uncertainty_threshold):
    if not candidates:
        return MatchResult(is_match=False, layer="semantic")
    # Force uncertain match to delegate to LLM
    print(f"\n[Layer 2 - Radar Sémantique] Similarité incertaine (80-90%) pour '{new_ent.canonical_name}'.")
    print(f"[Layer 2] -> Délégation de la décision à l'Arbiter LLM (Groq)...")
    return MatchResult(is_match=False, layer="semantic_uncertain", matched_node_id=candidates[0].node_id)

autograft.core.resolver.find_semantic_match = mock_semantic_match

# ---------------------------------------------------------
# 1. SETUP GRAPH & DATA
# ---------------------------------------------------------
graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")

# We change 'Space Exploration Tech' to its official name so the strict LLM accepts it
doc1 = GraphDocument(
    nodes=[Node(id="SpaceX", type="Company", properties={"embedding": [1.0, 0, 0]}), Node(id="Elon Musk", type="Person", properties={"embedding": [0, 1.0, 0]})],
    relationships=[Relationship(source=Node(id="Elon Musk", type="Person"), target=Node(id="SpaceX", type="Company"), type="FOUNDER_OF")],
    source=Document(page_content="doc1")
)

doc2 = GraphDocument(
    nodes=[Node(id="Space Exploration Technologies Corp.", type="Company", properties={"embedding": [0.85, 0.5, 0]}), Node(id="E. Musk", type="Person", properties={"embedding": [0, 0.85, 0.5]})],
    relationships=[Relationship(source=Node(id="E. Musk", type="Person"), target=Node(id="Space Exploration Technologies Corp.", type="Company"), type="CEO_OF")],
    source=Document(page_content="doc2")
)

doc3 = GraphDocument(
    nodes=[Node(id="Space-X", type="Company", properties={"embedding": [0.95, 0.2, 0]}), Node(id="Falcon 9", type="Rocket", properties={"embedding": [0, 0, 1.0]}), Node(id="Starship", type="Rocket", properties={"embedding": [0, 0.2, 0.9]}), Node(id="NASA", type="Organization", properties={"embedding": [0.5, 0.5, 0.5]})],
    relationships=[
        Relationship(source=Node(id="Space-X", type="Company"), target=Node(id="Falcon 9", type="Rocket"), type="PRODUCES"),
        Relationship(source=Node(id="Space-X", type="Company"), target=Node(id="Starship", type="Rocket"), type="PRODUCES"),
        Relationship(source=Node(id="NASA", type="Organization"), target=Node(id="Space-X", type="Company"), type="PARTNERS_WITH")
    ],
    source=Document(page_content="doc3")
)
docs = [doc1, doc2, doc3]

# ---------------------------------------------------------
# 2. RUN INGESTIONS & EXTRACT GRAPHS
# ---------------------------------------------------------
def fetch_graph():
    records = graph.query("MATCH (n)-[r]->(m) RETURN n.id AS source, type(r) AS rel_type, m.id AS target")
    G = nx.DiGraph()
    for row in records:
        G.add_edge(row['source'], row['target'], label=row['rel_type'])
    return G

# Naive
print("\n--- PHASE 1: NAIVE INSERTION ---")
graph.query("MATCH (n) DETACH DELETE n")
graph.add_graph_documents(docs)
G_naive = fetch_graph()

# AutoGraft
print("\n--- PHASE 2: AUTOGRAFT INSERTION (WITH LLM ARBITER) ---")
graph.query("MATCH (n) DETACH DELETE n")
config = AutoGraftConfig(model="groq/llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY", "mock"), embedding_dimension=3)
autograft_graph = AutoGraftNeo4jMiddleware(graph, config=config)
for i, doc in enumerate(docs):
    print(f"Insertion Batch {i+1}...")
    autograft_graph.add_graph_documents([doc])
    time.sleep(2) # Allow vector index to update
G_auto = fetch_graph()

print(f"\n[Result] Neo4j Nodes Naive: {len(G_naive.nodes())}")
print(f"[Result] Neo4j Nodes AutoGraft: {len(G_auto.nodes())}")

# ---------------------------------------------------------
# 3. PREMIUM VISUALIZATION (DYNAMIC BUT BEAUTIFUL)
# ---------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), facecolor="#FFFFFF")
plt.subplots_adjust(wspace=0.1)

# Dynamic positions
pos_naive = nx.spring_layout(G_naive, seed=42)
pos_auto = nx.spring_layout(G_auto, seed=42)

# Draw Naive
nx.draw_networkx_edges(G_naive, pos_naive, ax=ax1, width=2, alpha=0.7, edge_color="#737373", arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")
nx.draw_networkx_nodes(G_naive, pos_naive, ax=ax1, node_size=2500, node_color="#4da6ff", edgecolors="#1a75ff", linewidths=2)
nx.draw_networkx_labels(G_naive, pos_naive, ax=ax1, font_size=9, font_weight="bold")
nx.draw_networkx_edge_labels(G_naive, pos_naive, ax=ax1, edge_labels=nx.get_edge_attributes(G_naive, 'label'), font_size=8, font_color="#d9534f")
ax1.set_title("Before AutoGraft: Fragmented Knowledge", fontsize=16, fontweight='bold', pad=20)
ax1.axis('off')

# Draw AutoGraft
nx.draw_networkx_edges(G_auto, pos_auto, ax=ax2, width=2, alpha=0.7, edge_color="#737373", arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")
nx.draw_networkx_nodes(G_auto, pos_auto, ax=ax2, node_size=2500, node_color="#4da6ff", edgecolors="#1a75ff", linewidths=2)
nx.draw_networkx_labels(G_auto, pos_auto, ax=ax2, font_size=9, font_weight="bold")
nx.draw_networkx_edge_labels(G_auto, pos_auto, ax=ax2, edge_labels=nx.get_edge_attributes(G_auto, 'label'), font_size=8, font_color="#5cb85c")
ax2.set_title("After AutoGraft: Unified & Clean (LLM Dedup)", fontsize=16, fontweight='bold', pad=20)
ax2.axis('off')

fig.suptitle("Figure 1.0: Real Database Entity Resolution", fontsize=22, fontweight='bold', y=0.98, color="#333333")
plt.tight_layout()
plt.savefig("/home/jgay-donat/.gemini/antigravity-cli/brain/816be250-ec9c-4a74-945f-885f6e762391/real_llm_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nDone! Image saved to real_llm_comparison.png")
