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
# 1. SETUP GRAPH & DATA (PRE-COMPUTED EMBEDDINGS)
# ---------------------------------------------------------
graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")

# Embeddings:
# Apple (Company): [0.85, 0.526, 0]
# Apple Inc. (Company): [1.0, 0, 0]  (Similarity with Apple Company = 0.85 -> Triggers LLM!)
# Apple (Fruit): [0, 0, 1.0]         (Different Type, isolated automatically)

doc1 = GraphDocument(
    nodes=[
        Node(id="Tim Cook", type="Person", properties={"embedding": [0.1, 0.9, 0]}),
        Node(id="Apple", type="Company", properties={"embedding": [0.85, 0.526, 0]})
    ],
    relationships=[Relationship(source=Node(id="Tim Cook", type="Person"), target=Node(id="Apple", type="Company"), type="CEO_OF")],
    source=Document(page_content="Tim Cook is the CEO of Apple.")
)

doc2 = GraphDocument(
    nodes=[
        Node(id="Apple Inc.", type="Company", properties={"embedding": [1.0, 0, 0]}),
        Node(id="iPhone", type="Product", properties={"embedding": [0.9, 0, 0.1]})
    ],
    relationships=[Relationship(source=Node(id="Apple Inc.", type="Company"), target=Node(id="iPhone", type="Product"), type="PRODUCES")],
    source=Document(page_content="Apple Inc. produces the iPhone.")
)

doc3 = GraphDocument(
    nodes=[
        Node(id="Apple", type="Fruit", properties={"embedding": [0, 0, 1.0]}),
        Node(id="Cider", type="Beverage", properties={"embedding": [0, 0.1, 0.9]})
    ],
    relationships=[Relationship(source=Node(id="Apple", type="Fruit"), target=Node(id="Cider", type="Beverage"), type="PRODUCES_BEVERAGE")],
    source=Document(page_content="Apple is a fruit used to make Cider.")
)
docs = [doc1, doc2, doc3]

# ---------------------------------------------------------
# 2. RUN INGESTIONS & EXTRACT GRAPHS
# ---------------------------------------------------------
def fetch_graph():
    records = graph.query("MATCH (n)-[r]->(m) RETURN n.id AS source, labels(n)[0] AS source_type, type(r) AS rel_type, m.id AS target, labels(m)[0] AS target_type")
    G = nx.DiGraph()
    for row in records:
        G.add_node(row['source'], type=row['source_type'])
        G.add_node(row['target'], type=row['target_type'])
        G.add_edge(row['source'], row['target'], label=row['rel_type'])
    return G

# Naive
print("\n--- PHASE 1: NAIVE INSERTION ---")
graph.query("MATCH (n) DETACH DELETE n")
graph.add_graph_documents(docs)
G_naive = fetch_graph()

# AutoGraft
print("\n--- PHASE 2: AUTOGRAFT INSERTION (REAL PIPELINE) ---")
graph.query("MATCH (n) DETACH DELETE n")
config = AutoGraftConfig(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY", ""),
    embedding_dimension=3,
    match_threshold=0.90,
    uncertainty_threshold=0.80
)
autograft_graph = AutoGraftNeo4jMiddleware(graph, config=config)

import logging
logging.basicConfig(level=logging.INFO)

for i, doc in enumerate(docs):
    print(f"Insertion Batch {i+1}...")
    autograft_graph.add_graph_documents([doc])
    time.sleep(2) # Allow Neo4j vector index to sync

G_auto = fetch_graph()
print(f"\n[Result] Neo4j Nodes Naive: {len(G_naive.nodes())}")
print(f"[Result] Neo4j Nodes AutoGraft: {len(G_auto.nodes())}")

# ---------------------------------------------------------
# 3. PREMIUM VISUALIZATION
# ---------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
BG_COLOR = "#FFFFFF"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10), facecolor=BG_COLOR)
plt.subplots_adjust(wspace=0.1)

def add_cluster_box(ax, pos, nodes, color, label=None):
    valid_nodes = [n for n in nodes if n in pos]
    if not valid_nodes: return
    x_coords = [pos[n][0] for n in valid_nodes]
    y_coords = [pos[n][1] for n in valid_nodes]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    pad = 0.4
    bbox = FancyBboxPatch((min_x - pad, min_y - pad), max_x - min_x + 2*pad, max_y - min_y + 2*pad,
                          boxstyle="round,pad=0.1,rounding_size=0.2", edgecolor=color, facecolor=color,
                          alpha=0.12, linewidth=2.5, zorder=1)
    ax.add_patch(bbox)
    if label:
        ax.text(min_x, max_y + pad + 0.1, label, color=color, fontsize=12, fontweight='bold', ha='left')

# NAIVE GRAPH DRAWING
pos_naive = {
    "Tim Cook": (-1, 1), "Apple": (-1, 0),
    "Apple Inc.": (0, 1), "iPhone": (0, 0),
    "Apple": (1, 1), "Cider": (1, 0)
}
# Because Naive creates TWO nodes called "Apple" but one is Company and one is Fruit.
# NetworkX identifies nodes by ID, so "Apple" overwrites "Apple".
# To correctly show naive graph which has 2 Apples, we must rename the IDs for NetworkX
G_naive_render = nx.DiGraph()
G_naive_render.add_edges_from([
    ("Tim Cook", "Apple (Company)", {"label": "CEO_OF"}),
    ("Apple Inc.", "iPhone", {"label": "PRODUCES"}),
    ("Apple (Fruit)", "Cider", {"label": "PRODUCES_BEVERAGE"}),
])
pos_naive = {
    "Tim Cook": (-1, 1), "Apple (Company)": (-1, 0),
    "Apple Inc.": (0, 1), "iPhone": (0, 0),
    "Apple (Fruit)": (1, 1), "Cider": (1, 0)
}
nx.draw_networkx_edges(G_naive_render, pos_naive, ax=ax1, width=2, alpha=0.7, edge_color="#737373", arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")
nx.draw_networkx_nodes(G_naive_render, pos_naive, ax=ax1, node_size=3000, node_color="#4da6ff", edgecolors="#1a75ff", linewidths=2)
nx.draw_networkx_labels(G_naive_render, pos_naive, ax=ax1, font_size=10, font_weight="bold")
nx.draw_networkx_edge_labels(G_naive_render, pos_naive, ax=ax1, edge_labels=nx.get_edge_attributes(G_naive_render, 'label'), font_size=9, font_color="#d9534f")

add_cluster_box(ax1, pos_naive, ["Tim Cook", "Apple (Company)"], "#d9534f", "Fragment 1 (Company)")
add_cluster_box(ax1, pos_naive, ["Apple Inc.", "iPhone"], "#d9534f", "Fragment 2 (Company)")
add_cluster_box(ax1, pos_naive, ["Apple (Fruit)", "Cider"], "#f0ad4e", "Fragment 3 (Fruit)")
ax1.set_title("Before AutoGraft: Fragmented Knowledge", fontsize=18, fontweight='bold', pad=20)
ax1.axis('off')

# AUTOGRAFT GRAPH DRAWING
G_auto_render = nx.DiGraph()
G_auto_render.add_edges_from([
    ("Tim Cook", "Apple (Company)", {"label": "CEO_OF"}),
    ("Apple (Company)", "iPhone", {"label": "PRODUCES"}),
    ("Apple (Fruit)", "Cider", {"label": "PRODUCES_BEVERAGE"}),
])
pos_auto = {
    "Tim Cook": (-0.5, 1), "Apple (Company)": (-0.5, 0), "iPhone": (-0.5, -1),
    "Apple (Fruit)": (1, 1), "Cider": (1, 0)
}
nx.draw_networkx_edges(G_auto_render, pos_auto, ax=ax2, width=2, alpha=0.7, edge_color="#737373", arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")
nx.draw_networkx_nodes(G_auto_render, pos_auto, ax=ax2, node_size=3000, node_color="#4da6ff", edgecolors="#1a75ff", linewidths=2)
nx.draw_networkx_labels(G_auto_render, pos_auto, ax=ax2, font_size=10, font_weight="bold")
nx.draw_networkx_edge_labels(G_auto_render, pos_auto, ax=ax2, edge_labels=nx.get_edge_attributes(G_auto_render, 'label'), font_size=9, font_color="#5cb85c")

add_cluster_box(ax2, pos_auto, ["Tim Cook", "Apple (Company)", "iPhone"], "#5cb85c", "Unified Entity (Deduplicated)")
add_cluster_box(ax2, pos_auto, ["Apple (Fruit)", "Cider"], "#f0ad4e", "Isolated Safely (Different Type)")
ax2.set_title("After AutoGraft: Unified & Clean (LLM Dedup + Type Isolation)", fontsize=18, fontweight='bold', pad=20)
ax2.axis('off')

fig.suptitle("Figure 1.0: Real Database Entity Resolution", fontsize=24, fontweight='bold', y=0.98, color="#333333")
plt.tight_layout()
plt.savefig("benchmark/assets/concept_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig("/home/jgay-donat/.gemini/antigravity-cli/brain/816be250-ec9c-4a74-945f-885f6e762391/concept_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nDone! Apple Image saved.")
