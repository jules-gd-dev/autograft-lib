import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch

# Style matching the other charts
plt.style.use('seaborn-v0_8-whitegrid')

BG_COLOR = "#FFFFFF"
NODE_COLOR_TECH = "#4da6ff"
NODE_COLOR_AGRI = "#ffcc66"
EDGE_COLOR = "#737373"
TEXT_COLOR = "#333333"

def add_cluster_box(ax, pos, nodes, color, label=None):
    x_coords = [pos[n][0] for n in nodes]
    y_coords = [pos[n][1] for n in nodes]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    pad = 0.35
    
    bbox = FancyBboxPatch((min_x - pad, min_y - pad),
                          max_x - min_x + 2*pad,
                          max_y - min_y + 2*pad,
                          boxstyle="round,pad=0.1,rounding_size=0.2",
                          edgecolor=color,
                          facecolor=color,
                          alpha=0.12,
                          linewidth=2.5,
                          zorder=1)
    ax.add_patch(bbox)
    
    if label:
        ax.text(min_x, max_y + pad + 0.1, label, color=color, 
                fontsize=11, fontweight='bold', ha='left')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), facecolor=BG_COLOR)
plt.subplots_adjust(wspace=0.1)

# ==========================================
# GRAPH 1: NAIVE EXTRACTION (BEFORE)
# ==========================================
G1 = nx.DiGraph()
# Tech fragmented
G1.add_edge("T. Cook", "Apple", label="CEO_OF")
G1.add_edge("Apple Inc.", "iPhone", label="PRODUCES")
G1.add_edge("Apple Inc.", "MacBook", label="PRODUCES")
G1.add_edge("Steve Jobs", "Apple Computer", label="FOUNDER_OF")
# Agriculture
G1.add_edge("Orchard Farms", "Apple", label="GROWS")
G1.add_edge("Orchard Farms", "Banana", label="GROWS")

pos1 = {
    # Tech
    "T. Cook": (-1, 2), "Apple": (-0.5, 1),
    "Apple Inc.": (1, 1), "iPhone": (0.5, 2), "MacBook": (1.5, 2),
    "Steve Jobs": (2.5, 2), "Apple Computer": (2.5, 1),
    # Agri
    "Orchard Farms": (-1, -1), "Banana": (-1.5, -2)
}
# Adjust Apple (fruit) position for Agri
pos1["Apple"] = (-0.5, 1) # This is the collision node!

# Draw
nx.draw_networkx_edges(G1, pos1, ax=ax1, width=2, alpha=0.7, edge_color=EDGE_COLOR, arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")
nx.draw_networkx_nodes(G1, pos1, ax=ax1, node_size=3000, node_color=NODE_COLOR_TECH, edgecolors="#1a75ff", linewidths=2)
# Re-color agri nodes
agri_nodes = ["Orchard Farms", "Banana"]
nx.draw_networkx_nodes(G1, pos1, nodelist=agri_nodes, ax=ax1, node_size=3000, node_color=NODE_COLOR_AGRI, edgecolors="#e69900", linewidths=2)

nx.draw_networkx_labels(G1, pos1, ax=ax1, font_size=10, font_weight="bold", font_color=TEXT_COLOR)
nx.draw_networkx_edge_labels(G1, pos1, ax=ax1, edge_labels=nx.get_edge_attributes(G1, 'label'), font_size=9, font_color="#d9534f")

# Boxes
add_cluster_box(ax1, pos1, ["T. Cook", "Apple", "Orchard Farms", "Banana"], "#d9534f", "Semantic Collision & Duplicates")
add_cluster_box(ax1, pos1, ["Apple Inc.", "iPhone", "MacBook"], "#d9534f", "Isolated Context")
add_cluster_box(ax1, pos1, ["Steve Jobs", "Apple Computer"], "#d9534f")

ax1.set_title("Before AutoGraft: Fragmented & Collided Knowledge", fontsize=16, fontweight='bold', pad=20)
ax1.axis('off')

# ==========================================
# GRAPH 2: AUTOGRAFT (AFTER)
# ==========================================
G2 = nx.DiGraph()
# Tech unified
G2.add_edge("Tim Cook", "Apple Inc.", label="CEO_OF")
G2.add_edge("Apple Inc.", "iPhone", label="PRODUCES")
G2.add_edge("Apple Inc.", "MacBook", label="PRODUCES")
G2.add_edge("Steve Jobs", "Apple Inc.", label="FOUNDER_OF")
# Agriculture strictly separated (Semantics understood!)
G2.add_edge("Orchard Farms", "Apple", label="GROWS")
G2.add_edge("Orchard Farms", "Banana", label="GROWS")

pos2 = {
    # Tech
    "Apple Inc.": (1, 1),
    "Tim Cook": (0, 2), "iPhone": (1, 2), "MacBook": (2, 2), "Steve Jobs": (2, 0),
    # Agri
    "Orchard Farms": (-2, 0), "Apple": (-1, -1), "Banana": (-3, -1)
}

# Draw
nx.draw_networkx_edges(G2, pos2, ax=ax2, width=2, alpha=0.7, edge_color=EDGE_COLOR, arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")
nx.draw_networkx_nodes(G2, pos2, ax=ax2, node_size=3000, node_color=NODE_COLOR_TECH, edgecolors="#1a75ff", linewidths=2)
# Re-color agri nodes
agri_nodes2 = ["Orchard Farms", "Apple", "Banana"]
nx.draw_networkx_nodes(G2, pos2, nodelist=agri_nodes2, ax=ax2, node_size=3000, node_color=NODE_COLOR_AGRI, edgecolors="#e69900", linewidths=2)

nx.draw_networkx_labels(G2, pos2, ax=ax2, font_size=10, font_weight="bold", font_color=TEXT_COLOR)
nx.draw_networkx_edge_labels(G2, pos2, ax=ax2, edge_labels=nx.get_edge_attributes(G2, 'label'), font_size=9, font_color="#5cb85c")

# Boxes
add_cluster_box(ax2, pos2, ["Apple Inc.", "Tim Cook", "iPhone", "MacBook", "Steve Jobs"], "#5cb85c", "Unified Corporate Entity")
add_cluster_box(ax2, pos2, ["Orchard Farms", "Apple", "Banana"], "#f0ad4e", "Distinct Agricultural Entity")

ax2.set_title("After AutoGraft: Deduplicated & Semantically Accurate", fontsize=16, fontweight='bold', pad=20)
ax2.axis('off')

# General Title
fig.suptitle("Figure 1.0: Entity Resolution & Semantic Disambiguation", fontsize=22, fontweight='bold', y=0.98, color=TEXT_COLOR)

# Save
output_path = "/home/jgay-donat/.gemini/antigravity-cli/brain/816be250-ec9c-4a74-945f-885f6e762391/concept_comparison.png"
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=BG_COLOR)
print("Comparison graph generated.")
