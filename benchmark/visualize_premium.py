import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import numpy as np

# --- Premium Dark Mode Palette ---
BG_COLOR = "#0D1117"  # GitHub dark background
NODE_COLOR = "#58A6FF"
EDGE_COLOR = "#8B949E"
TEXT_COLOR = "#C9D1D9"
BOX_COLOR_BAD = "#F85149"  # Red-ish for bad fragmentation
BOX_COLOR_GOOD = "#3FB950" # Green-ish for good consolidation

def draw_premium_graph(G, pos, title, output_path, clusters, is_bad=True):
    fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Draw Bounding Boxes (Clusters)
    box_color = BOX_COLOR_BAD if is_bad else BOX_COLOR_GOOD
    for cluster_nodes in clusters:
        # Calculate bounding box for the cluster
        x_coords = [pos[n][0] for n in cluster_nodes]
        y_coords = [pos[n][1] for n in cluster_nodes]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Add padding
        pad_x = 0.25
        pad_y = 0.25
        
        bbox = FancyBboxPatch((min_x - pad_x, min_y - pad_y),
                              max_x - min_x + 2*pad_x,
                              max_y - min_y + 2*pad_y,
                              boxstyle="round,pad=0.1,rounding_size=0.1",
                              edgecolor=box_color,
                              facecolor=box_color,
                              alpha=0.15,
                              linewidth=2,
                              zorder=1)
        ax.add_patch(bbox)

    # Draw Edges
    nx.draw_networkx_edges(G, pos, ax=ax, width=2, alpha=0.8, edge_color=EDGE_COLOR, 
                           arrows=True, arrowsize=25, connectionstyle="arc3,rad=0.1")
    
    # Draw Edge Labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    # Custom edge labels drawing for better aesthetics
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, 
                                 font_size=10, font_color="#D2A8FF", 
                                 bbox=dict(facecolor=BG_COLOR, edgecolor='none', alpha=0.7),
                                 connectionstyle="arc3,rad=0.1")

    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3500, node_color=NODE_COLOR, 
                           edgecolors="#1F6FEB", linewidths=3)
    
    # Draw Node Labels
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_family="sans-serif", 
                            font_weight="bold", font_color="#0D1117")
    
    # Add Title and Legend
    plt.title(title, fontsize=18, fontweight='bold', color=TEXT_COLOR, pad=20)
    
    # Add a custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=NODE_COLOR, markersize=15, label='Extracted Entity'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=box_color, alpha=0.3, markersize=15, label='Isolated Cluster' if is_bad else 'Unified Entity')
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor=BG_COLOR, edgecolor=EDGE_COLOR, 
              labelcolor=TEXT_COLOR, fontsize=12)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

# --- Graph 1: Native Extractors (Fragmented) ---
G1 = nx.DiGraph()
G1.add_edge("T. Cook", "Apple", label="CEO_OF")
G1.add_edge("iPhone", "Apple Inc.", label="PRODUCT_OF")
G1.add_edge("Steve Jobs", "Apple Computer", label="FOUNDER_OF")

pos1 = {
    "T. Cook": (-1, 1), "Apple": (-1, 0),
    "iPhone": (0, 1), "Apple Inc.": (0, 0),
    "Steve Jobs": (1, 1), "Apple Computer": (1, 0)
}
clusters1 = [["T. Cook", "Apple"], ["iPhone", "Apple Inc."], ["Steve Jobs", "Apple Computer"]]

draw_premium_graph(
    G1, pos1, 
    "Figure 1.1: Native Extraction (Fragmented Knowledge)", 
    "/home/jgay-donat/Work/autograft/lib/benchmark/assets/figure1_fragmented.png", 
    clusters1, is_bad=True
)

# --- Graph 2: AutoGraft (Unified) ---
G2 = nx.DiGraph()
G2.add_edge("T. Cook", "Apple Inc.", label="CEO_OF")
G2.add_edge("iPhone", "Apple Inc.", label="PRODUCT_OF")
G2.add_edge("Steve Jobs", "Apple Inc.", label="FOUNDER_OF")

pos2 = {
    "T. Cook": (-1, 1), 
    "iPhone": (0, 1), 
    "Steve Jobs": (1, 1), 
    "Apple Inc.": (0, 0)
}
clusters2 = [["T. Cook", "iPhone", "Steve Jobs", "Apple Inc."]]

draw_premium_graph(
    G2, pos2, 
    "Figure 1.2: AutoGraft Extraction (Unified & Deduplicated)", 
    "/home/jgay-donat/Work/autograft/lib/benchmark/assets/figure2_autograft.png", 
    clusters2, is_bad=False
)

print("Premium graphs generated.")
