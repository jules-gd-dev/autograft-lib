import matplotlib.pyplot as plt
import networkx as nx

# --- 1. Graph without AutoGraft (Messy/Duplicates) ---
G1 = nx.DiGraph()

# Add duplicate nodes
G1.add_edge("Apple Inc.", "iPhone", label="PRODUCES")
G1.add_edge("Apple", "iPhone 15", label="PRODUCES")
G1.add_edge("Apple Computer", "MacBook Pro", label="PRODUCES")
G1.add_edge("Tim Cook", "Apple Inc.", label="CEO_OF")
G1.add_edge("T. Cook", "Apple", label="CEO_OF")
G1.add_edge("Steve Jobs", "Apple Computer", label="FOUNDER_OF")

plt.figure(figsize=(12, 8))
pos1 = nx.spring_layout(G1, seed=42)
nx.draw_networkx_nodes(G1, pos1, node_size=2000, node_color='#ff9999', alpha=0.9)
nx.draw_networkx_edges(G1, pos1, width=1.5, alpha=0.7, edge_color='gray', arrows=True, arrowsize=20)
nx.draw_networkx_labels(G1, pos1, font_size=10, font_weight="bold")
edge_labels1 = nx.get_edge_attributes(G1, 'label')
nx.draw_networkx_edge_labels(G1, pos1, edge_labels=edge_labels1, font_size=9, font_color='red')

plt.title("Before AutoGraft: Messy Graph with Duplicates (O(N*M) Cost)", fontsize=14)
plt.axis('off')
out1 = "/home/jgay-donat/.gemini/antigravity-cli/brain/816be250-ec9c-4a74-945f-885f6e762391/graph_before.png"
plt.tight_layout()
plt.savefig(out1, dpi=300, bbox_inches='tight')
plt.close()

# --- 2. Graph with AutoGraft (Clean/Deduplicated) ---
G2 = nx.DiGraph()

# Add deduplicated nodes
G2.add_edge("Apple Inc.", "iPhone", label="PRODUCES")
G2.add_edge("Apple Inc.", "MacBook Pro", label="PRODUCES")
G2.add_edge("Tim Cook", "Apple Inc.", label="CEO_OF")
G2.add_edge("Steve Jobs", "Apple Inc.", label="FOUNDER_OF")

plt.figure(figsize=(12, 8))
pos2 = nx.spring_layout(G2, seed=42)
nx.draw_networkx_nodes(G2, pos2, node_size=2000, node_color='#99ff99', alpha=0.9)
nx.draw_networkx_edges(G2, pos2, width=1.5, alpha=0.7, edge_color='gray', arrows=True, arrowsize=20)
nx.draw_networkx_labels(G2, pos2, font_size=10, font_weight="bold")
edge_labels2 = nx.get_edge_attributes(G2, 'label')
nx.draw_networkx_edge_labels(G2, pos2, edge_labels=edge_labels2, font_size=9, font_color='green')

plt.title("After AutoGraft: Clean Deduplicated Graph (O(N log M) Cost)", fontsize=14)
plt.axis('off')
out2 = "/home/jgay-donat/.gemini/antigravity-cli/brain/816be250-ec9c-4a74-945f-885f6e762391/graph_after.png"
plt.tight_layout()
plt.savefig(out2, dpi=300, bbox_inches='tight')
plt.close()

print("Graphs generated successfully.")
