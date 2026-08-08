import os
import matplotlib.pyplot as plt
import networkx as nx
from neo4j import GraphDatabase

URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ.get("NEO4J_PASSWORD", "password"))

def fetch_subgraph():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    # Fetch a small, highly connected component (limit to 50 relationships for readability)
    query = """
    MATCH (n)-[r]->(m)
    RETURN n.id AS source, type(r) AS rel_type, m.id AS target
    LIMIT 100
    """
    records = []
    try:
        with driver.session() as session:
            result = session.run(query)
            records = [record.data() for record in result]
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
    finally:
        driver.close()
    return records

records = fetch_subgraph()

if not records:
    print("No data found or connection failed. Please ensure Neo4j is running and contains data.")
else:
    G = nx.DiGraph()
    for row in records:
        source = str(row['source'])
        target = str(row['target'])
        rel = str(row['rel_type'])
        
        # truncate long names
        if len(source) > 20: source = source[:17] + "..."
        if len(target) > 20: target = target[:17] + "..."
        
        G.add_edge(source, target, label=rel)

    plt.figure(figsize=(16, 12))
    
    # Use spring layout for better node separation
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', alpha=0.9)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, edge_color='gray', arrows=True, arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_family="sans-serif", font_weight="bold")
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='red')

    plt.title("Neo4j Knowledge Graph Sample (AutoGraft)", fontsize=16)
    plt.axis('off')
    
    output_path = "/home/jgay-donat/.gemini/antigravity-cli/brain/816be250-ec9c-4a74-945f-885f6e762391/kg_visualization.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph visualization saved to {output_path}")
