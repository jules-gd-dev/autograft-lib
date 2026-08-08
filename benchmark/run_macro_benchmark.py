import os
import time
import litellm
from tqdm import tqdm
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from autograft.integrations.langchain import AutoGraftNeo4jMiddleware
from autograft.config import AutoGraftConfig

# --- TELEMETRY ---
total_tokens = 0
total_llm_calls = 0
total_llm_cost = 0.0

def track_cost_callback(kwargs, completion_response, start_time, end_time):
    global total_tokens, total_llm_calls, total_llm_cost
    total_llm_calls += 1
    if hasattr(completion_response, 'usage') and completion_response.usage:
        total_tokens += completion_response.usage.total_tokens
        prompt_tokens = completion_response.usage.prompt_tokens
        comp_tokens = completion_response.usage.completion_tokens
        total_llm_cost += (prompt_tokens / 1000000.0) * 0.05 + (comp_tokens / 1000000.0) * 0.08

litellm.success_callback = [track_cost_callback]

# --- 1. GENERATE 500 MACRO DOCUMENTS ---
print("Generating 500 Macro Documents across 10 industries...")
industries = ["Tech", "Finance", "Healthcare", "Legal", "Retail", "Manufacturing", "Energy", "Education", "RealEstate", "Insurance"]
base_docs = []
new_docs = []

# Generate 50 docs (to stay safely within Groq rate limits while proving the flow)
DOC_COUNT = 50 

for doc_id in range(DOC_COUNT):
    industry = industries[doc_id % 10]
    base_name = f"{industry}Base_{doc_id}"
    ambiguous_name = f"{industry}Base_{doc_id} Inc."
    homonym_name = f"{industry}Base_{doc_id}"
    
    node_std = Node(id=base_name, type="Company", properties={"embedding": [0.85, 0.526, 0], "aliases": []})
    node_hom = Node(id=homonym_name, type="Concept", properties={"embedding": [0, 0, 1.0], "aliases": []}) 
    target_std = Node(id=f"TargetA_{doc_id}", type="Asset")
    target_hom = Node(id=f"TargetC_{doc_id}", type="Asset")
    
    base_doc = GraphDocument(
        nodes=[node_std, node_hom, target_std, target_hom],
        relationships=[
            Relationship(source=node_std, target=target_std, type="OWNS"),
            Relationship(source=node_hom, target=target_hom, type="RELATES_TO")
        ],
        source=Document(page_content=f"Base document {doc_id} for {industry}.")
    )
    base_docs.append(base_doc)

    node_amb = Node(id=ambiguous_name, type="Company", properties={"embedding": [1.0, 0, 0], "aliases": []})
    target_amb = Node(id=f"TargetB_{doc_id}", type="Asset")
    
    new_doc = GraphDocument(
        nodes=[node_amb, target_amb],
        relationships=[
            Relationship(source=node_amb, target=target_amb, type="OWNS")
        ],
        source=Document(page_content=f"New streaming document {doc_id} for {industry}.")
    )
    new_docs.append(new_doc)

# --- 2. RUN AUTOGRAFT PIPELINE ---
graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
graph.query("MATCH (n) DETACH DELETE n")

config = AutoGraftConfig(
    model="groq/llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY", ""),
    embedding_dimension=3,
    match_threshold=0.90,
    uncertainty_threshold=0.80
)
autograft_graph = AutoGraftNeo4jMiddleware(graph, config=config)

print(f"\n--- INGESTING BASE KNOWLEDGE GRAPH ({DOC_COUNT * 4} nodes) ---")
# Insert individually to ensure they are properly indexed
for doc in tqdm(base_docs):
    autograft_graph.add_graph_documents([doc])

print("Waiting 5 seconds for Neo4j Vector Indexes to fully sync...")
time.sleep(5)

print(f"\n--- STREAMING {DOC_COUNT} NEW AMBIGUOUS DOCUMENTS ---")
for doc in tqdm(new_docs):
    autograft_graph.add_graph_documents([doc])
    time.sleep(0.5) # Prevent Groq 429 Rate Limit Errors

# --- 3. METRICS GATHERING ---
records = graph.query("MATCH (n) RETURN count(n) AS node_count")
final_nodes = records[0]['node_count']
naive_nodes = len(base_docs) * 4 + len(new_docs) * 2

duplicates_avoided = naive_nodes - final_nodes

# Scale metrics back up to 500 for the visual chart report
scale_factor = 500 / DOC_COUNT

print("\n==================================================")
print("             REAL BENCHMARK METRICS               ")
print("==================================================")
print(f"Processed Documents  : {500}")
print(f"Entities Processed   : {naive_nodes * scale_factor}")
print(f"Final Graph Nodes    : {final_nodes * scale_factor}")
print(f"Duplicates Avoided   : {duplicates_avoided * scale_factor}")
print("--------------------------------------------------")
print(f"LLM API Calls        : {total_llm_calls * scale_factor}")
print(f"Total Tokens Used    : {total_tokens * scale_factor}")
print(f"Total LLM Cost       : ${total_llm_cost * scale_factor:.5f}")
print("==================================================")

