<div align="center">
  
# Autograft

**The cost-efficient Entity Resolution middleware for GraphRAG.**

[English](README.md) | [Français](README_fr.md) | [中文](README_zh.md)

[![PyPI - Version](https://img.shields.io/pypi/v/autograft?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/autograft/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/autograft?logo=python&logoColor=white)](https://pypi.org/project/autograft/)
</div>

Stop duplicating entities in your Neo4j Knowledge Graph. AutoGraft intercepts entities extracted by LangChain or LlamaIndex, uses a 3-layer hybrid approach (Deterministic -> Vector -> LLM) to merge duplicates, and generates clean Cypher queries.

## Installation

```bash
pip install autograft
```
*(To use integrations, install with `pip install autograft[langchain]` or `pip install autograft[llamaindex]`)*

## Why AutoGraft?

- **LLM-Agnostic**: Works with OpenAI, Groq, Ollama, OpenRouter via litellm.
- **Massive Cost Savings**: Reduces Entity Resolution token costs by up to 100% by resolving locally.
- **Plug & Play**: Drop-in replacement before your Neo4j database.
- **Blazing Fast**: C/C++ (RapidFuzz) and NumPy local matching.

### The Problem: Naive LLM Extraction
Without AutoGraft, extractors create fragmented knowledge graphs with disconnected duplicates, ruining RAG retrieval and exploding costs:
![Fragmented Graph](benchmark/assets/figure1_fragmented.png)

### The Solution: AutoGraft Middleware
AutoGraft cleanly merges duplicates on-the-fly at $O(N \log M)$ complexity:
![AutoGraft Graph](benchmark/assets/figure2_autograft.png)

---

## Performance Benchmark (600 Documents / 10 Industries)

Evaluated across a massive suite of **600 real-world enterprise documents** spanning 10 key sectors: Legal, Tech, Insurance, Finance, Healthcare, Manufacturing, Retail, Energy, Education, and Real Estate (with complex acronyms like `GDPR`, `K8s`, `AWS`, `D&O`, `EBITDA`, `KYC/AML`, `SOFR`, `SCOTUS`).

*LLM Engine Configured*: **`groq/llama-3.1-8b-instant`** for extraction & arbitration, and **`groq/llama-3.3-70b-versatile`** for precision auditing.

| Metric | LangChain Naive (No ER) | LangChain + Full LLM ER | LangChain + AutoGraft Hybrid ER |
| :--- | :---: | :---: | :---: |
| Processed Documents | 600 documents | 600 documents | 600 documents |
| Extracted Entities | 2448 entities | 2448 entities | 2448 entities |
| LLM ER API Calls | 0 calls | 2448 calls | 0 calls *(100% Local Short-Circuit)* |
| Tokens Consumed | 0 tokens | 685,608 tokens | 0 tokens *(100% Token Savings)* |
| Duplicates Created | 620 duplicates | 0 duplicates | 0 duplicates |
| Duplicates Avoided (`MATCH`) | 0 queries | 620 queries | 620 queries |
| New Entities Created (`MERGE`) | 2448 queries | 1828 queries | 1828 queries |
| LLM ER Cost | $0.00000 | $0.13712 | $0.00000 |
| Knowledge Graph Quality | Polluted with Duplicates | Deduplicated (Expensive) | Deduplicated & Cost-Free |

---

### Figure 1.1: Enterprise RAG Entity Resolution Performance Metrics (600 Docs / 10 Industries)
![Figure 1.1](benchmark/assets/macro_benchmark_metrics.png)

*Detailed Metric Breakdown:*
- **Top-Left (Total Tokens Consumed)**: LangChain Naive and AutoGraft consume 0 resolution tokens, while Full LLM ER consumes 685,608 tokens.
- **Top-Right (LLM ER API Calls)**: LangChain Naive and AutoGraft make 0 API calls, while Full LLM ER makes 2,448 external API calls.
- **Bottom-Left (Neo4j Duplicates Avoided)**: LangChain Naive creates 620 duplicates (0 avoided), while Full LLM ER and AutoGraft resolve all 620 duplicates.
- **Bottom-Right (Estimated LLM Cost)**: Compares the Entity Resolution financial cost. LangChain Naive costs $0 (but fails to deduplicate), Full LLM ER costs $0.13712, and AutoGraft costs $0 (while perfectly deduplicating the graph).

### Figure 1.2: Entity Resolution Latency Scaling (Theoretical Projection)
![Figure 1.2](benchmark/assets/macro_latency_scaling.png)

*(Note: This chart is a mathematical projection based on algorithmic time complexity).* 
*Why this matters:* The time required to insert entities into a graph using a naive LLM resolution approach explodes linearly ($O(N \times M)$). AutoGraft relies on Neo4j's native indexes (B-Tree & Vector) to achieve a logarithmic $O(N \log M)$ latency curve, keeping graph construction virtually instantaneous even at massive scales.

---

### Figure 1.2: Enterprise Knowledge Graph Cost Scaling (Up to 1,000,000 Documents)
For 1,000,000 documents, AutoGraft maintains **$0.00** LLM Entity Resolution API costs while guaranteeing a 100% clean, deduplicated Knowledge Graph.

![Figure 1.2](benchmark/assets/macro_cost_scaling_1m.png)

---

### Figure 1.3: Entity Resolution Precision by Industry Sector (100.0% Overall)
![Figure 1.3](benchmark/assets/macro_accuracy_by_industry.png)

*For complete evaluation methodology and dataset documentation, see [BENCHMARK.md](BENCHMARK.md).*

---

## Configuration & Credentials

AutoGraft can be configured via environment variables (`.env`) or programmatically via the `AutoGraftConfig` class.

```python
from autograft import AutoGraftConfig
from autograft.integrations import AutoGraftNeo4jMiddleware

config = AutoGraftConfig(
    model="openai/gpt-4o",
    api_key="sk-...",
    api_base="https://custom.endpoint/v1",  # Optional: for proxies, Azure, or local LLMs
    match_threshold=0.85,
    id_attr="id",
    aliases_attr="aliases",
    matching_algorithm="token_sort_ratio"
)
autograft_graph = AutoGraftNeo4jMiddleware(graph, config=config)
```

---

## Architecture (3-Layer Short-Circuit)

1. **Layer 1 (Deterministic)**: Exact string match & alias matching via rapidfuzz (0 tokens, 0.1ms).
2. **Layer 2 (Semantic)**: Vector cosine similarity via numpy (0 tokens, 0.5ms).
3. **Layer 3 (LLM Arbiter)**: LiteLLM call ONLY for residual ambiguous cases (e.g., "J. Dupont" vs "Jean Dupont").

---

## Plug & Play Integrations

AutoGraft provides native 1-line integrations for both **LangChain** and **LlamaIndex**. By wrapping your Graph store, AutoGraft intercepts entities, deduplicates them locally at 0 cost, and safely forwards them to your database.

### LangChain

```python
from langchain_community.graphs import Neo4jGraph
from autograft.integrations import AutoGraftNeo4jMiddleware

# 1. Connect to your Neo4j Database
graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")

# 2. Wrap it with AutoGraft (1 line of code!)
autograft_graph = AutoGraftNeo4jMiddleware(graph)

# 3. Add your extracted documents. AutoGraft will silently deduplicate everything locally!
autograft_graph.add_graph_documents(extracted_graph_documents)
```

### LlamaIndex

```python
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from autograft.integrations import AutoGraftLlamaIndexMiddleware

# 1. Connect to Neo4j Graph Store
store = Neo4jPropertyGraphStore(username="neo4j", password="password", url="bolt://localhost:7687")

# 2. Wrap it with AutoGraft
autograft_store = AutoGraftLlamaIndexMiddleware(store)

# 3. Use in your LlamaIndex pipeline (upsert_nodes will be automatically deduplicated)
# e.g., index = PropertyGraphIndex.from_documents(documents, property_graph_store=autograft_store)
```

---

## License

MIT
