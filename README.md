# AutoGraft

The cost-efficient Entity Resolution middleware for GraphRAG.

Stop duplicating entities in your Neo4j Knowledge Graph. AutoGraft intercepts entities extracted by LangChain or LlamaIndex, uses a 3-layer hybrid approach (Deterministic -> Vector -> LLM) to merge duplicates, and generates clean Cypher queries.

## Why AutoGraft?

- **LLM-Agnostic**: Works with OpenAI, Groq, Ollama, OpenRouter via litellm.
- **Massive Cost Savings**: Reduces Entity Resolution token costs by up to 100% by resolving locally.
- **Plug & Play**: Drop-in replacement before your Neo4j database.
- **Blazing Fast**: C/C++ (RapidFuzz) and NumPy local matching.

---

## Performance Benchmark (200 Documents / 4 Industries)

Evaluated across **200 real-world enterprise documents** spanning 4 key scenarios: **Legal & Compliance**, **Tech & Enterprise Software**, **Insurance & Risk Management**, and **Finance & Investment Banking** (with complex acronyms like `GDPR`, `K8s`, `AWS`, `D&O`, `EBITDA`, `KYC/AML`, `SOFR`, `SCOTUS`).

| Metric | LangChain Naive (100% LLM) | AutoGraft Hybrid ER | Improvement / Savings |
| :--- | :---: | :---: | :---: |
| **Processed Documents** | 200 documents | 200 documents | Standardized Baseline |
| **Extracted Entities** | 742 entities | 742 entities | Identical Extraction Set |
| **LLM ER API Calls** | 742 calls | **0 calls** | **100% Local Short-Circuit** |
| **Tokens Consumed** | 207,760 tokens | **0 tokens** | **100% Token Savings** |
| **Duplicates Avoided (`MATCH`)** | `0` *(188 duplicates created!)* | **188 queries** | **188 Duplicates Avoided** |
| **New Entities Created (`MERGE`)** | 742 queries | **554 queries** | Clean Deduplicated Graph |

### Figure 1.1: Multi-Industry Benchmark Metrics
![Figure 1.1](benchmark/assets/macro_benchmark_metrics.png)

### Figure 1.2: Cost Scaling Projection (Up to 1,000,000 Documents)
For 1,000,000 documents, AutoGraft reduces projected LLM Entity Resolution API costs from **~$207.76** down to **~$0.00**, achieving **> 99.9% cost savings at enterprise scale**.

![Figure 1.2](benchmark/assets/macro_cost_scaling_1m.png)

### Figure 1.3: Resolution Accuracy Precision by Industry Sector
![Figure 1.3](benchmark/assets/macro_accuracy_by_industry.png)

*For complete evaluation methodology and dataset documentation, see [BENCHMARK.md](BENCHMARK.md).*

---

## Architecture (3-Layer Short-Circuit)

1. **Layer 1 (Deterministic)**: Exact string match & alias matching via rapidfuzz (0 tokens, 0.1ms).
2. **Layer 2 (Semantic)**: Vector cosine similarity via numpy (0 tokens, 0.5ms).
3. **Layer 3 (LLM Arbiter)**: LiteLLM call ONLY for residual ambiguous cases (e.g., "J. Dupont" vs "Jean Dupont").

---

## Quick Start

```python
from autograft import Entity, ExistingNode, resolve_and_generate_cypher

# 1. Define existing graph node in Neo4j
existing_node = ExistingNode(node_id="n1", canonical_name="Apple Inc.", type="Company", aliases=["Apple"])

# 2. New entity extracted from document
new_entity = Entity(canonical_name="Apple", type="Company")

# 3. Resolve and generate Cypher query
cypher_query = resolve_and_generate_cypher(new_entity, [existing_node])
print(cypher_query)
# Output: MATCH (n:Company {node_id: 'n1'}) SET n.aliases = coalesce(n.aliases, []) + ['Apple'] RETURN n;
```

---

## License

MIT
