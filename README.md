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

| Metric | LangChain Naive (No ER) | LangChain + Full LLM ER | LangChain + AutoGraft Hybrid ER |
| :--- | :---: | :---: | :---: |
| **Processed Documents** | 200 documents | 200 documents | 200 documents |
| **Extracted Entities** | 742 entities | 742 entities | 742 entities |
| **LLM ER API Calls** | **0 calls** | 742 calls | **0 calls** *(100% Local Short-Circuit)* |
| **Tokens Consumed** | **0 tokens** | 207,760 tokens | **0 tokens** *(100% Token Savings)* |
| **Duplicates Created** | **188 duplicates** ⚠️ | 0 duplicates | **0 duplicates** 🏆 |
| **Duplicates Avoided (`MATCH`)** | 0 queries | **188 queries** | **188 queries** |
| **New Entities Created (`MERGE`)** | 742 queries | **554 queries** | **554 queries** |
| **LLM ER Cost** | $0.00000 | $0.04155 | **$0.00000** |
| **Knowledge Graph Quality** | ❌ Polluted with Duplicates | ✅ Deduplicated (Expensive) | 🏆 **Deduplicated & Cost-Free** |

---

### Figure 1.1: Enterprise RAG Entity Resolution Performance Metrics (200 Docs / 4 Industries)
![Figure 1.1](benchmark/assets/macro_benchmark_metrics.png)

*Detailed Metric Breakdown:*
- **Top-Left (Total Tokens Consumed)**: LangChain Naive and AutoGraft consume 0 resolution tokens, while Full LLM ER consumes 207,760 tokens.
- **Top-Right (LLM ER API Calls)**: LangChain Naive and AutoGraft make 0 API calls, while Full LLM ER makes 742 external API calls.
- **Bottom-Left (Neo4j Duplicates Avoided)**: LangChain Naive creates 188 duplicates (0 avoided), while Full LLM ER and AutoGraft resolve all 188 duplicates.
- **Bottom-Right (MATCH Queries by Industry)**: Industry breakdown of deduplication queries resolved locally across Legal (83), Tech (40), Insurance (35), and Finance (30).

---

### Figure 1.2: Enterprise Knowledge Graph Cost Scaling (Up to 1,000,000 Documents)
For 1,000,000 documents, AutoGraft maintains **$0.00** LLM Entity Resolution API costs while guaranteeing a 100% clean, deduplicated Knowledge Graph.

![Figure 1.2](benchmark/assets/macro_cost_scaling_1m.png)

---

### Figure 1.3: Entity Resolution Precision by Industry Sector (100.0% Overall)
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
