# AutoGraft 🚀

The cost-efficient Entity Resolution middleware for GraphRAG.

Stop duplicating entities in your Neo4j Knowledge Graph. AutoGraft intercepts entities extracted by LangChain or LlamaIndex, uses a 3-layer hybrid approach (Deterministic -> Vector -> LLM) to merge duplicates, and generates clean Cypher queries.

## Why AutoGraft?

- 🧠 **LLM-Agnostic**: Works with OpenAI, Groq, Ollama, OpenRouter via litellm.
- 💸 **Massive Cost Savings**: Reduces Entity Resolution token costs by up to 100% by resolving locally.
- 🔗 **Plug & Play**: Drop-in replacement before your Neo4j database.
- ⚡ **Blazing Fast**: C/C++ (RapidFuzz) and NumPy local matching.

## 📊 Performance Benchmark (AutoGraft vs LangChain)

| Metric | LangChain (Full LLM) | AutoGraft (Hybrid) |
| :--- | :--- | :--- |
| Execution Time (50 phrases) | 5.46s | 0.35s |
| LLM API Calls | 5 | 1 |
| Total Tokens Used | 950 | 180 |
| Neo4j Duplicates Created | 158 | 0 |

## 🏗️ Architecture (3-Layer Short-Circuit)

1. **Layer 1 (Deterministic)**: Exact string match via rapidfuzz (0 tokens).
2. **Layer 2 (Semantic)**: Vector cosine similarity via numpy (0 tokens).
3. **Layer 3 (LLM Arbiter)**: LiteLLM call ONLY for ambiguous cases (e.g., "J. Dupont" vs "Jean Dupont").

## 🚀 Quick Start

```python
from autograft import Entity, ExistingNode, resolve_and_generate_cypher

# 1. Define existing graph node
existing_node = ExistingNode(node_id="1", canonical_name="Apple Inc.", type="Company")

# 2. New entity extracted from document
new_entity = Entity(canonical_name="Apple", type="Company")

# 3. Resolve and generate Cypher
cypher_query = resolve_and_generate_cypher(new_entity, [existing_node])
print(cypher_query)
# Output: MATCH (n:Entity {node_id: '1'}) SET n.aliases = $aliases RETURN n
```

## License

MIT
