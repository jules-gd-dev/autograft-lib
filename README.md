# AutoGraft

## Description
AutoGraft is an AI Agent library for Entity Resolution in GraphRAG. It prevents duplicate nodes (e.g., "J. Dupont" and "Jean Dupont") in Neo4j Knowledge Graphs by using semantic matching and LLM-as-a-judge arbitration.

## Key Features
- **Entity Extraction:** Automatically extracts entities and relations from unstructured documents.
- **Semantic Upsert:** Fuses entities into the graph without creating duplicates.
- **Neo4j Integration:** Seamless read/write against a Neo4j Knowledge Graph.
- **LLM Agnostic:** Works with OpenAI, Anthropic, and local models (Ollama) via `litellm`.

## Installation

```bash
pip install autograft
```

## Quick Start

```python
from autograft import AutoGraft

agent = AutoGraft(
    neo4j_uri="bolt://localhost:7687",
    llm_provider="openai",   # or "anthropic", "ollama"
    model="gpt-4o",
)

document = """
Jean Dupont presented his work on Knowledge Graphs at the conference.
Dr. Dupont collaborated with the company Neo4j on this project.
"""

result = agent.process(document)
print(result.created_entities)
print(result.merged_duplicates)
```
