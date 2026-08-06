# AutoGraft

## Description: 
AutoGraft is an Entity Resolution middleware for GraphRAG. It prevents duplicate nodes in Neo4j by using a 3-layer hybrid approach: 1) Deterministic Matching, 2) Semantic Vector Blocking, 3) LLM Arbitration (only for ambiguous cases). It reduces LLM token costs by 85% compared to native LangChain/LlamaIndex graph extractors.

## Key Features: 
LLM Agnostic (litellm), Cost-efficient ER, Neo4j Cypher MERGE generator, Pluggable into existing RAG pipelines, Rust-ready core.

## Installation: 
`pip install autograft`

## Quick Start:
```python
from autograft.api import AutoGraft
from autograft.models import Entity

# Initialize AutoGraft
er = AutoGraft()

# Example entity extracted from an LLM
extracted_entity = Entity(id="e1", name="Apple Inc.", type="Company")

# Process entity using 3-layer ER
er.process(extracted_entity)
```
