# AutoGraft Agent Instructions

## Project Overview

AutoGraft is a Python library that automates Entity Resolution for GraphRAG architectures. It extracts entities from documents and uses a multi-layered approach (Exact Match -> Vector Similarity -> LLM Arbitration) to merge them into a Neo4j graph without creating duplicates.

## Development Rules

AI Agents contributing to this repository MUST strictly follow the rules defined in GUIDELINES.md.

- Do not exceed 150 lines per file.
- Ensure 97% test coverage with pytest.
- Use type hints.
- Commit frequently using Conventional Commits on the dev branch.

## Tech Stack

- Python 3.10+
- litellm for LLM provider abstraction (OpenAI, Anthropic, Ollama).
- neo4j python driver for graph database operations.
- pydantic for data validation.
