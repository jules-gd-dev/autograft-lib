AutoGraft Agent Instructions
Project Overview

AutoGraft is a Python middleware library for Entity Resolution. It bridges the gap between raw LLM extraction and clean Knowledge Graph storage. It processes extracted entities and determines if they should be merged with existing graph nodes using a cost-effective, multi-layered approach.
Development Rules

AI Agents contributing to this repository MUST strictly follow the rules defined in GUIDELINES.md.

    Do not exceed 150 lines per file.
    Ensure 97% test coverage with pytest.
    Use type hints.
    Commit frequently using Conventional Commits on the dev branch.

Tech Stack

    Python 3.10+
    litellm for LLM provider abstraction (used ONLY for ambiguous entity arbitration).
    neo4j python driver for graph database operations.
    pydantic for data validation.
    rapidfuzz for deterministic string matching (Layer 1 of ER).

Architecture Note

The core ER logic must be decoupled from the Python API endpoints so it can be replaced by a compiled Rust/C++ extension in the future without breaking the API.
