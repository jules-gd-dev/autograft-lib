"""Pydantic models for entities, existing graph nodes, and ER layer match results."""
from typing import Optional
from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Represents an entity extracted from a document."""

    canonical_name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    embedding: Optional[list[float]] = None


class ExistingNode(BaseModel):
    """Represents an entity already stored in Neo4j."""

    node_id: str
    canonical_name: str
    type: str
    aliases: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None


class MatchResult(BaseModel):
    """Represents the result of a Layer 1 ER check."""

    is_match: bool
    matched_node_id: Optional[str] = None
    score: float = 0.0
    layer: str = "deterministic"
