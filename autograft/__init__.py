"""AutoGraft package top-level exports."""

from autograft.api.main import resolve_and_generate_cypher
from autograft.config import AutoGraftConfig
from autograft.core.batch import resolve_batch
from autograft.models.batch import BatchResult, ResolutionReport
from autograft.models.entities import Entity, ExistingNode, MatchResult

__all__ = [
    "AutoGraftConfig",
    "BatchResult",
    "Entity",
    "ExistingNode",
    "MatchResult",
    "ResolutionReport",
    "resolve_and_generate_cypher",
    "resolve_batch",
]
