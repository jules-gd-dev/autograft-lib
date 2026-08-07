from typing import Protocol

from autograft.models.entities import Entity, ExistingNode


class GraphDatabaseClient(Protocol):
    """Protocol defining how AutoGraft queries the graph database."""

    def find_exact_candidates(self, entity: Entity) -> list[ExistingNode]:
        """Find nodes matching exactly by ID or label to be used for deterministic matching."""
        ...

    def find_semantic_candidates(
        self, entity: Entity, limit: int = 5
    ) -> list[ExistingNode]:
        """Find the most semantically similar nodes to the given entity."""
        ...


class ListDatabaseClient:
    """A client that operates on an in-memory list of nodes for backward compatibility."""
    
    def __init__(self, nodes: list[ExistingNode]):
        self.nodes = nodes
        
    def find_exact_candidates(self, entity: Entity) -> list[ExistingNode]:
        return [
            n for n in self.nodes 
            if n.type.lower() == entity.type.lower()
        ]
        
    def find_semantic_candidates(
        self, entity: Entity, limit: int = 5
    ) -> list[ExistingNode]:
        # To maintain the exact same behavior, we just return the filtered nodes,
        # and the semantic layer (find_semantic_match) will do the cosine similarity filtering itself.
        # But wait, find_semantic_match expects a list of candidates and returns the best.
        # In the original code, we just passed all filtered nodes.
        return self.find_exact_candidates(entity)
