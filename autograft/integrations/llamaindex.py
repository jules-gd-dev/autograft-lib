"""LlamaIndex integration for AutoGraft."""

from typing import TYPE_CHECKING, Any

from autograft.config import AutoGraftConfig
from autograft.core.resolver import resolve_entity
from autograft.integrations.base import BaseGraphMiddleware
from autograft.models.entities import Entity

if TYPE_CHECKING:
    from llama_index.core.graph_stores.types import EntityNode, Relation  # type: ignore
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore  # type: ignore
else:
    EntityNode = Any
    Relation = Any
    Neo4jPropertyGraphStore = Any


class AutoGraftLlamaIndexMiddleware(BaseGraphMiddleware):
    """Plug & Play LlamaIndex PropertyGraphStore wrapper for zero-cost entity resolution."""

    def __init__(
        self,
        neo4j_store: Neo4jPropertyGraphStore,
        config: AutoGraftConfig | None = None,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        super().__init__(config, model, api_key, api_base)
        self.store = neo4j_store

    def _execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        results, _ = self.store.structured_query(query, param_map=params or {})
        return results

    def upsert_nodes(self, nodes: list[EntityNode]) -> None:
        """Intercepts, deduplicates, and passes nodes to LlamaIndex's store."""
        for node in nodes:
            embedding = None
            if hasattr(node, "properties") and isinstance(node.properties, dict):
                embedding = node.properties.get(self.config.embedding_attr)

            entity = Entity(
                canonical_name=str(node.name),
                type=str(node.label),
                embedding=embedding,
            )
            match_result = resolve_entity(entity, db_client=self, config=self.config)

            if match_result.is_match:
                node.name = str(match_result.matched_node_id)
                self.persist_alias(
                    str(match_result.matched_node_id), match_result.new_alias
                )

        self.store.upsert_nodes(nodes)

    def upsert_relations(self, relations: list[Relation]) -> None:
        """Passes relations to LlamaIndex's store."""
        self.store.upsert_relations(relations)

    def __getattr__(self, name: str) -> Any:
        """Delegate all other method calls to the underlying store."""
        return getattr(self.store, name)
