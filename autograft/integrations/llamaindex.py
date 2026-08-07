"""LlamaIndex integration for AutoGraft."""

import logging
from typing import TYPE_CHECKING, Any

from autograft.config import AutoGraftConfig
from autograft.core.resolver import resolve_entity
from autograft.models.entities import Entity, ExistingNode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llama_index.core.graph_stores.types import EntityNode, Relation  # type: ignore
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore  # type: ignore
else:
    EntityNode = Any
    Relation = Any
    Neo4jPropertyGraphStore = Any


class AutoGraftLlamaIndexMiddleware:
    """Plug & Play LlamaIndex PropertyGraphStore wrapper for zero-cost entity resolution."""

    def __init__(
        self,
        neo4j_store: Neo4jPropertyGraphStore,
        config: AutoGraftConfig | None = None,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        self.store = neo4j_store
        self.config = config or AutoGraftConfig()
        if model:
            self.config.model = model
        if api_key:
            self.config.api_key = api_key
        if api_base:
            self.config.api_base = api_base
        self._node_cache: dict[str, Any] = {}

    def _ensure_vector_index(self, label: str) -> None:
        if not self.config.auto_create_indexes:
            return
        index_name = f"autograft_{label.lower()}_vector_index"
        if index_name in self._node_cache:
            return
        self._node_cache[index_name] = True
        try:
            query = f"""
            CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
            FOR (n:`{label}`) ON (n.{self.config.embedding_attr})
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {self.config.embedding_dimension},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
            self.store.structured_query(query)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to create vector index {index_name}: {e}")

    def find_exact_candidates(self, entity: Entity) -> list[ExistingNode]:
        """Queries Neo4j for exact match candidates."""
        query = f"""
        MATCH (n:`{entity.type}`)
        WHERE n.{self.config.id_attr} = $name OR $name IN n.{self.config.aliases_attr}
        RETURN n.{self.config.id_attr} AS id, n.{self.config.aliases_attr} AS aliases
        LIMIT 100
        """
        try:
            results, _ = self.store.structured_query(
                query, param_map={"name": entity.canonical_name}
            )
            nodes = []
            for r in results:
                node_id = str(r.get("id") or "")
                if node_id:
                    nodes.append(
                        ExistingNode(
                            node_id=node_id,
                            canonical_name=node_id,
                            type=entity.type,
                            aliases=r.get("aliases") or [],
                        )
                    )
            return nodes
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Query for exact candidates failed: {e}")
            return []

    def find_semantic_candidates(
        self, entity: Entity, limit: int = 5
    ) -> list[ExistingNode]:
        """Queries Neo4j vector index for semantic candidates."""
        if not entity.embedding:
            return []

        self._ensure_vector_index(entity.type)
        index_name = f"autograft_{entity.type.lower()}_vector_index"

        query = f"""
        CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
        YIELD node, score
        RETURN node.{self.config.id_attr} AS id, node.{self.config.aliases_attr} AS aliases, score
        """
        try:
            results, _ = self.store.structured_query(
                query,
                param_map={
                    "index_name": index_name,
                    "limit": limit,
                    "embedding": entity.embedding,
                },
            )
            nodes = []
            for r in results:
                node_id = str(r.get("id") or "")
                if node_id:
                    nodes.append(
                        ExistingNode(
                            node_id=node_id,
                            canonical_name=node_id,
                            type=entity.type,
                            aliases=r.get("aliases") or [],
                        )
                    )
            return nodes
        except Exception:  # noqa: BLE001
            return []

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
                # Canonicalize the new node's name (which acts as ID in LlamaIndex)
                node.name = str(match_result.matched_node_id)
            else:
                pass

        self.store.upsert_nodes(nodes)

    def upsert_relations(self, relations: list[Relation]) -> None:
        """Passes relations to LlamaIndex's store."""
        self.store.upsert_relations(relations)

    def __getattr__(self, name: str) -> Any:
        """Delegate all other method calls to the underlying store."""
        return getattr(self.store, name)
