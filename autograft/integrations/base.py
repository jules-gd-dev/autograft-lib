"""Base integration module for AutoGraft."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from autograft.config import AutoGraftConfig
from autograft.db.cypher import quote_label
from autograft.models.entities import Entity, ExistingNode

logger = logging.getLogger(__name__)


class BaseGraphMiddleware(ABC):
    """Base middleware for graph stores."""

    def __init__(
        self,
        config: AutoGraftConfig | None = None,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        self.config = config or AutoGraftConfig()
        if model:
            self.config.model = model
        if api_key:
            self.config.api_key = api_key
        if api_base:
            self.config.api_base = api_base
        self._node_cache: dict[str, Any] = {}

    @abstractmethod
    def _execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Executes a graph query and returns a list of dictionaries."""

    def _ensure_vector_index(self, label: str) -> None:
        if not self.config.auto_create_indexes:
            return
        index_name = f"autograft_{label.lower()}_vector_index"
        if index_name in self._node_cache:
            return
        self._node_cache[index_name] = True
        try:
            query = f"""
            CREATE VECTOR INDEX {quote_label(index_name)} IF NOT EXISTS
            FOR (n:{quote_label(label)}) ON (n.{self.config.embedding_attr})
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {self.config.embedding_dimension},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
            self._execute_query(query)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to create vector index {index_name}: {e}")

    def find_exact_candidates(self, entity: Entity) -> list[ExistingNode]:
        """Queries Neo4j for nodes matching the entity type to perform deterministic match."""
        query = f"""
        MATCH (n:{quote_label(entity.type)})
        WHERE n.{self.config.id_attr} = $name OR $name IN n.{self.config.aliases_attr}
        RETURN n.{self.config.id_attr} AS id, n.{self.config.aliases_attr} AS aliases
        LIMIT 100
        """
        try:
            results = self._execute_query(query, params={"name": entity.canonical_name})
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
            logger.warning(
                f"Exact-candidate query failed for '{entity.canonical_name}' "
                f"(db unreachable? failing open, entity will be treated as new): {e}"
            )
            return []

    def persist_alias(self, node_id: str, alias: str | None) -> None:
        """Best-effort append of the incoming name to the matched node's aliases
        so later repeats hit Layer 1 deterministically instead of the LLM."""
        if not alias:
            return
        id_attr, aliases_attr = self.config.id_attr, self.config.aliases_attr
        query = (
            f"MATCH (n) WHERE n.{id_attr} = $id "
            f"SET n.{aliases_attr} = CASE WHEN $alias IN coalesce(n.{aliases_attr}, []) "
            f"THEN n.{aliases_attr} ELSE coalesce(n.{aliases_attr}, []) + $alias END"
        )
        try:
            self._execute_query(query, params={"id": node_id, "alias": alias})
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Failed to persist alias '{alias}' on '{node_id}': {e}")

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
        RETURN node.{self.config.id_attr} AS id, node.{self.config.aliases_attr} AS aliases, node.{self.config.embedding_attr} AS embedding
        """
        try:
            results = self._execute_query(
                query,
                params={
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
                            embedding=r.get("embedding"),
                        )
                    )
            return nodes
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Vector search failed for {entity.type} (index might be missing, "
                f"failing open to no candidates): {e}"
            )
            return []
