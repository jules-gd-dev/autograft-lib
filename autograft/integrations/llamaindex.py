"""LlamaIndex integration for AutoGraft."""

import contextlib
from typing import TYPE_CHECKING, Any

from autograft.core.resolver import resolve_entity
from autograft.models.entities import Entity, ExistingNode

if TYPE_CHECKING:
    from llama_index.core.graph_stores.types import EntityNode, Relation  # type: ignore
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore  # type: ignore
else:
    EntityNode = Any
    Relation = Any
    Neo4jPropertyGraphStore = Any


class AutoGraftLlamaIndexMiddleware:
    """Plug & Play LlamaIndex PropertyGraphStore wrapper for zero-cost entity resolution."""

    def __init__(self, neo4j_store: Neo4jPropertyGraphStore):
        self.store = neo4j_store
        self._node_cache: dict[str, list[ExistingNode]] = {}

    def _fetch_cached_nodes(self, labels: set[str]) -> list[ExistingNode]:
        """Fetches nodes from Neo4j natively, caching them locally per label."""
        nodes = []
        for label in labels:
            if label not in self._node_cache:
                self._node_cache[label] = []
                with contextlib.suppress(Exception):
                    # Access the underlying Neo4j driver from LlamaIndex's store
                    query = f"MATCH (n:`{label}`) RETURN n.id AS id, n.aliases AS aliases LIMIT 10000"
                    results, _ = self.store.structured_query(query)
                    for r in results:
                        node_id = str(r.get("id") or "")
                        aliases = r.get("aliases") or []
                        if node_id:
                            self._node_cache[label].append(
                                ExistingNode(
                                    node_id=node_id,
                                    canonical_name=node_id,
                                    type=label,
                                    aliases=aliases,
                                )
                            )
            nodes.extend(self._node_cache[label])
        return nodes

    def upsert_nodes(self, nodes: list[EntityNode]) -> None:
        """Intercepts, deduplicates, and passes nodes to LlamaIndex's store."""
        labels = {n.label for n in nodes if n.label}
        existing_nodes = self._fetch_cached_nodes(labels)

        for node in nodes:
            entity = Entity(canonical_name=str(node.name), type=str(node.label))
            match_result = resolve_entity(entity, existing_nodes)

            if match_result.is_match:
                # Canonicalize the new node's name (which acts as ID in LlamaIndex)
                node.name = str(match_result.matched_node_id)
            else:
                new_ex_node = ExistingNode(
                    node_id=str(node.name),
                    canonical_name=str(node.name),
                    type=str(node.label),
                    aliases=[str(node.name)],
                )
                if node.label not in self._node_cache:
                    self._node_cache[node.label] = []
                self._node_cache[node.label].append(new_ex_node)
                existing_nodes.append(new_ex_node)

        self.store.upsert_nodes(nodes)

    def upsert_relations(self, relations: list[Relation]) -> None:
        """Passes relations to LlamaIndex's store."""
        self.store.upsert_relations(relations)

    def __getattr__(self, name: str) -> Any:
        """Delegate all other method calls to the underlying store."""
        return getattr(self.store, name)
