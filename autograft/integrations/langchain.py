"""LangChain integration for AutoGraft."""
from typing import Any, Dict, List, Set

from autograft.core.resolver import resolve_entity
from autograft.models.entities import Entity, ExistingNode

try:
    from langchain_community.graphs import Neo4jGraph
    from langchain_community.graphs.graph_document import GraphDocument
except ImportError:
    Neo4jGraph = Any
    GraphDocument = Any


class AutoGraftNeo4jMiddleware:
    """Plug & Play LangChain Neo4jGraph wrapper for zero-cost entity resolution."""

    def __init__(self, neo4j_graph: Neo4jGraph):
        self.graph = neo4j_graph
        self._node_cache: Dict[str, List[ExistingNode]] = {}

    def _fetch_cached_nodes(self, labels: Set[str]) -> List[ExistingNode]:
        """Fetches nodes from Neo4j natively, caching them locally per label."""
        nodes = []
        for label in labels:
            if label not in self._node_cache:
                self._node_cache[label] = []
                try:
                    query = f"MATCH (n:`{label}`) RETURN n.id AS id, n.aliases AS aliases LIMIT 10000"
                    results = self.graph.query(query)
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
                except Exception:
                    pass  # If query fails, cache remains empty for this label
            nodes.extend(self._node_cache[label])
        return nodes

    def add_graph_documents(
        self, graph_documents: List[GraphDocument], **kwargs: Any
    ) -> None:
        """Intercepts, deduplicates, and passes documents to LangChain's Neo4jGraph."""
        for doc in graph_documents:
            labels = {n.type for n in doc.nodes if n.type}
            existing_nodes = self._fetch_cached_nodes(labels)

            id_mapping = {}

            # 1. Resolve Nodes
            for node in doc.nodes:
                entity = Entity(canonical_name=str(node.id), type=str(node.type))
                match_result = resolve_entity(entity, existing_nodes)

                if match_result.is_match:
                    id_mapping[node.id] = match_result.matched_node_id
                    node.id = match_result.matched_node_id
                else:
                    new_ex_node = ExistingNode(
                        node_id=str(node.id),
                        canonical_name=str(node.id),
                        type=str(node.type),
                        aliases=[str(node.id)],
                    )
                    if node.type not in self._node_cache:
                        self._node_cache[node.type] = []
                    self._node_cache[node.type].append(new_ex_node)
                    existing_nodes.append(new_ex_node)

            # 2. Remap Relationships
            for rel in doc.relationships:
                if rel.source.id in id_mapping:
                    rel.source.id = id_mapping[rel.source.id]
                if rel.target.id in id_mapping:
                    rel.target.id = id_mapping[rel.target.id]

        # 3. Pass canonicalized documents to LangChain
        self.graph.add_graph_documents(graph_documents, **kwargs)
