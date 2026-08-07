"""LangChain integration for AutoGraft."""

import logging
from typing import TYPE_CHECKING, Any

from autograft.config import AutoGraftConfig
from autograft.core.resolver import resolve_entity
from autograft.models.entities import Entity, ExistingNode

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langchain_community.graphs import Neo4jGraph
    from langchain_community.graphs.graph_document import GraphDocument
else:
    Neo4jGraph = Any
    GraphDocument = Any


class AutoGraftNeo4jMiddleware:
    """Plug & Play LangChain Neo4jGraph wrapper for zero-cost entity resolution."""

    def __init__(
        self,
        neo4j_graph: Neo4jGraph,
        config: AutoGraftConfig | None = None,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        self.graph = neo4j_graph
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
            return  # Using cache to just track if we checked the index
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
            self.graph.query(query)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to create vector index {index_name}: {e}")

    def find_exact_candidates(self, entity: Entity) -> list[ExistingNode]:
        """Queries Neo4j for nodes matching the entity type to perform deterministic match."""
        # For performance, we can query by type. To scale, we could query by alias/id.
        # But since find_exact_match handles the logic, we fetch a limited batch of likely candidates.
        # For a truly scalable exact match, we rely on the DB.
        query = f"""
        MATCH (n:`{entity.type}`)
        WHERE n.{self.config.id_attr} = $name OR $name IN n.{self.config.aliases_attr}
        RETURN n.{self.config.id_attr} AS id, n.{self.config.aliases_attr} AS aliases
        LIMIT 100
        """
        results = self.graph.query(query, params={"name": entity.canonical_name})
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
            results = self.graph.query(
                query, 
                params={
                    "index_name": index_name,
                    "limit": limit,
                    "embedding": entity.embedding
                }
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
            logger.debug(f"Vector search failed for {entity.type} (index might be missing): {e}")
            return []

    def add_graph_documents(
        self, graph_documents: list[GraphDocument], **kwargs: Any
    ) -> None:
        """Intercepts, deduplicates, and passes documents to LangChain's Neo4jGraph."""
        for doc in graph_documents:
            id_mapping = {}

            # 1. Resolve Nodes
            for node in doc.nodes:
                entity = Entity(canonical_name=str(node.id), type=str(node.type))
                match_result = resolve_entity(
                    entity, db_client=self, config=self.config
                )

                if match_result.is_match:
                    matched_id = str(match_result.matched_node_id)
                    id_mapping[node.id] = matched_id
                    node.id = matched_id
                else:
                    # New node, it will be added to the graph by LangChain
                    pass

            # 2. Remap Relationships
            for rel in doc.relationships:
                if rel.source.id in id_mapping:
                    rel.source.id = str(id_mapping[rel.source.id])
                if rel.target.id in id_mapping:
                    rel.target.id = str(id_mapping[rel.target.id])

        # 3. Pass canonicalized documents to LangChain
        self.graph.add_graph_documents(graph_documents, **kwargs)
