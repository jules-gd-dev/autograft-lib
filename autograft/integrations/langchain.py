"""LangChain integration for AutoGraft."""

from typing import TYPE_CHECKING, Any

from autograft.config import AutoGraftConfig
from autograft.core.resolver import resolve_entity
from autograft.integrations.base import BaseGraphMiddleware
from autograft.models.entities import Entity

if TYPE_CHECKING:
    from langchain_community.graphs import Neo4jGraph
    from langchain_community.graphs.graph_document import GraphDocument
else:
    Neo4jGraph = Any
    GraphDocument = Any


class AutoGraftNeo4jMiddleware(BaseGraphMiddleware):
    """Plug & Play LangChain Neo4jGraph wrapper for zero-cost entity resolution."""

    def __init__(
        self,
        neo4j_graph: Neo4jGraph,
        config: AutoGraftConfig | None = None,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        super().__init__(config, model, api_key, api_base)
        self.graph = neo4j_graph

    def _execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return self.graph.query(query, params=params or {})

    def add_graph_documents(
        self, graph_documents: list[GraphDocument], **kwargs: Any
    ) -> None:
        """Intercepts, deduplicates, and passes documents to LangChain's Neo4jGraph."""
        for doc in graph_documents:
            id_mapping = {}

            # 1. Resolve Nodes
            for node in doc.nodes:
                # Extract embedding from properties if it exists
                embedding = None
                if hasattr(node, "properties") and isinstance(node.properties, dict):
                    embedding = node.properties.get(self.config.embedding_attr)

                entity = Entity(
                    canonical_name=str(node.id),
                    type=str(node.type),
                    embedding=embedding,
                )
                match_result = resolve_entity(
                    entity, db_client=self, config=self.config
                )

                if match_result.is_match:
                    matched_id = str(match_result.matched_node_id)
                    id_mapping[node.id] = matched_id
                    node.id = matched_id

            # 2. Remap Relationships
            for rel in doc.relationships:
                if rel.source.id in id_mapping:
                    rel.source.id = str(id_mapping[rel.source.id])
                if rel.target.id in id_mapping:
                    rel.target.id = str(id_mapping[rel.target.id])

        # 3. Pass canonicalized documents to LangChain
        self.graph.add_graph_documents(graph_documents, **kwargs)
