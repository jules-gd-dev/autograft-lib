"""LangChain integration for AutoGraft."""

from typing import TYPE_CHECKING, Any

from autograft.config import AutoGraftConfig
from autograft.core.batch import resolve_batch
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
        """Intercepts, deduplicates, and passes documents to LangChain's Neo4jGraph.

        Documents are resolved through resolve_batch so variants inside the same
        document are merged with each other before anything is written: the graph
        only sees cluster representatives, never raw per-node duplicates.
        """
        for doc in graph_documents:
            entities = []
            for node in doc.nodes:
                embedding = None
                if hasattr(node, "properties") and isinstance(node.properties, dict):
                    embedding = node.properties.get(self.config.embedding_attr)
                entities.append(
                    Entity(
                        canonical_name=str(node.id),
                        type=str(node.type),
                        embedding=embedding,
                    )
                )

            batch = resolve_batch(entities, db_client=self, config=self.config)
            id_mapping: dict[Any, Any] = {}
            new_cluster_ids: dict[str, Any] = {}
            for node, res, rep_id in zip(doc.nodes, batch.results, batch.rep_node_ids):
                original = node.id
                if res.is_match and res.matched_node_id:
                    matched_id = str(res.matched_node_id)
                    id_mapping[original] = matched_id
                    node.id = matched_id
                    self.persist_alias(matched_id, str(original))
                else:
                    # New node: collapse intra-batch variants onto their rep so
                    # only one node is created per cluster.
                    canonical = new_cluster_ids.setdefault(rep_id, original)
                    if original != canonical:
                        id_mapping[original] = canonical
                        node.id = canonical

            # 2. Remap Relationships
            for rel in doc.relationships:
                if rel.source.id in id_mapping:
                    rel.source.id = str(id_mapping[rel.source.id])
                if rel.target.id in id_mapping:
                    rel.target.id = str(id_mapping[rel.target.id])

        # 3. Pass canonicalized documents to LangChain
        self.graph.add_graph_documents(graph_documents, **kwargs)
