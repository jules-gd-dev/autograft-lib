"""Batch entity resolution: intra-batch dedup (0-token) then graph resolution."""

from autograft.config import AutoGraftConfig
from autograft.core.resolver import resolve_entity
from autograft.db.client import GraphDatabaseClient
from autograft.models.batch import BatchResult, ResolutionReport
from autograft.models.entities import Entity, ExistingNode, MatchResult


def _merge_layer_count(counts: dict[str, int], layer: str) -> dict[str, int]:
    counts[layer] = counts.get(layer, 0) + 1
    return counts


def _cluster_rep(node_id: str, entity: Entity) -> ExistingNode:
    return ExistingNode(
        node_id=node_id,
        canonical_name=entity.canonical_name,
        type=entity.type,
        aliases=list(entity.aliases),
        embedding=entity.embedding,
    )


def resolve_batch(
    entities: list[Entity],
    db_client: GraphDatabaseClient | list[ExistingNode],
    config: AutoGraftConfig | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> BatchResult:
    """Deduplicates a batch of extracted mentions against each other, then the graph.

    Pass 1 clusters mentions using only the 0-token layers (deterministic, lexical,
    semantic). Pass 2 resolves each cluster representative against `db_client` with
    the full resolver (LLM arbiter enabled). Every member of a cluster inherits its
    representative's graph result.
    """
    cfg = config or AutoGraftConfig()
    if model:
        cfg.model = model
    if api_key:
        cfg.api_key = api_key
    if api_base:
        cfg.api_base = api_base

    clusters: list[ExistingNode] = []
    members: dict[str, list[Entity]] = {}
    cluster_of: list[str] = []
    intra_merges_by_layer: dict[str, int] = {}

    for entity in entities:
        result = resolve_entity(entity, clusters, config=cfg, enable_arbiter=False)
        if result.is_match and result.matched_node_id is not None:
            node_id = result.matched_node_id
            rep = next(n for n in clusters if n.node_id == node_id)
            if entity.canonical_name not in rep.aliases:
                rep.aliases.append(entity.canonical_name)
            members[node_id].append(entity)
            cluster_of.append(node_id)
            intra_merges_by_layer = _merge_layer_count(
                intra_merges_by_layer, result.layer
            )
        else:
            node_id = f"batch-{len(clusters)}"
            clusters.append(_cluster_rep(node_id, entity))
            members[node_id] = [entity]
            cluster_of.append(node_id)

    graph_result_by_cluster: dict[str, MatchResult] = {}
    graph_merges_by_layer: dict[str, int] = {}
    new_nodes = 0
    total_tokens = 0

    for rep in clusters:
        graph_result = resolve_entity(
            Entity(
                canonical_name=rep.canonical_name,
                type=rep.type,
                aliases=list(rep.aliases),
                embedding=rep.embedding,
            ),
            db_client,
            config=cfg,
        )
        graph_result_by_cluster[rep.node_id] = graph_result
        if graph_result.is_match:
            graph_merges_by_layer = _merge_layer_count(
                graph_merges_by_layer, graph_result.layer
            )
        else:
            new_nodes += 1
        total_tokens += graph_result.tokens_used

    results = [graph_result_by_cluster[cid] for cid in cluster_of]

    report = ResolutionReport(
        input_count=len(entities),
        cluster_count=len(clusters),
        intra_merges=len(entities) - len(clusters),
        intra_merges_by_layer=intra_merges_by_layer,
        graph_merges_by_layer=graph_merges_by_layer,
        new_nodes=new_nodes,
        total_tokens=total_tokens,
    )
    return BatchResult(results=results, report=report, rep_node_ids=list(cluster_of))
