"""Public API entrypoint for AutoGraft entity resolution middleware."""

from autograft.core.resolver import resolve_entity
from autograft.db.neo4j_generator import generate_merge_query
from autograft.models.entities import Entity, ExistingNode


def resolve_and_generate_cypher(
    new_entity: Entity, existing_nodes: list[ExistingNode]
) -> str:
    """Resolves new_entity against existing_nodes and generates Neo4j Cypher query."""
    match_result = resolve_entity(new_entity, existing_nodes)
    return generate_merge_query(new_entity, match_result)
