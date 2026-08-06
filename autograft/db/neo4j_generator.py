"""Neo4j Cypher query generator for creating or merging entity nodes."""

from autograft.models.entities import Entity, MatchResult


def generate_merge_query(new_entity: Entity, match_result: MatchResult) -> str:
    """Generates a Cypher query to create or update an entity node in Neo4j."""
    if not match_result.is_match:
        return (
            "MERGE (n:Entity {canonical_name: $name}) "
            "SET n.type = $type, n.aliases = $aliases RETURN n"
        )

    return (
        f"MATCH (n:Entity {{node_id: '{match_result.matched_node_id}'}}) "
        "SET n.aliases = $aliases RETURN n"
    )
