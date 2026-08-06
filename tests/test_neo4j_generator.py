"""Unit tests for Neo4j Cypher query generator."""
from autograft.db.neo4j_generator import generate_merge_query
from autograft.models.entities import Entity, MatchResult


def test_generate_create_query() -> None:
    """Test query generation when there is no match (CREATE/MERGE)."""
    new_entity = Entity(canonical_name="Google", type="Company")
    match_result = MatchResult(is_match=False)

    query = generate_merge_query(new_entity, match_result)

    assert "MERGE" in query
    assert "n:Entity" in query


def test_generate_update_query() -> None:
    """Test query generation when a match exists (MATCH/UPDATE)."""
    new_entity = Entity(canonical_name="Apple Inc.", type="Company")
    match_result = MatchResult(
        is_match=True,
        matched_node_id="node_123",
        score=1.0,
        layer="deterministic",
    )

    query = generate_merge_query(new_entity, match_result)

    assert "MATCH" in query
    assert "node_123" in query
