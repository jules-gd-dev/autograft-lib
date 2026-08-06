"""Unit tests for AutoGraft Public API."""
from autograft.api.main import resolve_and_generate_cypher
from autograft.models.entities import Entity, ExistingNode


def test_api_resolve_and_generate_cypher() -> None:
    """Test public API resolves exact match and produces update Cypher query."""
    existing_node = ExistingNode(
        node_id="node_apple",
        canonical_name="Apple Inc.",
        type="Company",
        aliases=["Apple"],
    )
    new_entity = Entity(canonical_name="Apple Inc.", type="Company")

    query = resolve_and_generate_cypher(new_entity, [existing_node])

    assert "MATCH" in query
    assert "node_apple" in query
