"""End-to-end demonstration script for AutoGraft middleware."""
from autograft import Entity, ExistingNode, resolve_and_generate_cypher


def main() -> None:
    """Demonstrates AutoGraft Entity Resolution and Cypher generation."""
    print("=== AutoGraft Entity Resolution Demo ===")

    # Existing Knowledge Graph nodes in Neo4j
    existing_nodes = [
        ExistingNode(
            node_id="node_1",
            canonical_name="Jean Dupont",
            type="Person",
            aliases=["J. Dupont"],
        ),
        ExistingNode(
            node_id="node_2",
            canonical_name="Microsoft",
            type="Company",
            aliases=["MSFT", "Microsoft Corp"],
        ),
    ]

    # Extracted entities from an upstream RAG pipeline / LLM extractor
    test_entities = [
        Entity(canonical_name="Microsoft", type="Company"),
        Entity(canonical_name="Google", type="Company"),
    ]

    print(f"Loaded {len(existing_nodes)} existing graph nodes.\n")

    for entity in test_entities:
        print(f"Processing Entity: '{entity.canonical_name}' ({entity.type})")
        cypher = resolve_and_generate_cypher(entity, existing_nodes)
        print("Generated Cypher Query:")
        print(f"  {cypher}\n")


if __name__ == "__main__":
    main()
