"""End-to-end demonstration script for AutoGraft middleware."""
import os
from dotenv import load_dotenv
from autograft import Entity, ExistingNode, resolve_and_generate_cypher
from autograft.layers.llm_arbiter import arbitrate_match

load_dotenv()


def main() -> None:
    """Demonstrates AutoGraft Entity Resolution and Cypher generation."""
    model = os.getenv("AUTOGRRAFT_LLM_MODEL", "groq/llama3-8b-8192")
    print("=== AutoGraft Entity Resolution Demo ===")
    print(f"Using model: {model}\n")

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

    # Explicit Layer 3 (LLM Arbitration) Demonstration for Uncertain Match
    print("=== Layer 3 (LLM Arbitration) Demo ===")
    uncertain_entity = Entity(canonical_name="J. Dupont", type="Person")
    target_node = existing_nodes[0]  # Jean Dupont

    print(
        f"Arbitrating match between '{uncertain_entity.canonical_name}' "
        f"and '{target_node.canonical_name}'..."
    )

    try:
        result = arbitrate_match(uncertain_entity, target_node, model=model)
        print(f"MatchResult from Layer 3: {result}")
    except Exception as err:
        print(f"LLM Arbitration API Error ({type(err).__name__}): {err}")


if __name__ == "__main__":
    main()
