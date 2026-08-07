"""Verification script to test homonym resolution logic and output results."""
from autograft import Entity
from autograft.core.resolver import resolve_entity
from autograft.models.entities import ExistingNode


def verify_homonyms():
    print("--- STARTING HOMONYM VERIFICATION ---")
    
    # We simulate an existing graph with some base nodes
    existing_nodes = [
        ExistingNode(node_id="Apple Inc.", canonical_name="Apple Inc.", type="Company", aliases=["Apple", "Apple Corporation"]),
        ExistingNode(node_id="Washington (State)", canonical_name="Washington (State)", type="Location", aliases=["Washington"]),
        ExistingNode(node_id="Visa Inc.", canonical_name="Visa Inc.", type="Company", aliases=["Visa"]),
        ExistingNode(node_id="Target Corporation", canonical_name="Target Corporation", type="Retailer", aliases=["Target"]),
        ExistingNode(node_id="Subway (Restaurant)", canonical_name="Subway (Restaurant)", type="Company", aliases=["Subway"]),
        ExistingNode(node_id="Python (Language)", canonical_name="Python (Language)", type="Technology", aliases=["Python"]),
        ExistingNode(node_id="Java (Language)", canonical_name="Java (Language)", type="Technology", aliases=["Java"]),
        ExistingNode(node_id="Orange S.A.", canonical_name="Orange S.A.", type="Telecom", aliases=["Orange"]),
    ]

    # These are the entities LangChain WOULD extract from our tricky documents
    extracted_entities = [
        Entity(canonical_name="Apple", type="Fruit"),
        Entity(canonical_name="Apple", type="Company"),
        Entity(canonical_name="George Washington", type="Person"),
        Entity(canonical_name="Washington", type="Location"),
        Entity(canonical_name="Visa", type="Document"),
        Entity(canonical_name="Visa", type="Company"),
        Entity(canonical_name="Target", type="Weapon"),
        Entity(canonical_name="Target", type="Retailer"),
        Entity(canonical_name="Subway", type="Transport"),
        Entity(canonical_name="Subway", type="Company"),
        Entity(canonical_name="Python", type="Animal"),
        Entity(canonical_name="Python", type="Technology"),
        Entity(canonical_name="Orange", type="Color"),
        Entity(canonical_name="Orange", type="Telecom"),
    ]

    for entity in extracted_entities:
        res = resolve_entity(entity, existing_nodes)
        decision = "MATCH (Deduplicated)" if res.is_match else "MERGE (New Node created)"
        
        print(f"\nExtracted: '{entity.canonical_name}' (Type: {entity.type})")
        print(f"-> Decision: {decision}")
        if res.is_match:
            matched_node = next((n for n in existing_nodes if n.node_id == res.matched_node_id), None)
            print(f"-> Linked to Graph Node: {matched_node.canonical_name} (Type: {matched_node.type})")
        else:
            print("-> Successfully avoided merging with homonyms!")

if __name__ == "__main__":
    verify_homonyms()
