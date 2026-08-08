import os
from autograft.core.resolver import resolve_entity
from autograft.models.entities import Entity, ExistingNode
from autograft.config import AutoGraftConfig
import autograft.layers.semantic
import autograft.core.resolver
from autograft.models.entities import MatchResult

def mock_semantic_match(new_ent, candidates, match_threshold, uncertainty_threshold):
    print(f"[Layer 2] Radar Sémantique: Similarité incertaine détectée pour '{new_ent.canonical_name}' vs 'SpaceX'.")
    print("[Layer 2] -> Délégation à l'Arbiter LLM (Groq)...")
    return MatchResult(is_match=False, layer="semantic_uncertain", matched_node_id="SpaceX")

autograft.layers.semantic.find_semantic_match = mock_semantic_match
autograft.core.resolver.find_semantic_match = mock_semantic_match

class MockDBClient:
    def find_exact_candidates(self, entity):
        return []
    def find_semantic_candidates(self, entity, limit=5):
        return [ExistingNode(node_id="SpaceX", canonical_name="SpaceX", type="Company")]

config = AutoGraftConfig(
    model=os.environ.get("AUTOGRRAFT_LLM_MODEL", "groq/llama-3.3-70b-versatile"),
    api_key=os.environ.get("GROQ_API_KEY", ""),
    match_threshold=0.90,
    uncertainty_threshold=0.80
)

new_entity = Entity(canonical_name="Space Exploration Tech", type="Company")
print("Triggering AutoGraft Resolver (Layer 1 -> Layer 2 -> Layer 3)...")
result = resolve_entity(new_entity, db_client=MockDBClient(), config=config)

print("\n=== RÉSULTAT FINAL ===")
print(f"Match trouvé ? {result.is_match}")
print(f"Layer utilisé : {result.layer}")
if result.is_match:
    print(f"Fusionné avec le noeud ID : {result.matched_node_id}")
