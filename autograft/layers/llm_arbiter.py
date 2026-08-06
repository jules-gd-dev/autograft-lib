"""Layer 3: LLM Arbitration for ambiguous entity resolution cases using LiteLLM."""
import litellm
from autograft.models.entities import Entity, ExistingNode, MatchResult


def _ask_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Calls litellm completion and returns response content string."""
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return str(response.choices[0].message.content)


def arbitrate_match(
    new_entity: Entity,
    existing_node: ExistingNode,
    model: str = "gpt-4o-mini",
) -> MatchResult:
    """Arbitrates ambiguous match between new_entity and existing_node via LLM."""
    prompt = (
        "Entity Resolution Task:\n"
        f"Entity 1: Name='{new_entity.canonical_name}', Type='{new_entity.type}', "
        f"Aliases={new_entity.aliases}, Metadata={new_entity.metadata}\n"
        f"Entity 2: Name='{existing_node.canonical_name}', Type='{existing_node.type}', "
        f"Aliases={existing_node.aliases}\n\n"
        "Do Entity 1 and Entity 2 represent the exact same entity in the real world?\n"
        "Answer STRICTLY with the word 'OUI' or 'NON'."
    )
    response_text = _ask_llm(prompt, model=model)
    if "OUI" in response_text.upper():
        return MatchResult(
            is_match=True,
            matched_node_id=existing_node.node_id,
            score=1.0,
            layer="llm_arbiter",
        )
    return MatchResult(is_match=False, layer="llm_arbiter")
