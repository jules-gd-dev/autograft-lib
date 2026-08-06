"""Layer 3: LLM Arbitration for ambiguous entity resolution cases using LiteLLM."""
import os
from typing import Tuple, Union
from dotenv import load_dotenv
import litellm
from autograft.models.entities import Entity, ExistingNode, MatchResult

load_dotenv()


def _ask_llm(
    prompt: str,
    model: str = os.getenv("AUTOGRRAFT_LLM_MODEL", "groq/llama3-8b-8192"),
) -> Union[str, Tuple[str, int]]:
    """Calls litellm completion and returns (content_string, total_tokens)."""
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    content = str(response.choices[0].message.content)
    tokens = 0
    if hasattr(response, "usage") and response.usage:
        tokens = getattr(response.usage, "total_tokens", 0) or 0
    return content, tokens


def arbitrate_match(
    new_entity: Entity,
    existing_node: ExistingNode,
    model: str = os.getenv("AUTOGRRAFT_LLM_MODEL", "groq/llama3-8b-8192"),
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
    res = _ask_llm(prompt, model=model)
    if isinstance(res, tuple):
        response_text, tokens_used = res
    else:
        response_text, tokens_used = str(res), 0

    if "OUI" in response_text.upper():
        return MatchResult(
            is_match=True,
            matched_node_id=existing_node.node_id,
            score=1.0,
            layer="llm_arbiter",
            tokens_used=tokens_used,
        )
    return MatchResult(
        is_match=False,
        layer="llm_arbiter",
        tokens_used=tokens_used,
    )
