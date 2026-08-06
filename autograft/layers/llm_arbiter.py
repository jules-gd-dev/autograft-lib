"""Layer 3: LLM Arbitration for ambiguous entity resolution cases using LiteLLM."""
import os
from typing import Tuple
from dotenv import load_dotenv
import litellm
from autograft.models.entities import Entity, ExistingNode, MatchResult

load_dotenv()


def _ask_llm(
    prompt: str,
    model: str = os.getenv("AUTOGRRAFT_LLM_MODEL", "groq/llama-3.1-8b-instant"),
) -> Tuple[str, int]:
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
    model: str = os.getenv("AUTOGRRAFT_LLM_MODEL", "groq/llama-3.1-8b-instant"),
) -> MatchResult:
    """Arbitrates ambiguous match between new_entity and existing_node via LLM."""
    prompt = (
        "You are an expert Entity Resolution system. Compare the two entities below:\n\n"
        f"Entity A: Name='{new_entity.canonical_name}', Type='{new_entity.type}', "
        f"Aliases={new_entity.aliases}, Metadata={new_entity.metadata}\n"
        f"Entity B: Name='{existing_node.canonical_name}', Type='{existing_node.type}', "
        f"Aliases={existing_node.aliases}\n\n"
        "Guidelines:\n"
        "1. Standard acronyms, abbreviations, famous nick-names, or official rebrandings (e.g., 'MIT' = 'Massachusetts Institute of Technology', 'VW' = 'Volkswagen', 'Meta' = 'Facebook', 'NYC' = 'New York City', 'Warriors' = 'Golden State Warriors', 'F1' = 'Formula 1', 'Super Bowl' = 'NFL Championship') ARE THE SAME entity -> Answer OUI.\n"
        "2. Same name with DIFFERENT Entity Types (e.g., Apple as Fruit vs Apple Inc. as Company, Amazon as Location vs Amazon.com as Company, Python as Animal vs Python as Software) ARE DIFFERENT entities -> Answer NON.\n"
        "3. Distinct entities of the same type (e.g., OpenAI vs Anthropic, Emmanuel Macron vs Barack Obama, Windows 11 vs macOS, ChatGPT vs GPT-4o) ARE DIFFERENT entities -> Answer NON.\n\n"
        "Do Entity A and Entity B refer to the exact same real-world entity?\n"
        "Reply STRICTLY with one word: 'OUI' or 'NON'."
    )
    try:
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
    except Exception:
        return MatchResult(
            is_match=False,
            score=0.0,
            layer="llm_arbiter_error",
            tokens_used=0,
        )
