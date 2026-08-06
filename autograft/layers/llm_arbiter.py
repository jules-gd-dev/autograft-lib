"""Layer 3: LLM Arbitration for ambiguous entity resolution cases using LiteLLM."""
import os
from typing import Tuple
from dotenv import load_dotenv
import litellm
from autograft.models.entities import Entity, ExistingNode, MatchResult

load_dotenv()


def _ask_llm(
    prompt: str,
    model: str = os.getenv("AUTOGRRAFT_LLM_MODEL", "groq/llama-3.3-70b-versatile"),
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
    model: str = os.getenv("AUTOGRRAFT_LLM_MODEL", "groq/llama-3.3-70b-versatile"),
) -> MatchResult:
    """Arbitrates ambiguous match between new_entity and existing_node via LLM."""
    prompt = (
        "You are an expert Entity Resolution system.\n"
        "Determine whether Entity A and Entity B refer to the exact same real-world entity.\n\n"
        f"Entity A: Name='{new_entity.canonical_name}', Type='{new_entity.type}', Aliases={new_entity.aliases}\n"
        f"Entity B: Name='{existing_node.canonical_name}', Type='{existing_node.type}', Aliases={existing_node.aliases}\n\n"
        "Evaluation Rules:\n"
        "- Standard acronyms, abbreviations, nicknames, or official rebrandings (e.g. 'WHO' = 'World Health Organization', 'FBI' = 'Federal Bureau of Investigation', 'MIT' = 'Massachusetts Institute of Technology', 'Stanford' = 'Stanford University', 'Harvard' = 'Harvard University', 'VW' = 'Volkswagen', 'Meta' = 'Facebook Inc.', 'NYC' = 'New York City', 'Real Madrid' = 'Real Madrid C.F.', 'UN' = 'United Nations', 'Lakers' = 'Los Angeles Lakers', 'Warriors' = 'Golden State Warriors', 'F1' = 'Formula 1', 'Super Bowl' = 'NFL Championship', 'Olympic Games' = 'Olympics') represent the SAME entity -> Answer YES.\n"
        "- Different entity types (e.g. Apple Fruit vs Apple Inc Company, Amazon Location vs Amazon.com Company, Python Animal vs Python Software) represent DIFFERENT entities -> Answer NO.\n"
        "- Distinct entities or different specific products (e.g. OpenAI vs Anthropic, Emmanuel Macron vs Barack Obama, PlayStation vs Sony PS5, Windows 11 vs macOS, Manchester United vs Manchester City, UNESCO vs UNICEF) represent DIFFERENT entities -> Answer NO.\n\n"
        "Do Entity A and Entity B refer to the exact same real-world entity?\n"
        "Respond ONLY with 'YES' or 'NO'."
    )
    try:
        res = _ask_llm(prompt, model=model)
        if isinstance(res, tuple):
            response_text, tokens_used = res
        else:
            response_text, tokens_used = str(res), 0

        upper_res = response_text.strip().upper()
        is_match = "YES" in upper_res or "OUI" in upper_res or "MATCH" in upper_res

        if is_match:
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
