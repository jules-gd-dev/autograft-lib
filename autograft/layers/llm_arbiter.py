"""Layer 3: LLM Arbitration for ambiguous entity resolution cases using LiteLLM."""
import os
import time
from typing import Tuple
from dotenv import load_dotenv
import litellm
from autograft.models.entities import Entity, ExistingNode, MatchResult

load_dotenv()


def _ask_llm(
    prompt: str,
    model: str = os.getenv("AUTOGRRAFT_LLM_MODEL", "groq/llama-3.3-70b-versatile"),
) -> Tuple[str, int]:
    """Calls litellm completion with exponential backoff retry for rate limits."""
    for attempt in range(5):
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = str(response.choices[0].message.content)
            tokens = 0
            if hasattr(response, "usage") and response.usage:
                tokens = getattr(response.usage, "total_tokens", 0) or 0
            return content, tokens
        except Exception as err:
            err_str = str(err)
            if "RateLimit" in type(err).__name__ or "429" in err_str:
                time.sleep(3 * (attempt + 1))
                continue
            raise err

    raise RuntimeError("LLM request failed after 5 retry attempts.")


def arbitrate_match(
    new_entity: Entity,
    existing_node: ExistingNode,
    model: str = os.getenv("AUTOGRRAFT_LLM_MODEL", "groq/llama-3.3-70b-versatile"),
) -> MatchResult:
    """Arbitrates ambiguous match between new_entity and existing_node via LLM."""
    aliases_str = f", Aliases={new_entity.aliases}" if new_entity.aliases else ""
    node_aliases_str = f", Aliases={existing_node.aliases}" if existing_node.aliases else ""

    prompt = (
        "You are an expert Entity Resolution system.\n"
        f"Entity A: Name='{new_entity.canonical_name}', Type='{new_entity.type}'{aliases_str}\n"
        f"Entity B: Name='{existing_node.canonical_name}', Type='{existing_node.type}'{node_aliases_str}\n\n"
        "Do Entity A and Entity B refer to the exact same real-world entity?\n"
        "Rules:\n"
        "- Standard acronyms, abbreviations, nicknames, or official rebrandings (e.g. MIT, WHO, FBI, UN, Real Madrid, Lakers, F1, VW, Meta=Facebook, X=Twitter) ARE the same entity -> Answer YES.\n"
        "- Different entity types (e.g. Apple Fruit vs Apple Company) or distinct entities (e.g. OpenAI vs Anthropic) ARE NOT the same entity -> Answer NO.\n\n"
        "Reply STRICTLY with 'YES' or 'NO'."
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
