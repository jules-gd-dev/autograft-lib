"""Layer 3: LLM Arbitration for ambiguous entity resolution cases using LiteLLM."""

import re
import time

import litellm
from dotenv import load_dotenv

from autograft.config import AutoGraftConfig
from autograft.models.entities import Entity, ExistingNode, MatchResult

load_dotenv()

# A NO/NON token vetoes the merge first, so free-form answers like "NO MATCH"
# can never be read as positive. Anything without an explicit YES/OUI declines.
_NEGATIVE_TOKEN = re.compile(r"\b(?:NO|NON)\b")
_POSITIVE_TOKEN = re.compile(r"\b(?:YES|OUI)\b")


def _parse_verdict(response_text: str) -> bool:
    """Merge only on an explicit YES/OUI token; any NO/NON token vetoes."""
    upper = response_text.strip().upper()
    if _NEGATIVE_TOKEN.search(upper):
        return False
    return bool(_POSITIVE_TOKEN.search(upper))


def _ask_llm(
    prompt: str,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    config: AutoGraftConfig | None = None,
) -> tuple[str, int]:
    """Calls litellm completion with exponential backoff retry for rate limits."""
    cfg = config or AutoGraftConfig()
    target_model = model or cfg.model
    target_api_key = api_key or cfg.api_key
    target_api_base = api_base or cfg.api_base

    kwargs: dict = {}
    if target_api_key:
        kwargs["api_key"] = target_api_key
    if target_api_base:
        kwargs["api_base"] = target_api_base

    for attempt in range(5):
        try:
            response = litellm.completion(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
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
            raise

    raise RuntimeError("LLM request failed after 5 retry attempts.")


def arbitrate_match(
    new_entity: Entity,
    existing_node: ExistingNode,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    config: AutoGraftConfig | None = None,
) -> MatchResult:
    """Arbitrates ambiguous match between new_entity and existing_node via LLM."""
    aliases_str = f", Aliases={new_entity.aliases}" if new_entity.aliases else ""
    node_aliases_str = (
        f", Aliases={existing_node.aliases}" if existing_node.aliases else ""
    )

    prompt = (
        "You are an expert Entity Resolution system.\n"
        f"Entity A: Name='{new_entity.canonical_name}', Type='{new_entity.type}'{aliases_str}\n"
        f"Entity B: Name='{existing_node.canonical_name}', Type='{existing_node.type}'{node_aliases_str}\n\n"
        "Do Entity A and Entity B refer to the exact same real-world entity?\n"
        "Rules:\n"
        "- Short names, famous acronyms, abbreviations, nicknames, or full official names (e.g. MIT, WHO, FBI, UN, Real Madrid, Lakers, F1, VW, HBO=Home Box Office, Palantir=Palantir Technologies, Stanford=Stanford University, ICJ=International Court of Justice, Meta=Facebook, X=Twitter) ARE the same entity -> Answer YES.\n"
        "- Different entity types (e.g. Apple Fruit vs Apple Company) or distinct entities (e.g. OpenAI vs Anthropic) ARE NOT the same entity -> Answer NO.\n\n"
        "Reply STRICTLY with 'YES' or 'NO'."
    )
    try:
        res = _ask_llm(
            prompt,
            model=model,
            api_key=api_key,
            api_base=api_base,
            config=config,
        )
        if isinstance(res, tuple):
            response_text, tokens_used = res
        else:
            response_text, tokens_used = str(res), 0

        is_match = _parse_verdict(response_text)

        if is_match:
            return MatchResult(
                is_match=True,
                matched_node_id=existing_node.node_id,
                score=1.0,
                layer="llm_arbiter",
                tokens_used=tokens_used,
            )

        import logging

        logger = logging.getLogger("autograft.resolver")
        logger.info(
            f"LLM Arbiter explicitly declined merge between '{new_entity.canonical_name}' and '{existing_node.canonical_name}'"
        )

        return MatchResult(
            is_match=False,
            layer="llm_arbiter",
            tokens_used=tokens_used,
        )
    except Exception as e:  # noqa: BLE001
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(e)
        return MatchResult(
            is_match=False,
            score=0.0,
            layer="llm_arbiter_error",
            tokens_used=0,
        )
