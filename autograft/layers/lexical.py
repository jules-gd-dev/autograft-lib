"""Layer 1.5: Lexical Entity Resolution — suffix-strip + acronym, 0 token (V1)."""

from rapidfuzz import fuzz

from autograft.config import AutoGraftConfig
from autograft.models.entities import Entity, ExistingNode, MatchResult

SUFFIX_THRESHOLD = 90.0
ACRONYM_MAX_LEN = 6

# Corporate designators stripped before core comparison. "Bank" excluded to keep
# "Apple" vs "Apple Bank" distinct (V1 risk mitigation).
SUFFIXES = {
    "inc",
    "corp",
    "corporation",
    "co",
    "ltd",
    "llc",
    "plc",
    "group",
    "lp",
    "sa",
    "holdings",
    "motor",
    "motors",
    "university",
    "institute",
    "wholesale",
    "company",
    "limited",
}

# Fixed stopword set for the acronym `strict` key (V5).
STOPWORDS = {"of", "the", "and", "&", "for", "de", "la"}


def _tokens(name: str) -> list[str]:
    """Lowercased alnum tokens; punctuation/spaces dropped, "&" absorbed out."""
    return [
        "".join(c for c in tok if c.isalnum())
        for tok in name.lower().split()
        if any(c.isalnum() for c in tok)
    ]


def suffix_strip(name: str) -> str:
    """Strip trailing corporate designators, return lowercased core (signal 1)."""
    toks = _tokens(name)
    while toks and toks[-1] in SUFFIXES:
        toks.pop()
    return " ".join(toks)


def acronym_keys(name: str) -> tuple[str, str]:
    """Return (strict, all) lowercased initialism keys for a multi-word name (V5).

    strict = initials of non-stopword tokens; all = initials of every token.
    """
    toks = _tokens(name)
    if len(toks) < 2:
        return "", ""
    strict = "".join(t[0] for t in toks if t not in STOPWORDS)
    allk = "".join(t[0] for t in toks)
    return strict, allk


def _normalize(name: str) -> str:
    return "".join(c for c in name if c.isalnum()).lower()


def _acronym_hit(short: str, long_name: str) -> bool:
    """True if `short` equals the strict/all initials of multi-word `long_name`."""
    s = _normalize(short)
    if not s:
        return False
    strict, allk = acronym_keys(long_name)
    return s == strict or s == allk


def _pair_lexical(
    a: str, b: str, suffix_on: bool, acronym_on: bool
) -> tuple[bool, float]:
    """Return (matched, score) for one name pair via acronym then suffix-strip."""
    if acronym_on:
        a_multi, b_multi = len(_tokens(a)) >= 2, len(_tokens(b)) >= 2
        a_short, b_short = (
            len(_normalize(a)) <= ACRONYM_MAX_LEN,
            len(_normalize(b)) <= ACRONYM_MAX_LEN,
        )
        if a_multi and b_short and _acronym_hit(b, a):
            return True, 100.0
        if b_multi and a_short and _acronym_hit(a, b):
            return True, 100.0
    if suffix_on:
        core_a, core_b = suffix_strip(a), suffix_strip(b)
        if core_a and core_b:
            score = float(fuzz.token_sort_ratio(core_a, core_b))
            if score >= SUFFIX_THRESHOLD:
                return True, score
    return False, 0.0


def find_lexical_match(
    new_entity: Entity,
    existing_nodes: list[ExistingNode],
    config: AutoGraftConfig | None = None,
) -> MatchResult:
    """Type-gated lexical match: suffix-strip core (fuzz>=90) or exact acronym.

    Type gate (V4) is enforced here so homonym isolation holds regardless of how
    candidates are sourced. 0 token, 0 embedding required.
    """
    cfg = config or AutoGraftConfig()
    suffix_on = not cfg.lexical_suffix_disable
    acronym_on = not cfg.lexical_acronym_disable
    if not (suffix_on or acronym_on):
        return MatchResult(is_match=False)

    new_name = new_entity.canonical_name
    best_score = 0.0
    best_node_id: str | None = None

    for existing in existing_nodes:
        if existing.type.lower() != new_entity.type.lower():  # V4 type gate
            continue
        for cand in [existing.canonical_name, *existing.aliases]:
            matched, score = _pair_lexical(new_name, cand, suffix_on, acronym_on)
            if matched and score > best_score:
                best_score, best_node_id = score, existing.node_id

    if best_node_id is not None:
        return MatchResult(
            is_match=True,
            matched_node_id=best_node_id,
            score=best_score,
            layer="lexical",
        )
    return MatchResult(is_match=False)
