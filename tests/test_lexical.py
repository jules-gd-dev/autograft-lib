"""Unit tests for Layer 1.5 Lexical ER."""

from autograft.config import AutoGraftConfig
from autograft.layers.lexical import (
    _acronym_hit,
    acronym_keys,
    find_lexical_match,
    suffix_strip,
)
from autograft.models.entities import Entity, ExistingNode

# --- suffix_strip ---


def test_suffix_strip_basic() -> None:
    assert suffix_strip("Pfizer Inc") == "pfizer"
    assert suffix_strip("Zillow Group") == "zillow"
    assert suffix_strip("Toyota Motor") == "toyota"
    assert suffix_strip("Tesla Motors") == "tesla"


def test_suffix_strip_punctuation() -> None:
    assert suffix_strip("Apple Inc.") == "apple"
    assert suffix_strip("Pear Corp.") == "pear"


def test_suffix_strip_no_suffix() -> None:
    assert suffix_strip("Apple") == "apple"
    assert suffix_strip("OpenAI") == "openai"


# --- acronym_keys ---


def test_acronym_keys_multiword() -> None:
    strict, allk = acronym_keys("Department of Justice")
    assert strict == "dj"
    assert allk == "doj"


def test_acronym_keys_stopwords_excluded() -> None:
    strict, allk = acronym_keys("Securities and Exchange Commission")
    assert strict == "sec"
    assert allk == "saec"


def test_acronym_keys_single_word() -> None:
    assert acronym_keys("IBM") == ("", "")


# --- find_lexical_match ---


def test_find_lexical_suffix_match() -> None:
    entity = Entity(canonical_name="Pfizer Inc", type="Company")
    nodes = [ExistingNode(node_id="n1", canonical_name="Pfizer", type="Company")]
    result = find_lexical_match(entity, nodes)
    assert result.is_match is True
    assert result.layer == "lexical"
    assert result.matched_node_id == "n1"
    assert result.score >= 90.0


def test_find_lexical_acronym_match() -> None:
    entity = Entity(canonical_name="IBM", type="Company")
    nodes = [
        ExistingNode(
            node_id="n1",
            canonical_name="International Business Machines",
            type="Company",
        )
    ]
    result = find_lexical_match(entity, nodes)
    assert result.is_match is True
    assert result.layer == "lexical"
    assert result.score == 100.0


def test_find_lexical_alias_checked() -> None:
    entity = Entity(canonical_name="Pfizer Inc", type="Company")
    nodes = [
        ExistingNode(
            node_id="n1",
            canonical_name="Pfizer",
            type="Company",
            aliases=["PFE"],
        )
    ]
    result = find_lexical_match(entity, nodes)
    assert result.is_match is True
    assert result.matched_node_id == "n1"


def test_find_lexical_no_match() -> None:
    entity = Entity(canonical_name="Google", type="Company")
    nodes = [ExistingNode(node_id="n1", canonical_name="Apple Inc.", type="Company")]
    result = find_lexical_match(entity, nodes)
    assert result.is_match is False
    assert result.matched_node_id is None


def test_find_lexical_cross_type_blocked_v4() -> None:
    entity = Entity(canonical_name="IBM", type="Company")
    nodes = [
        ExistingNode(
            node_id="n1",
            canonical_name="International Business Machines",
            type="Organization",
        )
    ]
    result = find_lexical_match(entity, nodes)
    assert result.is_match is False


def test_find_lexical_suffix_disabled() -> None:
    entity = Entity(canonical_name="Pfizer Inc", type="Company")
    nodes = [ExistingNode(node_id="n1", canonical_name="Pfizer", type="Company")]
    cfg = AutoGraftConfig(lexical_suffix_disable=True)
    result = find_lexical_match(entity, nodes, config=cfg)
    assert result.is_match is False


def test_find_lexical_acronym_disabled() -> None:
    entity = Entity(canonical_name="IBM", type="Company")
    nodes = [
        ExistingNode(
            node_id="n1",
            canonical_name="International Business Machines",
            type="Company",
        )
    ]
    cfg = AutoGraftConfig(lexical_acronym_disable=True)
    result = find_lexical_match(entity, nodes, config=cfg)
    assert result.is_match is False


def test_find_lexical_both_disabled_falls_through() -> None:
    entity = Entity(canonical_name="Pfizer Inc", type="Company")
    nodes = [ExistingNode(node_id="n1", canonical_name="Pfizer", type="Company")]
    cfg = AutoGraftConfig(
        lexical_suffix_disable=True,
        lexical_acronym_disable=True,
    )
    result = find_lexical_match(entity, nodes, config=cfg)
    assert result.is_match is False
    assert result.layer == "deterministic"


def test_find_lexical_precision_apple_vs_apple_bank() -> None:
    entity = Entity(canonical_name="Apple", type="Company")
    nodes = [ExistingNode(node_id="n1", canonical_name="Apple Bank", type="Company")]
    result = find_lexical_match(entity, nodes)
    assert result.is_match is False


def test_acronym_hit_empty_short() -> None:
    """Empty or punctuation-only short names cannot form acronyms."""
    assert _acronym_hit("", "International Business Machines") is False
    assert _acronym_hit("!!!", "International Business Machines") is False


def test_find_lexical_acronym_match_multiword_entity() -> None:
    """Acronym match when the entity is the multi-word expansion and the node is short."""
    entity = Entity(canonical_name="International Business Machines", type="Company")
    nodes = [ExistingNode(node_id="n1", canonical_name="IBM", type="Company")]
    result = find_lexical_match(entity, nodes)
    assert result.is_match is True
    assert result.layer == "lexical"
    assert result.score == 100.0
