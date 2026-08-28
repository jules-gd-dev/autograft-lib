"""Tests for the LLM arbiter verdict parser (negative paths included)."""

from unittest.mock import patch

import pytest

from autograft.layers.llm_arbiter import _parse_verdict, arbitrate_match
from autograft.models.entities import Entity, ExistingNode


@pytest.mark.parametrize(
    "response,expected",
    [
        ("YES", True),
        ("yes.", True),
        ("OUI", True),
        ("Definitely YES, same entity.", True),
        ("NO", False),
        ("NON", False),
        ("NO MATCH", False),
        ("No, they are different entities.", False),
        ("MATCH", False),  # bare "MATCH" is ambiguous without YES/OUI: decline
        ("", False),
        ("Apple fruit vs Apple company.", False),
    ],
)
def test_parse_verdict(response: str, expected: bool) -> None:
    """Verdicts merge only on explicit YES/OUI; any NO/NON token vetoes."""
    assert _parse_verdict(response) is expected


def _pair() -> tuple[Entity, ExistingNode]:
    return (
        Entity(canonical_name="MIT", type="Organization"),
        ExistingNode(
            node_id="7",
            canonical_name="Massachusetts Institute of Technology",
            type="Organization",
        ),
    )


@patch("autograft.layers.llm_arbiter._ask_llm")
def test_no_match_response_is_not_a_merge(mock_ask_llm) -> None:
    """Regression: 'NO MATCH' contains MATCH and used to be parsed as positive."""
    mock_ask_llm.return_value = ("NO MATCH", 12)
    result = arbitrate_match(*_pair())
    assert result.is_match is False
    assert result.matched_node_id is None
    assert result.layer == "llm_arbiter"
    assert result.tokens_used == 12


@patch("autograft.layers.llm_arbiter._ask_llm")
def test_affirmative_sentence_still_merges(mock_ask_llm) -> None:
    """Free-form affirmative answers containing YES remain positive."""
    mock_ask_llm.return_value = ("YES, both refer to the same university.", 20)
    result = arbitrate_match(*_pair())
    assert result.is_match is True
    assert result.matched_node_id == "7"
