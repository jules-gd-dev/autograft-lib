"""Tests for Cypher identifier escaping and visible failure logging."""

import logging
from typing import Any

import pytest

from autograft.db.cypher import quote_label
from autograft.integrations.base import BaseGraphMiddleware
from autograft.models.entities import Entity


def test_quote_label_wraps_and_escapes_backticks() -> None:
    """Labels are backtick-wrapped; embedded backticks are doubled."""
    assert quote_label("Company") == "`Company`"
    assert quote_label("Rock`Roll") == "`Rock``Roll`"
    assert quote_label("") == "``"


class _CapturingMiddleware(BaseGraphMiddleware):
    """Concrete middleware recording every generated query."""

    def __init__(self) -> None:
        super().__init__()
        self.queries: list[str] = []

    def _execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.queries.append(query)
        return []


class _FailingMiddleware(_CapturingMiddleware):
    """Middleware whose store is unreachable."""

    def _execute_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        self.queries.append(query)
        raise RuntimeError("db down")


def test_find_exact_candidates_escapes_untrusted_type() -> None:
    """A backtick in an extracted type must not break out of the label."""
    mw = _CapturingMiddleware()
    mw.find_exact_candidates(Entity(canonical_name="X", type="Per`son"))
    assert "MATCH (n:`Per``son`)" in mw.queries[0]


def test_ensure_vector_index_escapes_untrusted_type() -> None:
    """Vector index creation escapes both index name and label."""
    mw = _CapturingMiddleware()
    mw._ensure_vector_index("Per`son")
    assert "CREATE VECTOR INDEX `autograft_per``son_vector_index`" in mw.queries[0]
    assert "FOR (n:`Per``son`)" in mw.queries[0]


def test_candidate_query_failure_warns_and_fails_open(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unreachable store returns no candidates and logs at WARNING, not DEBUG."""
    entity = Entity(canonical_name="X", type="Company", embedding=[1.0, 0.0])
    mw = _FailingMiddleware()
    with caplog.at_level(logging.WARNING, logger="autograft.integrations.base"):
        assert mw.find_exact_candidates(entity) == []
        assert mw.find_semantic_candidates(entity) == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) >= 2
