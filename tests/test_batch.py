"""Unit tests for batch resolution (intra-batch dedup + graph resolution)."""

from unittest.mock import patch

from autograft.core.batch import resolve_batch
from autograft.models.entities import Entity, ExistingNode, MatchResult

APPLE_EMB = [1.0, 0.0, 0.0]
GRAPE_EMB = [0.86, 0.51, 0.0]
PEAR_EMB = [0.86, -0.51, 0.0]


def _graph() -> list[ExistingNode]:
    return [
        ExistingNode(
            node_id="n-apple",
            canonical_name="Apple Inc.",
            type="Organization",
            embedding=APPLE_EMB,
        ),
        ExistingNode(
            node_id="n-python",
            canonical_name="Python",
            type="ProgrammingLanguage",
            embedding=APPLE_EMB,
        ),
    ]


def _corpus() -> list[Entity]:
    return [
        Entity(canonical_name="Apple Inc.", type="Organization", embedding=APPLE_EMB),
        Entity(canonical_name="Apple", type="Organization", embedding=APPLE_EMB),
        Entity(
            canonical_name="Apple Incorporated",
            type="Organization",
            embedding=APPLE_EMB,
        ),
        Entity(canonical_name="Python", type="ProgrammingLanguage", embedding=APPLE_EMB),
        Entity(canonical_name="Python", type="Animal", embedding=APPLE_EMB),
    ]


def test_empty_batch() -> None:
    """Empty batch yields zeroed results and report."""
    result = resolve_batch([], db_client=[])
    assert not result.results
    assert result.report.input_count == 0
    assert result.report.cluster_count == 0
    assert result.report.total_tokens == 0


def test_single_entity_matches_graph() -> None:
    """Single entity resolving to a graph node in Pass 2."""
    entity = Entity(canonical_name="Apple Inc.", type="Organization", embedding=APPLE_EMB)
    result = resolve_batch([entity], db_client=_graph())
    assert len(result.results) == 1
    assert result.results[0].is_match is True
    assert result.results[0].matched_node_id == "n-apple"
    assert result.results[0].layer == "deterministic"
    assert result.report.cluster_count == 1
    assert result.report.intra_merges == 0


def test_identical_batch_forms_one_cluster() -> None:
    """Identical mentions collapse into one cluster."""
    entity = Entity(canonical_name="Apple", type="Organization", embedding=APPLE_EMB)
    result = resolve_batch([entity, entity, entity], db_client=[])
    assert result.report.cluster_count == 1
    assert result.report.intra_merges == 2
    assert all(r.is_match is False for r in result.results)
    assert result.report.new_nodes == 1


def test_intra_batch_dedup_before_graph() -> None:
    """Full corpus clusters intra-batch, then resolves against the graph."""
    result = resolve_batch(_corpus(), db_client=_graph())
    assert result.report.input_count == 5
    assert result.report.cluster_count == 3
    assert result.report.intra_merges == 2
    assert result.report.intra_merges_by_layer == {"lexical": 1, "semantic": 1}
    assert result.report.graph_merges_by_layer == {"deterministic": 2}
    assert result.report.new_nodes == 1
    assert result.report.total_tokens == 0
    merged_ids = [r.matched_node_id for r in result.results if r.is_match]
    assert merged_ids == ["n-apple", "n-apple", "n-apple", "n-python"]


def test_type_gate_isolates_homonyms() -> None:
    """Same name, different types stay in separate clusters."""
    lang = Entity(canonical_name="Python", type="ProgrammingLanguage", embedding=APPLE_EMB)
    animal = Entity(canonical_name="Python", type="Animal", embedding=APPLE_EMB)
    result = resolve_batch([lang, animal], db_client=_graph())
    assert result.report.cluster_count == 2
    assert result.report.intra_merges == 0
    assert result.results[0].matched_node_id == "n-python"
    assert result.results[1].is_match is False


def test_no_embedding_degrades_to_det_lexical() -> None:
    """Mentions without embeddings still merge via det/lexical."""
    a = Entity(canonical_name="Apple Inc.", type="Organization")
    b = Entity(canonical_name="Apple", type="Organization")
    result = resolve_batch([a, b], db_client=[])
    assert result.report.cluster_count == 1
    assert result.report.intra_merges == 1
    assert result.report.intra_merges_by_layer == {"lexical": 1}


def test_two_clusters_map_to_same_graph_node() -> None:
    """Two clusters can both merge into the same graph node."""
    a = Entity(canonical_name="Apple", type="Organization", embedding=GRAPE_EMB)
    b = Entity(canonical_name="Apple Computer", type="Organization", embedding=PEAR_EMB)
    graph = [
        ExistingNode(
            node_id="n-apple",
            canonical_name="Apple Inc.",
            type="Organization",
            embedding=APPLE_EMB,
        )
    ]
    result = resolve_batch([a, b], db_client=graph)
    assert result.report.cluster_count == 2
    assert result.report.intra_merges == 0
    assert result.report.graph_merges_by_layer == {"lexical": 1, "semantic": 1}
    assert {r.matched_node_id for r in result.results} == {"n-apple"}


def test_no_graph_creates_new_nodes() -> None:
    """Without a graph, every cluster becomes a new node."""
    result = resolve_batch(_corpus(), db_client=[])
    assert all(r.is_match is False for r in result.results)
    assert result.report.new_nodes == 3
    assert not result.report.graph_merges_by_layer
    assert result.report.total_tokens == 0


@patch("autograft.core.resolver.arbitrate_match")
def test_pass1_never_invokes_arbiter(mock_arbitrate) -> None:
    """Pass 1 never calls the LLM arbiter, even on uncertain pairs."""
    a = Entity(canonical_name="Alpha", type="Organization", embedding=GRAPE_EMB)
    b = Entity(canonical_name="Alpha Systems", type="Organization", embedding=PEAR_EMB)
    result = resolve_batch([a, b], db_client=[])
    assert result.report.cluster_count == 2
    assert result.report.intra_merges == 0
    mock_arbitrate.assert_not_called()


def test_deterministic_output_for_same_order() -> None:
    """Same input order yields identical output."""
    first = resolve_batch(_corpus(), db_client=_graph())
    second = resolve_batch(_corpus(), db_client=_graph())
    assert [r.model_dump() for r in first.results] == [
        r.model_dump() for r in second.results
    ]
    assert first.report == second.report


@patch("autograft.core.batch.resolve_entity")
def test_provider_overrides_flow_to_config(mock_resolve) -> None:
    """model/api_key/api_base reach the resolver config."""
    mock_resolve.return_value = MatchResult(is_match=False)
    entity = Entity(canonical_name="Apple", type="Organization")
    resolve_batch(
        [entity],
        db_client=[],
        model="m",
        api_key="k",
        api_base="b",
    )
    calls = mock_resolve.call_args_list
    assert len(calls) == 2
    for _call in calls:
        cfg = _call.kwargs["config"]
        assert cfg.model == "m"
        assert cfg.api_key == "k"
        assert cfg.api_base == "b"
