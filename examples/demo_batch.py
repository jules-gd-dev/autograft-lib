"""AutoGraft batch demo: deduplicate a full extraction batch before the graph.

Run locally with:  python examples/demo_batch.py

Same corpus and guarantees as demo_60s.py (no Neo4j, no API key, no LLM call),
but through resolve_batch(): the 5 extracted mentions are first deduplicated
among themselves (Pass 1, 0 token), then the 3 survivors resolve against the
existing graph (Pass 2). The report shows exactly what a real ingestion
pipeline needs to see.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo_60s import build_corpus

from autograft.core.batch import resolve_batch


def main() -> None:
    """Run resolve_batch on the demo corpus and print the report."""
    graph, extracted = build_corpus()
    print("=== AutoGraft batch demo ===")
    print("No Neo4j. No API key. No LLM call.\n")
    print(f"Extraction batch: {len(extracted)} mentions")

    result = resolve_batch(extracted, db_client=graph)
    report = result.report

    print("\nPer-mention final fate (Pass 1 cluster -> Pass 2 graph):")
    for i, (entity, r) in enumerate(zip(extracted, result.results), start=1):
        if r.is_match:
            print(
                f"  {i}. {entity.canonical_name:20s} -> merged into "
                f"{r.matched_node_id} ({r.layer}, score {r.score:.1f})"
            )
        else:
            print(f"  {i}. {entity.canonical_name:20s} -> new node / declined")

    print(f"\nPass 1 - intra-batch dedup (0 token):")
    print(f"  {report.input_count} mentions -> {report.cluster_count} clusters "
          f"({report.intra_merges} intra-batch merges)")
    print(f"  by layer: {report.intra_merges_by_layer}")

    print(f"\nPass 2 - against the existing graph "
          f"({len(graph)} clean nodes):")
    print(f"  graph merges by layer: {report.graph_merges_by_layer}")
    print(f"  new nodes: {report.new_nodes}")

    print(f"\nTotal LLM tokens used: {report.total_tokens}")
    print("\nBefore (naive): 7 nodes  ->  After (AutoGraft): 3 nodes")
    print("A single resolve_batch() call replaced 5 resolve_entity() calls "
          "and wrote only the survivors.")


if __name__ == "__main__":
    main()
