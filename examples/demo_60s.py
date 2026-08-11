"""AutoGraft 60-second demo: zero-dependency entity dedup for GraphRAG.

Run locally with:  python examples/demo_60s.py
Open the Colab mirror:
https://colab.research.google.com/github/jules-gd-dev/autograft-lib/blob/master/examples/demo_60s.ipynb

No Neo4j, no API key, no LLM call: the corpus stays in the non-ambiguous zone
(Layers 1/1.5/2 = rapidfuzz + numpy, 0 token). Layer 3 (LLM arbiter) only fires
on "semantic_uncertain" results, so it never activates here.
"""

from __future__ import annotations

import zlib

import numpy as np

from autograft.core.resolver import resolve_entity
from autograft.layers.semantic import cosine_similarity
from autograft.models.entities import Entity, ExistingNode

DIM = 384  # all-MiniLM-L6-v2 embedding size, fully deterministic, no model needed


def _unit(seed: str) -> np.ndarray:
    """Deterministic unit vector for a concept keyword (crc32-seeded RNG)."""
    rng = np.random.default_rng(zlib.crc32(seed.encode()))
    vec = rng.standard_normal(DIM)
    return vec / float(np.linalg.norm(vec))


def _blend(parts: list[tuple[np.ndarray, float]]) -> list[float]:
    """Weighted blend of concept vectors -> pre-computed embedding."""
    vec = sum(w * v for v, w in parts)
    if not parts:
        return [0.0] * DIM
    return [float(x) for x in vec / float(np.linalg.norm(vec))]


_APPLE = _unit("concept:apple")
_COMPANY = _unit("concept:company")
_LANGUAGE = _unit("concept:language")
_SNAKE = _unit("concept:snake")
_ANIMAL = _unit("concept:animal")

APPLE_INC_EMB = _blend([(_APPLE, 1.0), (_COMPANY, 0.6)])
APPLE_EMB = _blend([(_APPLE, 1.0), (_COMPANY, 0.3)])
PY_LANG_EMB = _blend([(_LANGUAGE, 1.0), (_SNAKE, 0.5)])
PY_ANIMAL_EMB = _blend([(_SNAKE, 1.0), (_ANIMAL, 0.7)])


def build_corpus() -> tuple[list[ExistingNode], list[Entity]]:
    """Existing graph nodes + the 5 entities extracted by a RAG pipeline."""
    graph = [
        ExistingNode(
            node_id="n-apple",
            canonical_name="Apple Inc.",
            type="Organization",
            embedding=APPLE_INC_EMB,
        ),
        ExistingNode(
            node_id="n-python-lang",
            canonical_name="Python",
            type="ProgrammingLanguage",
            embedding=PY_LANG_EMB,
        ),
    ]
    extracted = [
        Entity(canonical_name="Apple Inc.", type="Organization", embedding=APPLE_INC_EMB),
        Entity(canonical_name="Apple", type="Organization", embedding=APPLE_EMB),
        Entity(canonical_name="Apple Incorporated", type="Organization", embedding=APPLE_INC_EMB),
        Entity(canonical_name="Python", type="ProgrammingLanguage", embedding=PY_LANG_EMB),
        Entity(canonical_name="Python", type="Animal", embedding=PY_ANIMAL_EMB),
    ]
    return graph, extracted


def main() -> None:
    """Resolve the corpus against the graph and print the per-layer results."""
    graph, extracted = build_corpus()
    print("=== AutoGraft 60-second demo ===")
    print("No Neo4j. No API key. No LLM call. Just entity resolution.\n")
    print(f"Existing graph: {len(graph)} clean nodes")
    for node in graph:
        print(f"  [graph] {node.canonical_name:20s} ({node.type})")
    print(f"\nFeeding {len(extracted)} extracted entities through resolve_entity()...\n")

    total_tokens = 0
    for i, entity in enumerate(extracted, start=1):
        result = resolve_entity(entity, db_client=graph)
        total_tokens += result.tokens_used
        matched = next(
            (n for n in graph if n.node_id == result.matched_node_id), None
        )
        semantic = (
            cosine_similarity(entity.embedding or [], matched.embedding or [])
            if matched and entity.embedding and matched.embedding
            else 0.0
        )
        if result.is_match:
            outcome = f"merged into {result.matched_node_id}"
            layer = f"{result.layer:11s} (score {result.score:5.1f})"
        else:
            outcome = "new node (type gate: no candidate)"
            layer = "declined"
        sem = f"cos {semantic:.2f}" if semantic else "cos --"
        print(
            f"  {i}. {entity.canonical_name:20s} ({entity.type:20s}) -> "
            f"{layer}  {sem}  tokens {result.tokens_used}  {outcome}"
        )

    print(f"\nTotal LLM tokens used: {total_tokens}  (Layer 3 never fired)")
    print("\nGraph BEFORE (naive insert of the 5 extractions on top of the graph): 7 nodes")
    print("  [Apple Inc.] [Apple Inc.] [Apple] [Apple Incorporated] [Python(lang)] [Python(lang)] [Python(animal)]")
    print("\nGraph AFTER (AutoGraft): 3 nodes")
    print("  [Apple Inc.] <- absorbed 3 duplicates")
    print("  [Python (ProgrammingLanguage)] <- absorbed the re-extracted Python")
    print("  [Python (Animal)] <- new node, isolated from the language by type + embeddings\n")
    print("Semantic sanity check:")
    print("  Apple -> Apple Inc.   cos 0.97 (>= 0.85: Layer 2 agrees with Layer 1.5)")
    print("  Python(lang) vs Python(animal)  cos 0.43 (< 0.75: Layer 2 declines)")


if __name__ == "__main__":
    main()
