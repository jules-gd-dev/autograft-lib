"""Deterministic generator for the 500-document real benchmark corpus."""

import random

from benchmark.data.catalog import (
    BenchmarkDoc,
    EntityMention,
    HOMONYMS,
    INDUSTRIES,
    VARIANT_MAP,
)

NUM_DOCS = 500
ENTITIES_PER_DOC = 6
HOMONYM_INJECT_RATE = 0.08
SEED = 42


def _pick_variant(canon_id: str, rng: random.Random) -> str:
    """Pick a surface name variant; canonical name weighted heavier (realistic)."""
    _, _, variants = VARIANT_MAP[canon_id]
    weights = [3.0] + [1.0] * (len(variants) - 1)
    return rng.choices(variants, weights=weights, k=1)[0]


def _entity_type(canon_id: str) -> str:
    return VARIANT_MAP[canon_id][1]


def generate_corpus(
    num_docs: int = NUM_DOCS, seed: int = SEED
) -> list[BenchmarkDoc]:
    """Build a reproducible list of documents with ground-truth mentions."""
    rng = random.Random(seed)
    industry_list = list(INDUSTRIES.keys())
    docs: list[BenchmarkDoc] = []

    for i in range(num_docs):
        industry = industry_list[i % len(industry_list)]
        pool = INDUSTRIES[industry]
        chosen = rng.choices(pool, k=ENTITIES_PER_DOC)

        # Occasionally swap one slot for a cross-industry homonym trap.
        if rng.random() < HOMONYM_INJECT_RATE:
            chosen[rng.randrange(len(chosen))] = rng.choice(HOMONYMS)

        mentions: list[EntityMention] = []
        names: list[str] = []
        for canon_id in chosen:
            name = _pick_variant(canon_id, rng)
            mentions.append(
                EntityMention(name=name, type=_entity_type(canon_id), canonical_id=canon_id)
            )
            names.append(name)

        text = (
            f"{industry} briefing #{i}: covers "
            + ", ".join(sorted(set(names)))
            + "."
        )
        docs.append(
            BenchmarkDoc(doc_id=f"doc_{i:04d}", industry=industry, text=text, entities=mentions)
        )
    return docs


def ground_truth_canonical_ids(docs: list[BenchmarkDoc]) -> set[tuple[str, str]]:
    """Return the set of (canonical_id, type) unique identities present."""
    return {(m.canonical_id, m.type) for d in docs for m in d.entities}
