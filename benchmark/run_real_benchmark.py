"""Real AutoGraft benchmark: streams 500 docs through resolve_entity.

Fully measured run (no scaling factors, no placeholder vectors). Uses real sentence-transformers
embeddings, real Groq LLM arbiter calls, real litellm pricing, and measures
accuracy against the ground-truth corpus.
"""

import json
import os
import time

from dotenv import load_dotenv
from tqdm import tqdm

from autograft.config import AutoGraftConfig
from autograft.core.resolver import resolve_entity
from autograft.models.entities import Entity, ExistingNode
from benchmark.data.real_corpus import generate_corpus
from benchmark.utils.harness import (
    ARBITER_MODEL,
    assemble_results,
    build_embeddings,
    litellm_callback,
)
from benchmark.utils.metrics import MentionResult, ResolutionReport

load_dotenv()
RATE_SLEEP = 2.2  # stay under Groq free-tier rate limits


def run() -> dict:
    """Stream the 500-doc corpus through the real 3-layer ER pipeline."""
    import litellm

    litellm.success_callback = [litellm_callback]
    docs = generate_corpus()
    emb_map = build_embeddings(docs)

    config = AutoGraftConfig(
        model=ARBITER_MODEL,
        api_key=os.environ["GROQ_API_KEY"],
        match_threshold=0.85,
        uncertainty_threshold=0.75,
    )

    db_nodes: list[ExistingNode] = []
    node_by_id: dict[str, ExistingNode] = {}
    node_canon: dict[str, str] = {}
    report = ResolutionReport(node_canon=node_canon)
    counter = 0

    for doc in tqdm(docs, desc="Streaming 500 docs"):
        for m in doc.entities:
            entity = Entity(
                canonical_name=m.name, type=m.type, embedding=emb_map[m.name]
            )
            t0 = time.perf_counter()
            res = resolve_entity(entity, db_client=db_nodes, config=config)
            dt = (time.perf_counter() - t0) * 1000.0

            if res.is_match:
                mid = res.matched_node_id
                mcanon = node_canon.get(res.matched_node_id)
                # V2: accumulate incoming name as alias on the matched node,
                # mirroring production MERGE so later repeats hit L1 for free.
                node = node_by_id.get(res.matched_node_id)
                if node is not None and res.new_alias and res.new_alias not in node.aliases:
                    node.aliases.append(res.new_alias)
            else:
                nid = f"n{counter}"
                counter += 1
                node = ExistingNode(
                    node_id=nid, canonical_name=m.name, type=m.type,
                    embedding=emb_map[m.name],
                )
                db_nodes.append(node)
                node_by_id[nid] = node
                node_canon[nid] = m.canonical_id
                mid, mcanon = None, None

            report.add(
                MentionResult(
                    m.name, m.type, m.canonical_id, mid, mcanon,
                    res.layer, res.is_match, res.tokens_used, dt,
                )
            )
            if res.layer.startswith("llm"):
                time.sleep(RATE_SLEEP)

    return assemble_results(report)


if __name__ == "__main__":
    out = run()
    os.makedirs("benchmark/assets", exist_ok=True)
    path = "benchmark/assets/real_benchmark_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nSaved -> {path}")
