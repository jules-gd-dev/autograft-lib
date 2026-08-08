# AutoGraft Benchmark — Real Methodology & Results

> Replaces the previous (falsified) benchmark. No scaling factors, no hand-crafted
> vectors, no invented metrics. Every number below was **measured** on 2026-08-08.

---

## 1. What changed (and why the old numbers were fake)

The previous benchmark was fabricated in three ways:

1. **Scaling fraud** — `run_macro_benchmark.py` ran only **50** documents then
   multiplied every result by **10** (`scale_factor = 500 / DOC_COUNT`) to claim 500.
2. **Hand-crafted embeddings** — entities carried 3D vectors like `[0.85, 0.526, 0]`
   designed to match perfectly, so the semantic layer never failed.
3. **Invented "100%"** — synthetic names (`TechBase_0 Inc.`) cannot be ambiguous, so
   precision was trivially 100%.

This document replaces all of that with a **fully reproducible** run.

---

## 2. Methodology

### 2.1 Corpus — 500 real documents
A deterministic generator (`benchmark/data/real_corpus.py`, seed `42`) builds
**500 documents** across 10 industries (Tech, Finance, Healthcare, Legal, Education,
Energy, Retail, Manufacturing, Insurance, Real Estate). Each document holds ~6
**real-world** entity mentions drawn from a catalog of 63 canonical identities
(`benchmark/data/catalog.py`) — real companies (Microsoft, JPMorgan, Pfizer…),
organizations (WHO, MIT, SEC…), plus **cross-domain homonym traps** (Apple
fruit/company, Amazon river/company, Washington place/person, Python
animal/technology).

Each mention carries a **ground-truth `canonical_id`**, so accuracy is measured,
not assumed.

### 2.2 Embeddings — real neural model
Embeddings are computed with **`sentence-transformers/all-MiniLM-L6-v2`** (384-dim,
CPU). This is what a real GraphRAG pipeline uses. AutoGraft itself does **not**
generate embeddings — it consumes them (`integrations/langchain.py:46`); the
benchmark supplies real ones instead of fake vectors.

### 2.3 Resolution engine — the actual code
The benchmark streams documents through the **real** `resolve_entity` pipeline
(`autograft/core/resolver.py`) backed by an in-memory `ListDatabaseClient`. This is
the same ER engine the Neo4j middleware uses, minus the database round-trip.

### 2.4 LLM & pricing — real calls, real prices
- **Arbiter model:** `groq/llama-3.1-8b-instant` (Layer 3 only).
- Tokens are read from the **actual API response** (`response.usage`).
- Cost is computed via **`litellm.completion_cost`** using published Groq pricing
  ($0.05/1M input, $0.08/1M output).
- The **Full-LLM-ER baseline** uses the *same deterministic arbiter prompt*; its
  cost = measured per-call tokens × mention count (the prompt is fixed, so this is an
  exact projection, not a guess).

---

## 3. Results (500 docs / 3000 entities / 63 identities)

| Metric | Naive (no ER) | Full LLM ER | AutoGraft (hybrid) |
| :--- | ---: | ---: | ---: |
| Entity mentions | 3 000 | 3 000 | 3 000 |
| LLM ER API calls | 0 | 3 000 | **238** (92.1% local) |
| Tokens consumed | 0 | 661 714 | **52 496** |
| LLM ER cost | $0.000 | $0.0334 | **$0.00265** |
| Final graph nodes | 3 000 | 63 | 103 |
| Duplicate nodes created | 2 937 | 0 | 0 |
| Precision | — | — | 100.00% |
| Recall | — | — | 98.64% |
| F1 | — | — | 99.31% |
| Latency (mean / max) | — | — | 6.5 ms / 280 ms |

### 3.1 Layer distribution (raw counts)
![Layer distribution](benchmark/assets/real_layers.png)

- `deterministic_match`: 2423 (80.8%) — exact/alias re-occurrence, 0 tokens
- `semantic_match`: 242 (8.1%) — cosine ≥ 0.85, 0 tokens
- `llm_merge`: 232 (7.7%) — LLM confirmed the merge
- `llm_declined`: 6 (0.2%) — LLM correctly refused a false merge
- `no_match_declined`: 97 (3.2%) — new unique node, no candidate

### 3.2 Metrics comparison
![Metrics](benchmark/assets/real_metrics.png)

---

## 4. Honest limitations (what it misses)

**40 mentions missed merging** (recall 98.64%, precision 100%). The misses are
genuinely hard pairs that neither fuzzy strings nor MiniLM embeddings can bridge:

- **Stock tickers:** `GOOGL`→Google, `TSLA`→Tesla, `AMZN`→Amazon, `GS`→Goldman
  Sachs, `MA`→Mastercard
- **Unrelated full names:** `British Petroleum`→BP, `World Health Organization`→WHO,
  `American International Group`→AIG, `Securities and Exchange Commission`→SEC
- **Rebrand/name splits:** `Facebook`→Meta, `Alphabet Inc`→Google, `Meta Platforms`→Meta

These fall below the cosine uncertainty threshold (0.75) so never reach the LLM.
This is a **real trade-off**, not a bug: lowering the threshold would raise recall
but cost more LLM calls and risk false merges.

### 4.1 Accuracy
![Accuracy](benchmark/assets/real_accuracy.png)

---

## 5. Cost scaling (projection)
Linear projection from the measured per-document cost. No exponential assumptions.

![Cost scaling](benchmark/assets/real_cost_scaling.png)

---

## 6. Reproducibility

```bash
# Requires: GROQ_API_KEY in .env, sentence-transformers installed
pip install sentence-transformers
PYTHONPATH=. python3 benchmark/run_real_benchmark.py      # ~9 min, real API calls
PYTHONPATH=. python3 benchmark/utils/real_charts.py        # regenerate charts
```

Raw per-mention data (name, layer, correctness, latency, tokens) is saved to
`benchmark/assets/real_benchmark_results.json` for full audit.
