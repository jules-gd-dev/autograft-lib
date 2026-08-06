# AutoGraft Benchmark Methodology & Empirical Results

This document provides technical documentation of the evaluation methodology, dataset composition, and empirical benchmark results for AutoGraft's 3-layer hybrid Entity Resolution (ER) middleware compared against naive GraphRAG pipelines.

---

## 1. Executive Summary

AutoGraft eliminates duplicate entity node creation in Neo4j Knowledge Graphs while achieving **100% token cost reduction** on Entity Resolution tasks. Across a macro suite of **200 real-world enterprise documents** spanning 4 key industries, AutoGraft processed 742 extracted entities without invoking a single unnecessary LLM Entity Resolution API call.

*LLM Engine Infrastructure*:
- **Extraction & ER Layer**: `groq/llama-3.1-8b-instant`
- **Precision Audit Judge**: `groq/llama-3.3-70b-versatile`

| Metric | LangChain Naive (No ER) | LangChain + Full LLM ER | LangChain + AutoGraft Hybrid ER |
| :--- | :---: | :---: | :---: |
| **Evaluated Documents** | 200 documents | 200 documents | 200 documents |
| **Extracted Graph Entities** | 742 entities | 742 entities | 742 entities |
| **LLM ER API Calls** | **0 calls** | 742 calls | **0 calls** *(100% Local Short-Circuit)* |
| **Tokens Consumed** | **0 tokens** | 207,760 tokens | **0 tokens** *(100% Token Savings)* |
| **Duplicates Created** | **188 duplicates** | 0 duplicates | **0 duplicates** |
| **Duplicates Avoided (`MATCH`)** | 0 queries | **188 queries** | **188 queries** |
| **New Nodes Created (`MERGE`)** | 742 queries | **554 queries** | **554 queries** |
| **Estimated LLM API Cost** | $0.00000 | $0.04155 | **$0.00000** |
| **Knowledge Graph Quality** | Polluted with Duplicates | Deduplicated (Expensive) | **Deduplicated & Cost-Free** |

---

## 2. Evaluation Suite Architecture

### 2.1 Multi-Industry Dataset Composition
The macro benchmark suite evaluates **200 real-world enterprise documents** across 4 sectors:

1. **Legal & Compliance (50 Documents)**: M&A asset purchase agreements, Master Services Agreements (MSA), Non-Disclosure Agreements (NDA), Data Protection Addendums (DPA/GDPR), Supreme Court litigation briefs.
2. **Tech & Enterprise Software (50 Documents)**: Cloud architecture deployments (AWS/GCP), Kubernetes (K8s) container manifests, Docker specs, vector database RAG benchmarks (Elasticsearch), REST API SDK documentation.
3. **Insurance & Risk Management (50 Documents)**: Underwriting policies, Errors & Omissions (E&O), Directors & Officers liability (D&O), Commercial General Liability (CGL), Property & Casualty (P&C), Replacement Cost Value (RCV) assessments.
4. **Finance & Investment Banking (50 Documents)**: Earnings reports (EBITDA), SEC regulatory filings, KYC/AML compliance audits, SOFR/LIBOR benchmark rate transitions, LBO valuation modeling.

### 2.2 Figure References & Analytical Charts

#### Figure 1.1: Enterprise RAG Entity Resolution Performance Metrics (200 Docs / 4 Industries)
![Figure 1.1: Total Tokens Consumed, LLM Calls, Duplicates Avoided, MATCH by Sector](benchmark/assets/macro_benchmark_metrics.png)

*Detailed Metric Breakdown:*
- **Top-Left (Total Tokens Consumed)**: LangChain Naive and AutoGraft consume 0 resolution tokens, while Full LLM ER consumes 207,760 tokens.
- **Top-Right (LLM ER API Calls)**: LangChain Naive and AutoGraft make 0 API calls, while Full LLM ER makes 742 external API calls.
- **Bottom-Left (Neo4j Duplicates Avoided)**: LangChain Naive creates 188 duplicates (0 avoided), while Full LLM ER and AutoGraft resolve all 188 duplicates.
- **Bottom-Right (MATCH Queries by Industry)**: Industry breakdown of deduplication queries resolved locally across Legal (83), Tech (40), Insurance (35), and Finance (30).

#### Figure 1.2: Enterprise Knowledge Graph Cost Scaling (Up to 1,000,000 Documents)
![Figure 1.2: Linear Financial Cost Scaling Projection up to 1,000,000 Documents](benchmark/assets/macro_cost_scaling_1m.png)

#### Figure 1.3: Entity Resolution Precision by Industry Sector (100.0% Overall)
![Figure 1.3: Entity Resolution Precision Breakdown by Industry Domain (100.0%)](benchmark/assets/macro_accuracy_by_industry.png)

---

## 3. Accuracy Audit (LLM-as-a-Judge)

To ensure zero loss in entity resolution precision, an independent LLM-as-a-Judge (`groq/llama-3.3-70b-versatile`) audited 550 tricky entity pairs across 11 domains (Tech, Products, People, Homonyms, Geography, Automotive, Finance, Entertainment, Sports, Institutions, Law & Legal).

- **Overall Precision Score**: **550/550 (100.0% Precision)**
- **Audit Log File**: `benchmark/assets/macro_audit_summary.json`

---

## 4. Reproducibility & Execution

To execute the benchmark suite locally:

```bash
# 1. Run the 200-document 4-industry macro benchmark
PYTHONPATH=. python3 benchmark/run_macro_benchmark.py

# 2. Run the 550-case LLM-as-a-Judge accuracy audit
PYTHONPATH=. python3 benchmark/run_accuracy_benchmark.py

# 3. Run the legal team scenario benchmark
PYTHONPATH=. python3 benchmark/run_legal_benchmark.py
```
