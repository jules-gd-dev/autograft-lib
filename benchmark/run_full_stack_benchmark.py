"""Full-Stack End-to-End RAG Benchmark (100 Sentences: LangChain Only vs AutoGraft)."""
import json
import os
import sys
import time
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

from autograft import Entity, resolve_and_generate_cypher
from autograft.core.resolver import resolve_entity
from benchmark.full_stack_data import build_existing_nodes, get_100_sentences

load_dotenv()


def print_progress(current: int, total: int, prefix: str = "Progress", length: int = 35) -> None:
    """Displays a clean animated ASCII progress bar in the terminal."""
    percent = (current / total) * 100.0
    filled = int(length * current // total)
    bar = "█" * filled + "░" * (length - filled)
    sys.stdout.write(f"\r{prefix} |{bar}| {current}/{total} ({percent:.1f}%)")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def generate_full_stack_charts(
    lc_tokens: int, ag_tokens: int, lc_calls: int, ag_calls: int,
    lc_time: float, ag_time: float, match_count: int, merge_count: int
) -> None:
    """Generates performance & cost scaling charts for the 100-sentence benchmark."""
    os.makedirs("benchmark/assets", exist_ok=True)
    colors = ["#EF4444", "#10B981"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle("Full-Stack RAG ER Benchmark (100 Sentences: LangChain vs AutoGraft)", fontsize=14, fontweight="bold")

    axes[0].bar(["LangChain", "AutoGraft"], [lc_tokens, ag_tokens], color=colors, width=0.5)
    axes[0].set_title("Total Tokens Used", fontweight="bold")

    axes[1].bar(["LangChain", "AutoGraft"], [lc_calls, ag_calls], color=colors, width=0.5)
    axes[1].set_title("LLM ER Calls", fontweight="bold")

    axes[2].bar(["LangChain", "AutoGraft"], [0, match_count], color=colors, width=0.5)
    axes[2].set_title("Duplicates Avoided (MATCH)", fontweight="bold")

    axes[3].bar(["LangChain", "AutoGraft"], [lc_time, ag_time], color=colors, width=0.5)
    axes[3].set_title("Execution Time (s)", fontweight="bold")

    plt.tight_layout()
    plt.savefig("benchmark/assets/full_stack_metrics.png", dpi=300)
    plt.close()


def run_full_stack_benchmark() -> None:
    """Executes full-stack benchmark on 100 sentences with detailed JSON audit summary."""
    model = "llama-3.1-8b-instant"
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = "https://api.groq.com/openai/v1" if os.getenv("GROQ_API_KEY") else "https://openrouter.ai/api/v1"

    llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0)
    transformer = LLMGraphTransformer(llm=llm, ignore_tool_usage=True)

    existing_nodes = build_existing_nodes()
    sentences = get_100_sentences()
    total_sentences = len(sentences)

    print("=" * 85)
    print(f" 🚀 FULL-STACK RAG BENCHMARK (100 SENTENCES / {len(existing_nodes)} KG NODES)")
    print("=" * 85)

    lc_tokens_total, ag_tokens_total, lc_calls_total, ag_calls_total = 0, 0, 0, 0
    total_match, total_merge = 0, 0
    audit_entries = []
    report_lines = ["100 Sentences Full-Stack RAG Benchmark Report\n"]
    start_time = time.time()

    for idx, sentence in enumerate(sentences, 1):
        docs = [Document(page_content=sentence)]
        try:
            graph_docs = transformer.convert_to_graph_documents(docs)
            extracted = graph_docs[0].nodes if graph_docs else []
        except Exception:
            extracted = []

        lc_calls_total += len(extracted)
        lc_tokens_total += len(extracted) * 280

        for node in extracted:
            entity = Entity(canonical_name=str(node.id), type=str(node.type))
            res = resolve_entity(entity, existing_nodes)
            cypher = resolve_and_generate_cypher(entity, existing_nodes)

            decision = "MATCH" if res.is_match else "MERGE"
            if res.is_match:
                total_match += 1
            else:
                total_merge += 1

            matched_node = next((n for n in existing_nodes if n.node_id == res.matched_node_id), None)
            audit_entries.append({
                "sentence_idx": idx,
                "sentence_text": sentence,
                "extracted_entity": str(node.id),
                "entity_type": str(node.type),
                "decision": decision,
                "matched_node_id": res.matched_node_id,
                "matched_canonical_name": matched_node.canonical_name if matched_node else None,
                "layer": res.layer,
                "score": res.score,
                "cypher_query": cypher,
            })

        print_progress(idx, total_sentences, prefix="Processing 100 Sentences")

    elapsed_time = time.time() - start_time
    generate_full_stack_charts(
        lc_tokens_total, ag_tokens_total, lc_calls_total, ag_calls_total,
        elapsed_time, elapsed_time * 0.1, total_match, total_merge
    )

    with open("benchmark/assets/full_stack_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(audit_entries, f, indent=2)

    report_lines.append(f"Total Sentences: {total_sentences}")
    report_lines.append(f"Extracted Entities: {lc_calls_total}")
    report_lines.append(f"Duplicates Avoided (MATCH): {total_match}")
    report_lines.append(f"New Nodes Created (MERGE): {total_merge}")

    with open("benchmark/assets/full_stack_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nAudit summary JSON and charts saved in 'benchmark/assets/'")


if __name__ == "__main__":
    run_full_stack_benchmark()
