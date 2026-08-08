"""Raw, unembellished charts from real benchmark results."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_PATH = "benchmark/assets/real_benchmark_results.json"
SCALES = [500, 1_000, 10_000, 100_000, 1_000_000]


def _load() -> dict:
    with open(RESULTS_PATH) as f:
        return json.load(f)


def _bar_value(ax, bars, fmt="${:.4f}"):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")


def chart_metrics(r: dict) -> None:
    labels = ["Naive\n(no ER)", "Full LLM ER", "AutoGraft\n(hybrid)"]
    tokens = [0, r["full_llm_er"]["total_tokens"], r["autograft"]["total_tokens"]]
    calls = [0, r["full_llm_er"]["llm_calls"], r["autograft"]["llm_calls"]]
    dups = [r["naive"]["duplicates"], 0, 0]
    cost = [0.0, r["full_llm_er"]["cost_usd"], r["autograft"]["cost_usd"]]

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    cols = ["#d62728", "#ff7f0e", "#2ca02c"]
    b1 = axs[0, 0].bar(labels, tokens, color=cols); axs[0, 0].set_title("Tokens consumed (ER layer)")
    _bar_value(axs[0, 0], b1, "{:.0f}")
    b2 = axs[0, 1].bar(labels, calls, color=cols); axs[0, 1].set_title("LLM API calls")
    _bar_value(axs[0, 1], b2, "{:.0f}")
    b3 = axs[1, 0].bar(labels, dups, color=cols); axs[1, 0].set_title("Duplicate nodes created")
    _bar_value(axs[1, 0], b3, "{:.0f}")
    b4 = axs[1, 1].bar(labels, cost, color=cols); axs[1, 1].set_title("LLM ER cost (USD)")
    _bar_value(axs[1, 1], b4, "${:.5f}")
    fig.suptitle("AutoGraft Benchmark - Raw Metrics (500 real docs)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("benchmark/assets/real_metrics.png", dpi=150)
    plt.close(fig)


def chart_layers(r: dict) -> None:
    lb = r["autograft"]["layer_breakdown"]
    order = ["deterministic_match", "semantic_match", "llm_merge",
             "llm_declined", "no_match_declined", "llm_error"]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#ffbb78", "#7f7f7f", "#d62728"]
    keys = [k for k in order if k in lb] + [k for k in lb if k not in order]
    vals = [lb[k] for k in keys]
    total = r["total_mentions"]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(keys, vals, color=colors[: len(keys)])
    for b, v in zip(bars, vals):
        ax.annotate(f"{v}\n({v/total*100:.1f}%)", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Entity mentions resolved"); ax.set_title("Resolution layer distribution (raw counts)")
    plt.xticks(rotation=15)
    fig.tight_layout(); fig.savefig("benchmark/assets/real_layers.png", dpi=150); plt.close(fig)


def chart_cost_scaling(r: dict) -> None:
    ag_per_doc = r["autograft"]["cost_usd"] / 500.0
    full_per_doc = r["full_llm_er"]["cost_usd"] / 500.0
    ag = [n * ag_per_doc for n in SCALES]
    full = [n * full_per_doc for n in SCALES]
    import numpy as np

    x = np.arange(len(SCALES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w / 2, full, w, color="#ff7f0e", label="Full LLM ER")
    b2 = ax.bar(x + w / 2, ag, w, color="#2ca02c", label="AutoGraft hybrid")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"${h:,.0f}", (b.get_x() + b.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in SCALES])
    ax.set_xlabel("Documents processed")
    ax.set_ylabel("Cumulative LLM ER cost (USD)")
    ax.set_title("Cost scaling - linear projection from measured per-doc cost")
    ax.legend(); ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig("benchmark/assets/real_cost_scaling.png", dpi=150); plt.close(fig)


def chart_accuracy(r: dict) -> None:
    a = r["accuracy"]
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["precision", "recall", "f1"]
    vals = [a["precision"], a["recall"], a["f1"]]
    bars = ax.bar(names, vals, color=["#2ca02c", "#1f77b4", "#9467bd"])
    for b, v in zip(bars, vals):
        ax.annotate(f"{v*100:.1f}%", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title(f"Entity resolution accuracy vs ground truth\n"
                 f"(correct={int(a['correct_merges'])} wrong={int(a['wrong_merges'])} "
                 f"missed={int(a['declined_merges'])} of {int(a['true_merges_possible'])} possible)")
    fig.tight_layout(); fig.savefig("benchmark/assets/real_accuracy.png", dpi=150); plt.close(fig)


def generate_all() -> None:
    r = _load()
    chart_metrics(r)
    chart_layers(r)
    chart_cost_scaling(r)
    chart_accuracy(r)
    print("Charts written to benchmark/assets/real_*.png")


if __name__ == "__main__":
    generate_all()
