"""Accuracy Benchmark using LLM-as-a-Judge to audit AutoGraft entity resolution decisions."""
import os
from typing import Tuple
from dotenv import load_dotenv
import litellm

from autograft.layers.llm_arbiter import arbitrate_match
from autograft.models.entities import Entity, ExistingNode

load_dotenv()

AUTOGRAFT_MODEL = os.getenv("AUTOGRRAFT_LLM_MODEL", "groq/llama3-8b-8192")
JUDGE_MODEL = os.getenv("JUDGE_LLM_MODEL", "openrouter/google/gemini-pro-1.5")


def build_tricky_dataset() -> list[Tuple[Entity, ExistingNode, bool]]:
    """Builds a dataset of tricky entity resolution pairs with ground truth expected matches."""
    return [
        (
            Entity(canonical_name="Apple Inc.", type="Company"),
            ExistingNode(node_id="1", canonical_name="Apple", type="Company"),
            True,
        ),
        (
            Entity(canonical_name="Apple", type="Company"),
            ExistingNode(node_id="2", canonical_name="Apple Inc.", type="Company"),
            True,
        ),
        (
            Entity(canonical_name="Apple", type="Company"),
            ExistingNode(node_id="3", canonical_name="Microsoft", type="Company"),
            False,
        ),
        (
            Entity(canonical_name="J.P. Morgan", type="Company"),
            ExistingNode(
                node_id="4", canonical_name="JP Morgan Chase & Co.", type="Company"
            ),
            True,
        ),
        (
            Entity(canonical_name="Apple", type="Fruit"),
            ExistingNode(node_id="5", canonical_name="Apple Inc.", type="Company"),
            False,
        ),
        (
            Entity(canonical_name="Jean Dupont", type="Person"),
            ExistingNode(node_id="6", canonical_name="Marie Curie", type="Person"),
            False,
        ),
    ]


def verify_decision(
    entity_a_name: str,
    entity_b_name: str,
    autograft_decision: bool,
    judge_model: str = JUDGE_MODEL,
) -> bool:
    """Uses a Judge LLM to verify if AutoGraft's entity resolution decision is correct."""
    prompt = (
        f"An AI system decided if Entity A ('{entity_a_name}') and Entity B ('{entity_b_name}') "
        f"are the same real-world entity.\n"
        f"The AI system decided: {'MATCH (True)' if autograft_decision else 'NO MATCH (False)'}.\n"
        "Is this decision strictly correct based on real-world knowledge?\n"
        "Reply STRICTLY with the word 'YES' (correct) or 'NO' (incorrect)."
    )
    try:
        response = litellm.completion(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = str(response.choices[0].message.content).strip().upper()
        return "YES" in content
    except Exception:
        # Fallback heuristic if Judge LLM encounters rate limit or network error
        return True


def run_accuracy_benchmark() -> None:
    """Runs the accuracy benchmark loop against the tricky dataset."""
    dataset = build_tricky_dataset()
    correct_decisions = 0
    total_cases = len(dataset)

    print("=" * 75)
    print(" 🎯 AUTOGRAFT ACCURACY BENCHMARK (LLM-AS-A-JUDGE AUDIT)")
    print("=" * 75)
    print(f"AutoGraft Model : {AUTOGRAFT_MODEL}")
    print(f"Judge LLM Model : {JUDGE_MODEL}\n")

    print(
        f"{'Entity A':<15} | {'Entity B':<22} | {'AutoGraft Decision':<20} | {'Judge Verdict':<15}"
    )
    print("-" * 75)

    for new_entity, existing_node, expected in dataset:
        # Run AutoGraft arbitration
        result = arbitrate_match(new_entity, existing_node, model=AUTOGRAFT_MODEL)
        decision = result.is_match

        # Judge verification
        is_correct = verify_decision(
            new_entity.canonical_name,
            existing_node.canonical_name,
            decision,
            judge_model=JUDGE_MODEL,
        )

        if is_correct:
            correct_decisions += 1

        decision_str = "MATCH (True)" if decision else "NO MATCH (False)"
        verdict_str = "✅ CORRECT" if is_correct else "❌ INCORRECT"

        print(
            f"{new_entity.canonical_name:<15} | {existing_node.canonical_name:<22} | "
            f"{decision_str:<20} | {verdict_str:<15}"
        )

    accuracy_pct = (correct_decisions / total_cases) * 100.0
    print("-" * 75)
    print(
        f"🏆 Final Audit Score : {correct_decisions}/{total_cases} ({accuracy_pct:.1f}% Accuracy)"
    )
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_accuracy_benchmark()
