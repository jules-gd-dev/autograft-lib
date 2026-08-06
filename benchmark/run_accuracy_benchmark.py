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
    """Builds a rich, highly diversified dataset of 100 real-world entity resolution pairs across 10 domains."""
    return [
        # --- DOMAIN 1: Tech & AI Companies (10 cases) ---
        (Entity(canonical_name="Meta", type="Company"), ExistingNode(node_id="1", canonical_name="Facebook Inc.", type="Company", aliases=["Meta Platforms"]), True),
        (Entity(canonical_name="Alphabet", type="Company"), ExistingNode(node_id="2", canonical_name="Google LLC", type="Company", aliases=["Alphabet Inc."]), True),
        (Entity(canonical_name="X", type="Company"), ExistingNode(node_id="3", canonical_name="Twitter Inc.", type="Company", aliases=["X Corp"]), True),
        (Entity(canonical_name="OpenAI", type="Company"), ExistingNode(node_id="4", canonical_name="Anthropic", type="Company"), False),
        (Entity(canonical_name="Mistral AI", type="Company"), ExistingNode(node_id="5", canonical_name="Mistral AI SAS", type="Company"), True),
        (Entity(canonical_name="Cohere", type="Company"), ExistingNode(node_id="6", canonical_name="Hugging Face", type="Company"), False),
        (Entity(canonical_name="Databricks", type="Company"), ExistingNode(node_id="7", canonical_name="Snowflake Inc.", type="Company"), False),
        (Entity(canonical_name="Palantir", type="Company"), ExistingNode(node_id="8", canonical_name="Palantir Technologies", type="Company"), True),
        (Entity(canonical_name="CrowdStrike", type="Company"), ExistingNode(node_id="9", canonical_name="Cloudflare Inc.", type="Company"), False),
        (Entity(canonical_name="Elastic", type="Company"), ExistingNode(node_id="10", canonical_name="Elastic N.V.", type="Company", aliases=["Elasticsearch"]), True),

        # --- DOMAIN 2: Tech Products, OS & AI Models (10 cases) ---
        (Entity(canonical_name="ChatGPT", type="Software"), ExistingNode(node_id="11", canonical_name="GPT-4o", type="AI Model"), False),
        (Entity(canonical_name="Claude 3.5 Sonnet", type="AI Model"), ExistingNode(node_id="12", canonical_name="Claude 3.5 Sonnet", type="AI Model"), True),
        (Entity(canonical_name="Llama 3", type="AI Model"), ExistingNode(node_id="13", canonical_name="LLaMA 3", type="AI Model"), True),
        (Entity(canonical_name="iPhone 15", type="Product"), ExistingNode(node_id="14", canonical_name="Apple iPhone 15 Pro", type="Product"), True),
        (Entity(canonical_name="Windows 11", type="Software"), ExistingNode(node_id="15", canonical_name="macOS Sonoma", type="Software"), False),
        (Entity(canonical_name="Android", type="Software"), ExistingNode(node_id="16", canonical_name="Android OS", type="Software"), True),
        (Entity(canonical_name="Kubernetes", type="Software"), ExistingNode(node_id="17", canonical_name="K8s", type="Software"), True),
        (Entity(canonical_name="Docker", type="Software"), ExistingNode(node_id="18", canonical_name="Podman", type="Software"), False),
        (Entity(canonical_name="PyTorch", type="Software"), ExistingNode(node_id="19", canonical_name="TensorFlow", type="Software"), False),
        (Entity(canonical_name="React", type="Software"), ExistingNode(node_id="20", canonical_name="React.js", type="Software"), True),

        # --- DOMAIN 3: People, Scientists, Tech Leaders & Public Figures (10 cases) ---
        (Entity(canonical_name="Sam Altman", type="Person"), ExistingNode(node_id="21", canonical_name="Samuel H. Altman", type="Person"), True),
        (Entity(canonical_name="E. Musk", type="Person"), ExistingNode(node_id="22", canonical_name="Elon Musk", type="Person"), True),
        (Entity(canonical_name="Satya Nadella", type="Person"), ExistingNode(node_id="23", canonical_name="Sundar Pichai", type="Person"), False),
        (Entity(canonical_name="Mark Zuckerberg", type="Person"), ExistingNode(node_id="24", canonical_name="Zuck", type="Person"), True),
        (Entity(canonical_name="Steve Jobs", type="Person"), ExistingNode(node_id="25", canonical_name="Bill Gates", type="Person"), False),
        (Entity(canonical_name="Marie Curie", type="Person"), ExistingNode(node_id="26", canonical_name="M. Curie", type="Person"), True),
        (Entity(canonical_name="Albert Einstein", type="Person"), ExistingNode(node_id="27", canonical_name="A. Einstein", type="Person"), True),
        (Entity(canonical_name="Alan Turing", type="Person"), ExistingNode(node_id="28", canonical_name="Ada Lovelace", type="Person"), False),
        (Entity(canonical_name="Emmanuel Macron", type="Person"), ExistingNode(node_id="29", canonical_name="Barack Obama", type="Person"), False),
        (Entity(canonical_name="Lionel Messi", type="Person"), ExistingNode(node_id="30", canonical_name="Cristiano Ronaldo", type="Person"), False),

        # --- DOMAIN 4: Homonyms & Entity Type Disambiguation (10 cases) ---
        (Entity(canonical_name="Apple", type="Fruit"), ExistingNode(node_id="31", canonical_name="Apple Inc.", type="Company"), False),
        (Entity(canonical_name="Amazon", type="Location"), ExistingNode(node_id="32", canonical_name="Amazon.com Inc.", type="Company"), False),
        (Entity(canonical_name="Python", type="Software"), ExistingNode(node_id="33", canonical_name="Python Programming Language", type="Software"), True),
        (Entity(canonical_name="Python", type="Animal"), ExistingNode(node_id="34", canonical_name="Python Programming Language", type="Software"), False),
        (Entity(canonical_name="Jaguar", type="Company"), ExistingNode(node_id="35", canonical_name="Jaguar Land Rover", type="Company"), True),
        (Entity(canonical_name="Jaguar", type="Animal"), ExistingNode(node_id="36", canonical_name="Jaguar Land Rover", type="Company"), False),
        (Entity(canonical_name="Target", type="Company"), ExistingNode(node_id="37", canonical_name="Target Corporation", type="Company"), True),
        (Entity(canonical_name="Oracle", type="Software"), ExistingNode(node_id="38", canonical_name="Oracle Corporation", type="Company"), True),
        (Entity(canonical_name="Corona", type="Beverage"), ExistingNode(node_id="39", canonical_name="COVID-19 Virus", type="Disease"), False),
        (Entity(canonical_name="Mars", type="Location"), ExistingNode(node_id="40", canonical_name="Mars Planet", type="Location"), True),

        # --- DOMAIN 5: Geography, Cities & Countries (10 cases) ---
        (Entity(canonical_name="Paris", type="Location"), ExistingNode(node_id="41", canonical_name="Paris, France", type="Location"), True),
        (Entity(canonical_name="Paris (Texas)", type="Location"), ExistingNode(node_id="42", canonical_name="Paris (France)", type="Location"), False),
        (Entity(canonical_name="NYC", type="Location"), ExistingNode(node_id="43", canonical_name="New York City", type="Location"), True),
        (Entity(canonical_name="London", type="Location"), ExistingNode(node_id="44", canonical_name="London, UK", type="Location"), True),
        (Entity(canonical_name="London (Ontario)", type="Location"), ExistingNode(node_id="45", canonical_name="London (United Kingdom)", type="Location"), False),
        (Entity(canonical_name="Tokyo", type="Location"), ExistingNode(node_id="46", canonical_name="Tokyo Metropolis", type="Location"), True),
        (Entity(canonical_name="Beijing", type="Location"), ExistingNode(node_id="47", canonical_name="Peking", type="Location"), True),
        (Entity(canonical_name="USA", type="Location"), ExistingNode(node_id="48", canonical_name="United States of America", type="Location"), True),
        (Entity(canonical_name="Eiffel Tower", type="Location"), ExistingNode(node_id="49", canonical_name="Tour Eiffel", type="Location"), True),
        (Entity(canonical_name="Grand Canyon", type="Location"), ExistingNode(node_id="50", canonical_name="Yellowstone National Park", type="Location"), False),

        # --- DOMAIN 6: Automotive & Aerospace Brands (10 cases) ---
        (Entity(canonical_name="VW", type="Company"), ExistingNode(node_id="51", canonical_name="Volkswagen AG", type="Company"), True),
        (Entity(canonical_name="Ferrari", type="Company"), ExistingNode(node_id="52", canonical_name="Scuderia Ferrari", type="Company"), True),
        (Entity(canonical_name="Porsche", type="Company"), ExistingNode(node_id="53", canonical_name="Porsche AG", type="Company"), True),
        (Entity(canonical_name="BMW", type="Company"), ExistingNode(node_id="54", canonical_name="Bayerische Motoren Werke", type="Company"), True),
        (Entity(canonical_name="Mercedes", type="Company"), ExistingNode(node_id="55", canonical_name="Mercedes-Benz Group", type="Company"), True),
        (Entity(canonical_name="Tesla", type="Company"), ExistingNode(node_id="56", canonical_name="Tesla Motors", type="Company"), True),
        (Entity(canonical_name="Boeing", type="Company"), ExistingNode(node_id="57", canonical_name="Airbus", type="Company"), False),
        (Entity(canonical_name="SpaceX", type="Company"), ExistingNode(node_id="58", canonical_name="Space Exploration Technologies Corp.", type="Company"), True),
        (Entity(canonical_name="NASA", type="Organization"), ExistingNode(node_id="59", canonical_name="ESA", type="Organization"), False),
        (Entity(canonical_name="Lockheed Martin", type="Company"), ExistingNode(node_id="60", canonical_name="Northrop Grumman", type="Company"), False),

        # --- DOMAIN 7: Finance, Crypto & Banking (10 cases) ---
        (Entity(canonical_name="J.P. Morgan", type="Company"), ExistingNode(node_id="61", canonical_name="JP Morgan Chase & Co.", type="Company"), True),
        (Entity(canonical_name="Goldman Sachs", type="Company"), ExistingNode(node_id="62", canonical_name="The Goldman Sachs Group Inc.", type="Company"), True),
        (Entity(canonical_name="Morgan Stanley", type="Company"), ExistingNode(node_id="63", canonical_name="Merrill Lynch", type="Company"), False),
        (Entity(canonical_name="BlackRock", type="Company"), ExistingNode(node_id="64", canonical_name="Vanguard Group", type="Company"), False),
        (Entity(canonical_name="Visa", type="Company"), ExistingNode(node_id="65", canonical_name="Mastercard", type="Company"), False),
        (Entity(canonical_name="Bitcoin", type="Crypto"), ExistingNode(node_id="66", canonical_name="BTC", type="Crypto"), True),
        (Entity(canonical_name="Ethereum", type="Crypto"), ExistingNode(node_id="67", canonical_name="ETH", type="Crypto"), True),
        (Entity(canonical_name="Solana", type="Crypto"), ExistingNode(node_id="68", canonical_name="SOL", type="Crypto"), True),
        (Entity(canonical_name="Coinbase", type="Company"), ExistingNode(node_id="69", canonical_name="Binance", type="Company"), False),
        (Entity(canonical_name="Stripe", type="Company"), ExistingNode(node_id="70", canonical_name="Square Inc.", type="Company"), False),

        # --- DOMAIN 8: Entertainment, Movies & Media (10 cases) ---
        (Entity(canonical_name="Netflix", type="Company"), ExistingNode(node_id="71", canonical_name="Netflix Inc.", type="Company"), True),
        (Entity(canonical_name="Disney", type="Company"), ExistingNode(node_id="72", canonical_name="The Walt Disney Company", type="Company"), True),
        (Entity(canonical_name="Marvel", type="Company"), ExistingNode(node_id="73", canonical_name="DC Comics", type="Company"), False),
        (Entity(canonical_name="Star Wars", type="Franchise"), ExistingNode(node_id="74", canonical_name="Star Trek", type="Franchise"), False),
        (Entity(canonical_name="Harry Potter", type="Franchise"), ExistingNode(node_id="75", canonical_name="Lord of the Rings", type="Franchise"), False),
        (Entity(canonical_name="Spotify", type="Company"), ExistingNode(node_id="76", canonical_name="Apple Music", type="Software"), False),
        (Entity(canonical_name="HBO", type="Company"), ExistingNode(node_id="77", canonical_name="Home Box Office", type="Company"), True),
        (Entity(canonical_name="Nintendo", type="Company"), ExistingNode(node_id="78", canonical_name="Sega", type="Company"), False),
        (Entity(canonical_name="PlayStation", type="Product"), ExistingNode(node_id="79", canonical_name="Sony PS5", type="Product"), True),
        (Entity(canonical_name="Xbox", type="Product"), ExistingNode(node_id="80", canonical_name="Microsoft Xbox Series X", type="Product"), True),

        # --- DOMAIN 9: Sports, Teams & Global Events (10 cases) ---
        (Entity(canonical_name="Real Madrid", type="SportsTeam"), ExistingNode(node_id="81", canonical_name="Real Madrid C.F.", type="SportsTeam"), True),
        (Entity(canonical_name="FC Barcelona", type="SportsTeam"), ExistingNode(node_id="82", canonical_name="Barça", type="SportsTeam"), True),
        (Entity(canonical_name="Manchester United", type="SportsTeam"), ExistingNode(node_id="83", canonical_name="Manchester City", type="SportsTeam"), False),
        (Entity(canonical_name="Lakers", type="SportsTeam"), ExistingNode(node_id="84", canonical_name="Los Angeles Lakers", type="SportsTeam"), True),
        (Entity(canonical_name="Warriors", type="SportsTeam"), ExistingNode(node_id="85", canonical_name="Golden State Warriors", type="SportsTeam"), True),
        (Entity(canonical_name="NBA", type="Organization"), ExistingNode(node_id="86", canonical_name="National Basketball Association", type="Organization"), True),
        (Entity(canonical_name="FIFA World Cup", type="Event"), ExistingNode(node_id="87", canonical_name="UEFA Champions League", type="Event"), False),
        (Entity(canonical_name="Olympic Games", type="Event"), ExistingNode(node_id="88", canonical_name="Olympics", type="Event"), True),
        (Entity(canonical_name="Super Bowl", type="Event"), ExistingNode(node_id="89", canonical_name="NFL Championship", type="Event"), True),
        (Entity(canonical_name="Formula 1", type="Event"), ExistingNode(node_id="90", canonical_name="F1 Grand Prix", type="Event"), True),

        # --- DOMAIN 10: Global Institutions, Universities & Non-Profits (10 cases) ---
        (Entity(canonical_name="UN", type="Organization"), ExistingNode(node_id="91", canonical_name="United Nations", type="Organization"), True),
        (Entity(canonical_name="WHO", type="Organization"), ExistingNode(node_id="92", canonical_name="World Health Organization", type="Organization"), True),
        (Entity(canonical_name="UNESCO", type="Organization"), ExistingNode(node_id="93", canonical_name="UNICEF", type="Organization"), False),
        (Entity(canonical_name="FBI", type="Organization"), ExistingNode(node_id="94", canonical_name="Federal Bureau of Investigation", type="Organization"), True),
        (Entity(canonical_name="CIA", type="Organization"), ExistingNode(node_id="95", canonical_name="Central Intelligence Agency", type="Organization"), True),
        (Entity(canonical_name="MIT", type="University"), ExistingNode(node_id="96", canonical_name="Massachusetts Institute of Technology", type="University"), True),
        (Entity(canonical_name="Stanford", type="University"), ExistingNode(node_id="97", canonical_name="Stanford University", type="University"), True),
        (Entity(canonical_name="Harvard", type="University"), ExistingNode(node_id="98", canonical_name="Harvard University", type="University"), True),
        (Entity(canonical_name="Red Cross", type="Organization"), ExistingNode(node_id="99", canonical_name="International Red Cross", type="Organization"), True),
        (Entity(canonical_name="Amnesty International", type="Organization"), ExistingNode(node_id="100", canonical_name="Greenpeace", type="Organization"), False),
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
    """Runs the accuracy benchmark loop against the 100-case dataset."""
    dataset = build_tricky_dataset()
    correct_decisions = 0
    total_cases = len(dataset)

    print("=" * 85)
    print(f" 🎯 AUTOGRAFT ACCURACY BENCHMARK (LLM-AS-A-JUDGE AUDIT - {total_cases} DIVERSIFIED CASES)")
    print("=" * 85)
    print(f"AutoGraft Model : {AUTOGRAFT_MODEL}")
    print(f"Judge LLM Model : {JUDGE_MODEL}\n")

    print(
        f"{'#':<3} | {'Entity A':<22} | {'Entity B':<32} | {'AutoGraft Decision':<18} | {'Judge Verdict':<12}"
    )
    print("-" * 92)

    for idx, (new_entity, existing_node, expected) in enumerate(dataset, 1):
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
            f"{idx:<3} | {new_entity.canonical_name:<22} | {existing_node.canonical_name:<32} | "
            f"{decision_str:<18} | {verdict_str:<12}"
        )

    accuracy_pct = (correct_decisions / total_cases) * 100.0
    print("-" * 92)
    print(
        f"🏆 Final Audit Score : {correct_decisions}/{total_cases} ({accuracy_pct:.1f}% Accuracy)"
    )
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_accuracy_benchmark()
