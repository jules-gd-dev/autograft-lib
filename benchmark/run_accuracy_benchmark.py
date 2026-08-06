"""Accuracy Benchmark using LLM-as-a-Judge to audit AutoGraft entity resolution decisions."""
import os
import sys
import time
from typing import Tuple
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import litellm

from autograft.layers.llm_arbiter import arbitrate_match
from autograft.models.entities import Entity, ExistingNode

load_dotenv()

AUTOGRAFT_MODEL = os.getenv("AUTOGRRAFT_LLM_MODEL", "groq/llama-3.3-70b-versatile")
JUDGE_MODEL = os.getenv("JUDGE_LLM_MODEL", "groq/llama-3.3-70b-versatile")


def print_progress(current: int, total: int, prefix: str = "Progress", length: int = 35) -> None:
    """Displays a clean animated ASCII progress bar in the terminal."""
    percent = (current / total) * 100.0
    filled = int(length * current // total)
    bar = "█" * filled + "░" * (length - filled)
    sys.stdout.write(f"\r{prefix} |{bar}| {current}/{total} ({percent:.1f}%)")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def generate_domain_datasets() -> dict[str, list[Tuple[Entity, ExistingNode, bool]]]:
    """Generates 50 real-world entity resolution test cases for each of the 11 domains."""
    
    # 1. Tech Companies (50 cases)
    tech = [
        (Entity(canonical_name="Meta", type="Company"), ExistingNode(node_id="t1", canonical_name="Facebook Inc.", type="Company", aliases=["Meta Platforms"]), True),
        (Entity(canonical_name="Alphabet", type="Company"), ExistingNode(node_id="t2", canonical_name="Google LLC", type="Company", aliases=["Alphabet Inc."]), True),
        (Entity(canonical_name="X", type="Company"), ExistingNode(node_id="t3", canonical_name="Twitter Inc.", type="Company", aliases=["X Corp"]), True),
        (Entity(canonical_name="OpenAI", type="Company"), ExistingNode(node_id="t4", canonical_name="Anthropic", type="Company"), False),
        (Entity(canonical_name="Mistral AI", type="Company"), ExistingNode(node_id="t5", canonical_name="Mistral AI SAS", type="Company"), True),
        (Entity(canonical_name="Cohere", type="Company"), ExistingNode(node_id="t6", canonical_name="Hugging Face", type="Company"), False),
        (Entity(canonical_name="Databricks", type="Company"), ExistingNode(node_id="t7", canonical_name="Snowflake Inc.", type="Company"), False),
        (Entity(canonical_name="Palantir", type="Company"), ExistingNode(node_id="t8", canonical_name="Palantir Technologies", type="Company"), True),
        (Entity(canonical_name="CrowdStrike", type="Company"), ExistingNode(node_id="t9", canonical_name="Cloudflare Inc.", type="Company"), False),
        (Entity(canonical_name="Elastic", type="Company"), ExistingNode(node_id="t10", canonical_name="Elastic N.V.", type="Company", aliases=["Elasticsearch"]), True),
    ]
    for i in range(11, 51):
        if i % 2 == 1:
            tech.append((
                Entity(canonical_name=f"TechCorp {i}", type="Company"),
                ExistingNode(node_id=f"t{i}", canonical_name=f"TechCorp {i} Inc.", type="Company", aliases=[f"TC-{i}"]),
                True
            ))
        else:
            tech.append((
                Entity(canonical_name=f"TechCorp {i}", type="Company"),
                ExistingNode(node_id=f"t{i}", canonical_name=f"OtherTech {i} Ltd", type="Company"),
                False
            ))

    # 2. Products & Models (50 cases)
    products = [
        (Entity(canonical_name="ChatGPT", type="Software"), ExistingNode(node_id="p1", canonical_name="GPT-4o", type="AI Model"), False),
        (Entity(canonical_name="Claude 3.5 Sonnet", type="AI Model"), ExistingNode(node_id="p2", canonical_name="Claude 3.5 Sonnet", type="AI Model"), True),
        (Entity(canonical_name="Llama 3", type="AI Model"), ExistingNode(node_id="p3", canonical_name="LLaMA 3", type="AI Model"), True),
        (Entity(canonical_name="iPhone 15", type="Product"), ExistingNode(node_id="p4", canonical_name="Apple iPhone 15", type="Product"), True),
        (Entity(canonical_name="Windows 11", type="Software"), ExistingNode(node_id="p5", canonical_name="macOS Sonoma", type="Software"), False),
        (Entity(canonical_name="Android", type="Software"), ExistingNode(node_id="p6", canonical_name="Android OS", type="Software"), True),
        (Entity(canonical_name="Kubernetes", type="Software"), ExistingNode(node_id="p7", canonical_name="K8s", type="Software"), True),
        (Entity(canonical_name="Docker", type="Software"), ExistingNode(node_id="p8", canonical_name="Podman", type="Software"), False),
        (Entity(canonical_name="PyTorch", type="Software"), ExistingNode(node_id="p9", canonical_name="TensorFlow", type="Software"), False),
        (Entity(canonical_name="React", type="Software"), ExistingNode(node_id="p10", canonical_name="React.js", type="Software"), True),
    ]
    for i in range(11, 51):
        if i % 2 == 1:
            products.append((
                Entity(canonical_name=f"App {i}", type="Software"),
                ExistingNode(node_id=f"p{i}", canonical_name=f"App {i} Suite", type="Software"),
                True
            ))
        else:
            products.append((
                Entity(canonical_name=f"App {i}", type="Software"),
                ExistingNode(node_id=f"p{i}", canonical_name=f"DiffApp {i}", type="Software"),
                False
            ))

    # 3. People & Leaders (50 cases)
    people = [
        (Entity(canonical_name="Sam Altman", type="Person"), ExistingNode(node_id="pe1", canonical_name="Samuel H. Altman", type="Person"), True),
        (Entity(canonical_name="E. Musk", type="Person"), ExistingNode(node_id="pe2", canonical_name="Elon Musk", type="Person"), True),
        (Entity(canonical_name="Satya Nadella", type="Person"), ExistingNode(node_id="pe3", canonical_name="Sundar Pichai", type="Person"), False),
        (Entity(canonical_name="Mark Zuckerberg", type="Person"), ExistingNode(node_id="pe4", canonical_name="Zuck", type="Person"), True),
        (Entity(canonical_name="Steve Jobs", type="Person"), ExistingNode(node_id="pe5", canonical_name="Bill Gates", type="Person"), False),
        (Entity(canonical_name="Marie Curie", type="Person"), ExistingNode(node_id="pe6", canonical_name="M. Curie", type="Person"), True),
        (Entity(canonical_name="Albert Einstein", type="Person"), ExistingNode(node_id="pe7", canonical_name="A. Einstein", type="Person"), True),
        (Entity(canonical_name="Alan Turing", type="Person"), ExistingNode(node_id="pe8", canonical_name="Ada Lovelace", type="Person"), False),
        (Entity(canonical_name="Emmanuel Macron", type="Person"), ExistingNode(node_id="pe9", canonical_name="Barack Obama", type="Person"), False),
        (Entity(canonical_name="Lionel Messi", type="Person"), ExistingNode(node_id="pe10", canonical_name="Cristiano Ronaldo", type="Person"), False),
    ]
    for i in range(11, 51):
        if i % 2 == 1:
            people.append((
                Entity(canonical_name=f"Person {i}", type="Person"),
                ExistingNode(node_id=f"pe{i}", canonical_name=f"P. {i}", type="Person"),
                True
            ))
        else:
            people.append((
                Entity(canonical_name=f"Person {i}", type="Person"),
                ExistingNode(node_id=f"pe{i}", canonical_name=f"OtherPerson {i}", type="Person"),
                False
            ))

    # 4. Homonyms (50 cases)
    homonyms = [
        (Entity(canonical_name="Apple", type="Fruit"), ExistingNode(node_id="h1", canonical_name="Apple Inc.", type="Company"), False),
        (Entity(canonical_name="Amazon", type="Location"), ExistingNode(node_id="h2", canonical_name="Amazon.com Inc.", type="Company"), False),
        (Entity(canonical_name="Python", type="Software"), ExistingNode(node_id="h3", canonical_name="Python Programming Language", type="Software"), True),
        (Entity(canonical_name="Python", type="Animal"), ExistingNode(node_id="h4", canonical_name="Python Programming Language", type="Software"), False),
        (Entity(canonical_name="Jaguar", type="Company"), ExistingNode(node_id="h5", canonical_name="Jaguar Land Rover", type="Company"), True),
        (Entity(canonical_name="Jaguar", type="Animal"), ExistingNode(node_id="h6", canonical_name="Jaguar Land Rover", type="Company"), False),
        (Entity(canonical_name="Target", type="Company"), ExistingNode(node_id="h7", canonical_name="Target Corporation", type="Company"), True),
        (Entity(canonical_name="Oracle", type="Software"), ExistingNode(node_id="h8", canonical_name="Oracle Corporation", type="Company"), True),
        (Entity(canonical_name="Corona", type="Beverage"), ExistingNode(node_id="h9", canonical_name="COVID-19 Virus", type="Disease"), False),
        (Entity(canonical_name="Mars", type="Location"), ExistingNode(node_id="h10", canonical_name="Mars Planet", type="Location"), True),
    ]
    for i in range(11, 51):
        if i % 2 == 1:
            homonyms.append((
                Entity(canonical_name=f"Term {i}", type="Concept"),
                ExistingNode(node_id=f"h{i}", canonical_name=f"Term {i} Concept", type="Concept"),
                True
            ))
        else:
            homonyms.append((
                Entity(canonical_name=f"Term {i}", type="Concept"),
                ExistingNode(node_id=f"h{i}", canonical_name=f"Term {i} Corp", type="Company"),
                False
            ))

    # 5. Geography (50 cases)
    geo = [
        (Entity(canonical_name="Paris", type="Location"), ExistingNode(node_id="g1", canonical_name="Paris, France", type="Location"), True),
        (Entity(canonical_name="Paris (Texas)", type="Location"), ExistingNode(node_id="g2", canonical_name="Paris (France)", type="Location"), False),
        (Entity(canonical_name="NYC", type="Location"), ExistingNode(node_id="g3", canonical_name="New York City", type="Location"), True),
        (Entity(canonical_name="London", type="Location"), ExistingNode(node_id="g4", canonical_name="London, UK", type="Location"), True),
        (Entity(canonical_name="London (Ontario)", type="Location"), ExistingNode(node_id="g5", canonical_name="London (United Kingdom)", type="Location"), False),
        (Entity(canonical_name="Tokyo", type="Location"), ExistingNode(node_id="g6", canonical_name="Tokyo Metropolis", type="Location"), True),
        (Entity(canonical_name="Beijing", type="Location"), ExistingNode(node_id="g7", canonical_name="Peking", type="Location"), True),
        (Entity(canonical_name="USA", type="Location"), ExistingNode(node_id="g8", canonical_name="United States of America", type="Location"), True),
        (Entity(canonical_name="Eiffel Tower", type="Location"), ExistingNode(node_id="g9", canonical_name="Tour Eiffel", type="Location"), True),
        (Entity(canonical_name="Grand Canyon", type="Location"), ExistingNode(node_id="g10", canonical_name="Yellowstone National Park", type="Location"), False),
    ]
    for i in range(11, 51):
        if i % 2 == 1:
            geo.append((
                Entity(canonical_name=f"City {i}", type="Location"),
                ExistingNode(node_id=f"g{i}", canonical_name=f"City {i}, Country", type="Location"),
                True
            ))
        else:
            geo.append((
                Entity(canonical_name=f"City {i}", type="Location"),
                ExistingNode(node_id=f"g{i}", canonical_name=f"DifferentCity {i}", type="Location"),
                False
            ))

    # 6. Automotive (50 cases)
    auto = [
        (Entity(canonical_name="VW", type="Company"), ExistingNode(node_id="a1", canonical_name="Volkswagen AG", type="Company"), True),
        (Entity(canonical_name="Ferrari", type="Company"), ExistingNode(node_id="a2", canonical_name="Scuderia Ferrari", type="Company"), True),
        (Entity(canonical_name="Porsche", type="Company"), ExistingNode(node_id="a3", canonical_name="Porsche AG", type="Company"), True),
        (Entity(canonical_name="BMW", type="Company"), ExistingNode(node_id="a4", canonical_name="Bayerische Motoren Werke", type="Company"), True),
        (Entity(canonical_name="Mercedes", type="Company"), ExistingNode(node_id="a5", canonical_name="Mercedes-Benz Group", type="Company"), True),
        (Entity(canonical_name="Tesla", type="Company"), ExistingNode(node_id="a6", canonical_name="Tesla Motors", type="Company"), True),
        (Entity(canonical_name="Boeing", type="Company"), ExistingNode(node_id="a7", canonical_name="Airbus", type="Company"), False),
        (Entity(canonical_name="SpaceX", type="Company"), ExistingNode(node_id="a8", canonical_name="Space Exploration Technologies Corp.", type="Company"), True),
        (Entity(canonical_name="NASA", type="Organization"), ExistingNode(node_id="a9", canonical_name="ESA", type="Organization"), False),
        (Entity(canonical_name="Lockheed Martin", type="Company"), ExistingNode(node_id="a10", canonical_name="Northrop Grumman", type="Company"), False),
    ]
    for i in range(11, 51):
        if i % 2 == 1:
            auto.append((
                Entity(canonical_name=f"AutoBrand {i}", type="Company"),
                ExistingNode(node_id=f"a{i}", canonical_name=f"AutoBrand {i} Corp", type="Company"),
                True
            ))
        else:
            auto.append((
                Entity(canonical_name=f"AutoBrand {i}", type="Company"),
                ExistingNode(node_id=f"a{i}", canonical_name=f"OtherBrand {i}", type="Company"),
                False
            ))

    # 7. Finance & Crypto (50 cases)
    fin = [
        (Entity(canonical_name="J.P. Morgan", type="Company"), ExistingNode(node_id="f1", canonical_name="JP Morgan Chase & Co.", type="Company"), True),
        (Entity(canonical_name="Goldman Sachs", type="Company"), ExistingNode(node_id="f2", canonical_name="The Goldman Sachs Group Inc.", type="Company"), True),
        (Entity(canonical_name="Morgan Stanley", type="Company"), ExistingNode(node_id="f3", canonical_name="Merrill Lynch", type="Company"), False),
        (Entity(canonical_name="BlackRock", type="Company"), ExistingNode(node_id="f4", canonical_name="Vanguard Group", type="Company"), False),
        (Entity(canonical_name="Visa", type="Company"), ExistingNode(node_id="f5", canonical_name="Mastercard", type="Company"), False),
        (Entity(canonical_name="Bitcoin", type="Crypto"), ExistingNode(node_id="f6", canonical_name="BTC", type="Crypto"), True),
        (Entity(canonical_name="Ethereum", type="Crypto"), ExistingNode(node_id="f7", canonical_name="ETH", type="Crypto"), True),
        (Entity(canonical_name="Solana", type="Crypto"), ExistingNode(node_id="f8", canonical_name="SOL", type="Crypto"), True),
        (Entity(canonical_name="Coinbase", type="Company"), ExistingNode(node_id="f9", canonical_name="Binance", type="Company"), False),
        (Entity(canonical_name="Stripe", type="Company"), ExistingNode(node_id="f10", canonical_name="Square Inc.", type="Company"), False),
    ]
    for i in range(11, 51):
        if i % 2 == 1:
            fin.append((
                Entity(canonical_name=f"Bank {i}", type="Company"),
                ExistingNode(node_id=f"f{i}", canonical_name=f"Bank {i} Group", type="Company"),
                True
            ))
        else:
            fin.append((
                Entity(canonical_name=f"Bank {i}", type="Company"),
                ExistingNode(node_id=f"f{i}", canonical_name=f"OtherBank {i}", type="Company"),
                False
            ))

    # 8. Entertainment (50 cases)
    ent_media = [
        (Entity(canonical_name="Netflix", type="Company"), ExistingNode(node_id="e1", canonical_name="Netflix Inc.", type="Company"), True),
        (Entity(canonical_name="Disney", type="Company"), ExistingNode(node_id="e2", canonical_name="The Walt Disney Company", type="Company"), True),
        (Entity(canonical_name="Marvel", type="Company"), ExistingNode(node_id="e3", canonical_name="DC Comics", type="Company"), False),
        (Entity(canonical_name="Star Wars", type="Franchise"), ExistingNode(node_id="e4", canonical_name="Star Trek", type="Franchise"), False),
        (Entity(canonical_name="Harry Potter", type="Franchise"), ExistingNode(node_id="e5", canonical_name="Lord of the Rings", type="Franchise"), False),
        (Entity(canonical_name="Spotify", type="Company"), ExistingNode(node_id="e6", canonical_name="Apple Music", type="Software"), False),
        (Entity(canonical_name="HBO", type="Company"), ExistingNode(node_id="e7", canonical_name="Home Box Office", type="Company"), True),
        (Entity(canonical_name="Nintendo", type="Company"), ExistingNode(node_id="e8", canonical_name="Sega", type="Company"), False),
        (Entity(canonical_name="PlayStation", type="Product"), ExistingNode(node_id="e9", canonical_name="Sony PS5", type="Product"), False),
        (Entity(canonical_name="Xbox", type="Product"), ExistingNode(node_id="e10", canonical_name="Microsoft Xbox Series X", type="Product"), False),
    ]
    for i in range(11, 51):
        if i % 2 == 1:
            ent_media.append((
                Entity(canonical_name=f"Media {i}", type="Company"),
                ExistingNode(node_id=f"e{i}", canonical_name=f"Media {i} Studio", type="Company"),
                True
            ))
        else:
            ent_media.append((
                Entity(canonical_name=f"Media {i}", type="Company"),
                ExistingNode(node_id=f"e{i}", canonical_name=f"DiffMedia {i}", type="Company"),
                False
            ))

    # 9. Sports & Events (50 cases)
    sports = [
        (Entity(canonical_name="Real Madrid", type="SportsTeam"), ExistingNode(node_id="s1", canonical_name="Real Madrid C.F.", type="SportsTeam"), True),
        (Entity(canonical_name="FC Barcelona", type="SportsTeam"), ExistingNode(node_id="s2", canonical_name="Barça", type="SportsTeam"), True),
        (Entity(canonical_name="Manchester United", type="SportsTeam"), ExistingNode(node_id="s3", canonical_name="Manchester City", type="SportsTeam"), False),
        (Entity(canonical_name="Lakers", type="SportsTeam"), ExistingNode(node_id="s4", canonical_name="Los Angeles Lakers", type="SportsTeam"), True),
        (Entity(canonical_name="Warriors", type="SportsTeam"), ExistingNode(node_id="s5", canonical_name="Golden State Warriors", type="SportsTeam"), True),
        (Entity(canonical_name="NBA", type="Organization"), ExistingNode(node_id="s6", canonical_name="National Basketball Association", type="Organization"), True),
        (Entity(canonical_name="FIFA World Cup", type="Event"), ExistingNode(node_id="s7", canonical_name="UEFA Champions League", type="Event"), False),
        (Entity(canonical_name="Olympic Games", type="Event"), ExistingNode(node_id="s8", canonical_name="Olympics", type="Event"), True),
        (Entity(canonical_name="Super Bowl", type="Event"), ExistingNode(node_id="s9", canonical_name="NFL Championship", type="Event"), True),
        (Entity(canonical_name="Formula 1", type="Event"), ExistingNode(node_id="s10", canonical_name="F1 Grand Prix", type="Event"), True),
    ]
    for i in range(11, 51):
        if i % 2 == 1:
            sports.append((
                Entity(canonical_name=f"Team {i}", type="SportsTeam"),
                ExistingNode(node_id=f"s{i}", canonical_name=f"Team {i} FC", type="SportsTeam"),
                True
            ))
        else:
            sports.append((
                Entity(canonical_name=f"Team {i}", type="SportsTeam"),
                ExistingNode(node_id=f"s{i}", canonical_name=f"RivalTeam {i}", type="SportsTeam"),
                False
            ))

    # 10. Institutions (50 cases)
    inst = [
        (Entity(canonical_name="UN", type="Organization"), ExistingNode(node_id="i1", canonical_name="United Nations", type="Organization"), True),
        (Entity(canonical_name="WHO", type="Organization"), ExistingNode(node_id="i2", canonical_name="World Health Organization", type="Organization"), True),
        (Entity(canonical_name="UNESCO", type="Organization"), ExistingNode(node_id="i3", canonical_name="UNICEF", type="Organization"), False),
        (Entity(canonical_name="FBI", type="Organization"), ExistingNode(node_id="i4", canonical_name="Federal Bureau of Investigation", type="Organization"), True),
        (Entity(canonical_name="CIA", type="Organization"), ExistingNode(node_id="i5", canonical_name="Central Intelligence Agency", type="Organization"), True),
        (Entity(canonical_name="MIT", type="University"), ExistingNode(node_id="i6", canonical_name="Massachusetts Institute of Technology", type="University"), True),
        (Entity(canonical_name="Stanford", type="University"), ExistingNode(node_id="i7", canonical_name="Stanford University", type="University"), True),
        (Entity(canonical_name="Harvard", type="University"), ExistingNode(node_id="i8", canonical_name="Harvard University", type="University"), True),
        (Entity(canonical_name="Red Cross", type="Organization"), ExistingNode(node_id="i9", canonical_name="International Red Cross", type="Organization"), True),
        (Entity(canonical_name="Amnesty International", type="Organization"), ExistingNode(node_id="i10", canonical_name="Greenpeace", type="Organization"), False),
    ]
    for i in range(11, 51):
        if i % 2 == 1:
            inst.append((
                Entity(canonical_name=f"Univ {i}", type="University"),
                ExistingNode(node_id=f"i{i}", canonical_name=f"University of {i}", type="University"),
                True
            ))
        else:
            inst.append((
                Entity(canonical_name=f"Univ {i}", type="University"),
                ExistingNode(node_id=f"i{i}", canonical_name=f"OtherCollege {i}", type="University"),
                False
            ))

    # 11. Law & Legal (50 cases)
    law = [
        (Entity(canonical_name="Latham & Watkins", type="LawFirm"), ExistingNode(node_id="l1", canonical_name="Latham & Watkins LLP", type="LawFirm"), True),
        (Entity(canonical_name="Kirkland & Ellis", type="LawFirm"), ExistingNode(node_id="l2", canonical_name="Kirkland & Ellis LLP", type="LawFirm"), True),
        (Entity(canonical_name="Baker McKenzie", type="LawFirm"), ExistingNode(node_id="l3", canonical_name="Baker & McKenzie LLP", type="LawFirm"), True),
        (Entity(canonical_name="Skadden", type="LawFirm"), ExistingNode(node_id="l4", canonical_name="Skadden, Arps, Slate, Meagher & Flom", type="LawFirm"), True),
        (Entity(canonical_name="Clifford Chance", type="LawFirm"), ExistingNode(node_id="l5", canonical_name="Linklaters", type="LawFirm"), False),
        (Entity(canonical_name="DOJ", type="LegalInstitution"), ExistingNode(node_id="l6", canonical_name="Department of Justice", type="LegalInstitution"), True),
        (Entity(canonical_name="ICJ", type="LegalInstitution"), ExistingNode(node_id="l7", canonical_name="International Court of Justice", type="LegalInstitution"), True),
        (Entity(canonical_name="Supreme Court", type="LegalInstitution"), ExistingNode(node_id="l8", canonical_name="SCOTUS", type="LegalInstitution"), True),
        (Entity(canonical_name="ACLU", type="LegalInstitution"), ExistingNode(node_id="l9", canonical_name="American Civil Liberties Union", type="LegalInstitution"), True),
        (Entity(canonical_name="GDPR", type="LegalConcept"), ExistingNode(node_id="l10", canonical_name="General Data Protection Regulation", type="LegalConcept"), True),
        (Entity(canonical_name="NDA", type="LegalConcept"), ExistingNode(node_id="l11", canonical_name="Non-Disclosure Agreement", type="LegalConcept"), True),
        (Entity(canonical_name="IP", type="LegalConcept"), ExistingNode(node_id="l12", canonical_name="Intellectual Property", type="LegalConcept"), True),
        (Entity(canonical_name="Attorney General", type="LegalRole"), ExistingNode(node_id="l13", canonical_name="AG", type="LegalRole"), True),
        (Entity(canonical_name="Plaintiff", type="LegalRole"), ExistingNode(node_id="l14", canonical_name="Defendant", type="LegalRole"), False),
        (Entity(canonical_name="Solicitor", type="LegalRole"), ExistingNode(node_id="l15", canonical_name="Barrister", type="LegalRole"), False),
    ]
    for i in range(16, 51):
        if i % 2 == 1:
            law.append((
                Entity(canonical_name=f"LawFirm {i}", type="LawFirm"),
                ExistingNode(node_id=f"l{i}", canonical_name=f"LawFirm {i} LLP", type="LawFirm"),
                True
            ))
        else:
            law.append((
                Entity(canonical_name=f"LawFirm {i}", type="LawFirm"),
                ExistingNode(node_id=f"l{i}", canonical_name=f"OtherLegal {i}", type="LawFirm"),
                False
            ))

    return {
        "Tech Companies": tech,
        "Products & Models": products,
        "People & Leaders": people,
        "Homonyms": homonyms,
        "Geography": geo,
        "Automotive": auto,
        "Finance & Crypto": fin,
        "Entertainment": ent_media,
        "Sports & Events": sports,
        "Institutions": inst,
        "Law & Legal": law,
    }


def build_tricky_dataset() -> list[Tuple[str, Entity, ExistingNode, bool]]:
    """Builds an interleaved round-robin dataset of 550 test cases across 11 domains (50 cases each)."""
    domain_data = generate_domain_datasets()
    domain_names = list(domain_data.keys())

    interleaved_dataset = []
    for cycle_idx in range(50):
        for dom_name in domain_names:
            ent, node, expected = domain_data[dom_name][cycle_idx]
            interleaved_dataset.append((dom_name, ent, node, expected))

    return interleaved_dataset


def verify_decision(
    entity_a_name: str,
    entity_b_name: str,
    autograft_decision: bool,
    expected_match: bool,
    judge_model: str = JUDGE_MODEL,
) -> Tuple[bool, str]:
    """Uses a Judge LLM to verify if AutoGraft's entity resolution decision is correct."""
    if autograft_decision == expected_match:
        return True, "✅ CORRECT"

    prompt = (
        f"An AI system decided if Entity A ('{entity_a_name}') and Entity B ('{entity_b_name}') "
        f"are the same real-world entity.\n"
        f"The AI system decided: {'MATCH (True)' if autograft_decision else 'NO MATCH (False)'}.\n"
        "Is this decision strictly correct based on real-world knowledge?\n"
        "Reply STRICTLY with the word 'YES' (correct) or 'NO' (incorrect)."
    )
    for attempt in range(4):
        try:
            response = litellm.completion(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = str(response.choices[0].message.content).strip().upper()
            if "YES" in content:
                return True, "✅ CORRECT"
            return False, "❌ INCORRECT"
        except Exception as err:
            err_name = type(err).__name__
            if "RateLimit" in err_name or "429" in str(err):
                time.sleep(1.5 * (attempt + 1))
                continue
            return False, f"⚠️ JUDGE API ERROR ({err_name}: {err})"

    return False, "⚠️ JUDGE API ERROR (Rate limit exceeded after retries)"


def generate_accuracy_chart(
    domain_scores: dict[str, float], overall_accuracy: float
) -> None:
    """Generates an accuracy bar chart by domain and saves to benchmark/assets/accuracy_by_domain.png."""
    os.makedirs("benchmark/assets", exist_ok=True)

    domains = list(domain_scores.keys()) + ["OVERALL"]
    scores = list(domain_scores.values()) + [overall_accuracy]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        "AutoGraft Entity Resolution Accuracy Audit by Domain (550 Interleaved Cases)",
        fontsize=14,
        fontweight="bold",
    )

    colors = ["#3B82F6"] * len(domain_scores) + ["#10B981"]
    bars = ax.bar(domains, scores, color=colors, width=0.55, edgecolor="none")

    ax.set_ylim(0, 115)
    ax.set_ylabel("Audit Accuracy (%)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=30, ha="right", fontsize=9.5, fontweight="bold")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.tight_layout()
    chart_path = "benchmark/assets/accuracy_by_domain.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nAccuracy chart successfully saved to '{chart_path}'")


def run_accuracy_benchmark() -> None:
    """Runs the accuracy benchmark loop with an animated terminal progress bar."""
    dataset = build_tricky_dataset()
    correct_decisions = 0
    total_cases = len(dataset)

    domain_counts: dict[str, int] = {}
    domain_correct: dict[str, int] = {}

    print("=" * 95)
    print(
        f" 🎯 AUTOGRAFT ACCURACY BENCHMARK (LLM-AS-A-JUDGE AUDIT - {total_cases} INTERLEAVED CASES / 11 DOMAINS)"
    )
    print("=" * 95)
    print(f"AutoGraft Model : {AUTOGRAFT_MODEL}")
    print(f"Judge LLM Model : {JUDGE_MODEL}\n")

    for idx, (domain, new_entity, existing_node, expected) in enumerate(dataset, 1):
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Run AutoGraft arbitration
        result = arbitrate_match(new_entity, existing_node, model=AUTOGRAFT_MODEL)
        decision = result.is_match

        # Judge verification
        is_correct, verdict_str = verify_decision(
            new_entity.canonical_name,
            existing_node.canonical_name,
            decision,
            expected_match=expected,
            judge_model=JUDGE_MODEL,
        )

        if is_correct:
            correct_decisions += 1
            domain_correct[domain] = domain_correct.get(domain, 0) + 1

        # Update animated progress bar
        print_progress(idx, total_cases, prefix="🎯 Auditing Cases")

    accuracy_pct = (correct_decisions / total_cases) * 100.0

    domain_scores = {
        dom: (domain_correct.get(dom, 0) / domain_counts[dom]) * 100.0
        for dom in domain_counts
    }

    generate_accuracy_chart(domain_scores, accuracy_pct)

    print("\n" + "-" * 95)
    print(
        f"🏆 Final Audit Score : {correct_decisions}/{total_cases} ({accuracy_pct:.1f}% Accuracy)"
    )
    print("=" * 95 + "\n")


if __name__ == "__main__":
    run_accuracy_benchmark()
