"""Real-world entity catalog with ground-truth canonical identities.

Each tuple: (canonical_id, type, [name_variants]). Entities sharing a
canonical_id MUST merge; homonyms share a surface name but differ in type
and canonical_id, exercising the resolver's type isolation.
"""

from dataclasses import dataclass, field


@dataclass
class EntityMention:
    """A single entity mention inside a document, with ground-truth identity."""

    name: str
    type: str
    canonical_id: str


@dataclass
class BenchmarkDoc:
    """A document in the benchmark corpus."""

    doc_id: str
    industry: str
    text: str
    entities: list[EntityMention] = field(default_factory=list)


CATALOG: list[tuple[str, str, list[str]]] = [
    # Tech
    ("tech_msft", "Company", ["Microsoft", "Microsoft Corp", "Microsoft Corporation", "MSFT"]),
    ("tech_aapl", "Company", ["Apple", "Apple Inc", "AAPL"]),
    ("tech_googl", "Company", ["Google", "Alphabet", "Alphabet Inc", "GOOGL"]),
    ("tech_amzn", "Company", ["Amazon", "Amazon.com", "AMZN"]),
    ("tech_meta", "Company", ["Meta", "Meta Platforms", "Facebook"]),
    ("tech_nvda", "Company", ["NVIDIA", "Nvidia Corp", "NVDA"]),
    ("tech_tsla", "Company", ["Tesla", "Tesla Inc", "Tesla Motors", "TSLA"]),
    ("tech_openai", "Company", ["OpenAI", "OpenAI LP"]),
    ("tech_anthropic", "Company", ["Anthropic"]),
    ("tech_ibm", "Company", ["IBM", "International Business Machines"]),
    # Finance
    ("fin_jpm", "Company", ["JPMorgan", "J.P. Morgan Chase", "JPMorgan Chase", "JPM"]),
    ("fin_gs", "Company", ["Goldman Sachs", "Goldman Sachs Group", "GS"]),
    ("fin_bac", "Company", ["Bank of America", "BofA", "BAC"]),
    ("fin_brk", "Company", ["Berkshire Hathaway", "BRK", "BRK.A"]),
    ("fin_visa", "Company", ["Visa", "Visa Inc"]),
    ("fin_ma", "Company", ["Mastercard", "Mastercard Inc", "MA"]),
    ("fin_blackrock", "Company", ["BlackRock"]),
    ("fin_ms", "Company", ["Morgan Stanley", "MS"]),
    ("fin_citi", "Company", ["Citigroup", "Citi", "Citibank"]),
    # Healthcare
    ("hc_pfizer", "Company", ["Pfizer", "Pfizer Inc"]),
    ("hc_jnj", "Company", ["Johnson & Johnson", "J&J", "JNJ"]),
    ("hc_moderna", "Company", ["Moderna"]),
    ("hc_azn", "Company", ["AstraZeneca"]),
    ("hc_who", "Organization", ["WHO", "World Health Organization"]),
    ("hc_cdc", "Organization", ["CDC", "Centers for Disease Control"]),
    ("hc_fda", "Organization", ["FDA", "Food and Drug Administration"]),
    # Legal / Institutions
    ("legal_scotus", "Organization", ["Supreme Court", "SCOTUS", "U.S. Supreme Court"]),
    ("legal_doj", "Organization", ["DOJ", "Department of Justice"]),
    ("legal_sec", "Organization", ["SEC", "Securities and Exchange Commission"]),
    ("legal_eu", "Organization", ["EU", "European Union"]),
    ("legal_icc", "Organization", ["ICC", "International Criminal Court"]),
    # Education
    ("edu_harvard", "Organization", ["Harvard", "Harvard University"]),
    ("edu_mit", "Organization", ["MIT", "Massachusetts Institute of Technology"]),
    ("edu_stanford", "Organization", ["Stanford", "Stanford University"]),
    ("edu_oxford", "Organization", ["Oxford", "University of Oxford"]),
    # Energy
    ("en_xom", "Company", ["ExxonMobil", "Exxon Mobil"]),
    ("en_shell", "Company", ["Shell", "Royal Dutch Shell"]),
    ("en_chevron", "Company", ["Chevron", "Chevron Corp"]),
    ("en_bp", "Company", ["BP", "British Petroleum"]),
    ("en_aramco", "Company", ["Saudi Aramco", "Aramco"]),
    # Retail
    ("ret_wmt", "Company", ["Walmart", "Wal-Mart"]),
    ("ret_cost", "Company", ["Costco", "Costco Wholesale"]),
    ("ret_target", "Company", ["Target", "Target Corp"]),
    ("ret_nike", "Company", ["Nike", "Nike Inc"]),
    # Manufacturing
    ("mfg_toyota", "Company", ["Toyota", "Toyota Motor"]),
    ("mfg_ford", "Company", ["Ford", "Ford Motor"]),
    ("mfg_boeing", "Company", ["Boeing"]),
    ("mfg_ge", "Company", ["GE", "General Electric"]),
    ("mfg_siemens", "Company", ["Siemens"]),
    # Insurance
    ("ins_aig", "Company", ["AIG", "American International Group"]),
    ("ins_axa", "Company", ["AXA"]),
    ("ins_allianz", "Company", ["Allianz"]),
    # Real Estate
    ("re_blackstone", "Company", ["Blackstone"]),
    ("re_cbre", "Company", ["CBRE", "CBRE Group"]),
    ("re_zillow", "Company", ["Zillow", "Zillow Group"]),
    # --- Homonym traps: same surface name, different type + identity ---
    ("apple_fruit", "Product", ["Apple"]),
    ("amazon_river", "Location", ["Amazon", "Amazon River"]),
    ("visa_document", "Concept", ["Visa"]),
    ("washington_person", "Person", ["Washington", "George Washington"]),
    ("washington_place", "Location", ["Washington", "Washington D.C."]),
    ("orange_fruit", "Product", ["Orange"]),
    ("python_animal", "Animal", ["Python"]),
    ("python_lang", "Technology", ["Python", "Python language"]),
]

INDUSTRIES: dict[str, list[str]] = {
    "Tech": ["tech_msft", "tech_aapl", "tech_googl", "tech_amzn", "tech_meta",
             "tech_nvda", "tech_tsla", "tech_openai", "tech_anthropic", "tech_ibm"],
    "Finance": ["fin_jpm", "fin_gs", "fin_bac", "fin_brk", "fin_visa",
                "fin_ma", "fin_blackrock", "fin_ms", "fin_citi"],
    "Healthcare": ["hc_pfizer", "hc_jnj", "hc_moderna", "hc_azn", "hc_who",
                   "hc_cdc", "hc_fda"],
    "Legal": ["legal_scotus", "legal_doj", "legal_sec", "legal_eu", "legal_icc"],
    "Education": ["edu_harvard", "edu_mit", "edu_stanford", "edu_oxford"],
    "Energy": ["en_xom", "en_shell", "en_chevron", "en_bp", "en_aramco"],
    "Retail": ["ret_wmt", "ret_cost", "ret_target", "ret_nike"],
    "Manufacturing": ["mfg_toyota", "mfg_ford", "mfg_boeing", "mfg_ge", "mfg_siemens"],
    "Insurance": ["ins_aig", "ins_axa", "ins_allianz"],
    "RealEstate": ["re_blackstone", "re_cbre", "re_zillow"],
}

# Cross-industry homonym pool injected occasionally to stress false merges.
HOMONYMS: list[str] = [
    "apple_fruit", "amazon_river", "visa_document", "washington_person",
    "washington_place", "orange_fruit", "python_animal", "python_lang",
]

VARIANT_MAP: dict[str, tuple[str, str, list[str]]] = {c[0]: c for c in CATALOG}
