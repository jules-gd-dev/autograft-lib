"""Documents dataset generator for 1000 multi-industry massive benchmark documents."""
from langchain_core.documents import Document


def _generate_massive_texts() -> dict[str, list[str]]:
    domains = {
        "Legal": [
            "PATENT DISPUTE: Apple Inc. sued for patent infringement regarding iOS.",
            "AGRICULTURAL CONTRACT: A massive shipment of fresh Apple fruit was inspected.",
            "IMMIGRATION LAW: The client's H1B Visa was approved by the embassy.",
            "FINANCIAL REGULATION: Visa Inc. announced new credit card fees.",
            "LITIGATION BRIEF: The state of Washington filed a lawsuit against the EPA.",
            "DEFAMATION SUIT: George Washington's descendants filed a historical claim.",
            "LABOR DISPUTE: Target Corporation faces a class-action lawsuit from cashiers.",
            "HUNTING REGULATION: The legal target for deer hunting was reduced this year.",
            "FOOD SAFETY: Subway franchises were investigated for sandwich ingredients.",
            "TRANSPORT LAW: The New York Subway system received federal funding."
        ],
        "Tech": [
            "SOFTWARE ARCHITECTURE: The backend is entirely written in Java.",
            "GEOGRAPHY DATA: We mapped the population density of Java, Indonesia.",
            "MACHINE LEARNING: We used Python for the AI training script.",
            "ZOOLOGY DATABASE: The zoo registered a new Reticulated Python species.",
            "CLOUD DEPLOYMENT: Amazon Web Services (AWS) launched new clusters.",
            "ECOLOGY REPORT: Deforestation in the Amazon rainforest accelerated.",
            "TELECOM INFRASTRUCTURE: Orange S.A. deployed 5G networks in Paris.",
            "UI DESIGN: The primary button color should be a bright Orange.",
            "HARDWARE LOGISTICS: Microsoft Surface tablets were shipped.",
            "CLEANING SERVICE: The surface of the table must be disinfected."
        ],
        "Insurance": [
            "AGRICULTURAL POLICY: Insuring a farm producing avocado and apple trees.",
            "LIABILITY CLAIM: The avocat (lawyer) filed a claim for malpractice.",
            "TRAVEL POLICY: The tourist lost their Visa and passport in Rome.",
            "CORPORATE LIABILITY: Visa Inc. executives insured for D&O.",
            "NATURAL DISASTER: The Amazon basin suffered extreme flooding.",
            "COMMERCIAL REAL ESTATE: Amazon fulfillment center damaged by fire.",
            "ANIMAL BITE: A tourist was bitten by a Python snake.",
            "CYBER INSURANCE: A Python script caused a major data breach.",
            "PUBLIC TRANSIT: The city subway flooded, triggering the policy.",
            "FRANCHISE INSURANCE: A Subway restaurant caught fire."
        ],
        "Finance": [
            "EARNINGS CALL: Apple reported record iPhone sales this quarter.",
            "COMMODITY MARKET: Apple juice futures dropped by 10%.",
            "CREDIT RISK: Visa reported higher default rates on consumer cards.",
            "GOVERNMENT SPENDING: Visa processing fees for immigrants increased.",
            "TECH STOCKS: Amazon shares surged after the earnings report.",
            "ESG INVESTMENT: A fund dedicated to protecting the Amazon rainforest.",
            "TELECOM SECTOR: Orange stock plummeted after the CEO resigned.",
            "AGRICULTURAL FUTURES: Frozen orange juice concentrate is volatile.",
            "MERGERS: Java coffee shops acquired by a larger conglomerate.",
            "VENTURE CAPITAL: A startup building Java tools raised $5M."
        ],
        "Healthcare": [
            "NUTRITION STUDY: Eating an apple a day reduces cardiovascular risk.",
            "WEARABLES: The Apple Watch can now detect atrial fibrillation.",
            "EPIDEMIOLOGY: Malaria outbreak reported in the Amazon region.",
            "WORKPLACE SAFETY: Amazon warehouse workers report high injury rates.",
            "TROPICAL MEDICINE: Treating snake bites from a Python.",
            "BIOINFORMATICS: A new Python library for DNA sequencing.",
            "DIETARY GUIDELINES: Orange juice consumption is linked to sugar spikes.",
            "TELEMEDICINE: Orange Healthcare launched a remote monitoring app.",
            "PUBLIC HEALTH: The state of Washington mandated vaccines.",
            "MEDICAL HISTORY: George Washington died from epiglottitis."
        ],
        "Manufacturing": [
            "FOOD PROCESSING: Automated sorting of apple and pear harvests.",
            "ELECTRONICS: Foxconn assembly lines for the new Apple iPhone.",
            "LUMBER PRODUCTION: Sustainable timber sourcing from the Amazon.",
            "PACKAGING: Cardboard boxes for Amazon Prime deliveries.",
            "TEXTILES: Manufacturing synthetic Python skin for fashion.",
            "SOFTWARE TOOLS: Using Python to control robotic assembly arms.",
            "BEVERAGE BOTTLING: Production line for Orange soda.",
            "TELECOM EQUIPMENT: Manufacturing routers for Orange S.A.",
            "TRANSIT MANUFACTURING: Building new subway cars for the MTA.",
            "FOOD PREP: Stainless steel counters for Subway sandwich shops."
        ],
        "Retail": [
            "GROCERY: Organic apple sales increased by 20%.",
            "ELECTRONICS: The new Apple store opened on 5th Avenue.",
            "E-COMMERCE: Amazon Prime Day broke all sales records.",
            "BOOKSTORE: Selling books about the Amazon rainforest.",
            "FASHION: Faux Python leather boots are trending.",
            "IT HARDWARE: Retail software rewritten in Python.",
            "FRUIT STAND: Valencia orange shipments arrived late.",
            "MOBILE PHONES: Orange stores offered discounts on data plans.",
            "BIG BOX RETAIL: Target announced Black Friday deals.",
            "ARCHERY SHOP: Foam target sales doubled."
        ],
        "Energy": [
            "BIOFUELS: Using apple cider vinegar byproducts for biomass.",
            "DATA CENTERS: Apple Inc. committed to 100% renewable energy.",
            "HYDROELECTRIC: New dams proposed in the Amazon basin.",
            "FLEET ELECTRIFICATION: Amazon ordered 100,000 Rivian EV vans.",
            "GEOTHERMAL: Exploring thermal vents in Washington state.",
            "BIOPIC: George Washington's views on early coal usage.",
            "SOLAR PANELS: The target efficiency is now 25%.",
            "CORPORATE PPA: Target stores installed rooftop solar.",
            "OFFSHORE WIND: Orange cable networks powered by wind.",
            "BIOENERGY: Fermenting orange peels for ethanol."
        ],
        "Education": [
            "NUTRITION CLASS: Students learned about the vitamins in an apple.",
            "IT DEPARTMENT: Providing Apple iPads to all students.",
            "GEOGRAPHY: Studying the ecosystem of the Amazon rainforest.",
            "BUSINESS SCHOOL: Case study on Amazon's logistics network.",
            "COMPUTER SCIENCE: Introduction to Python programming.",
            "BIOLOGY: Dissecting a Python snake in lab.",
            "HISTORY: The presidency of George Washington.",
            "STATE HISTORY: The founding of Washington university.",
            "URBAN PLANNING: Designing efficient subway systems.",
            "CULINARY ARTS: Making sandwiches at Subway."
        ],
        "Real Estate": [
            "ORCHARDS: 50 acres of apple orchards sold in Oregon.",
            "CORPORATE CAMPUS: Apple Park in Cupertino is valued at $5B.",
            "WAREHOUSING: Amazon leased 1 million sq ft of industrial space.",
            "CONSERVATION: Buying land in the Amazon to prevent logging.",
            "TECH OFFICES: Python Software Foundation leased new office space.",
            "ZOOLOGICAL PARK: Building a new enclosure for the Python.",
            "RETAIL LEASE: Target anchored the new shopping mall.",
            "SHOOTING RANGE: The target practice facility was rezoned.",
            "TELECOM SITES: Orange S.A. leased cell tower space.",
            "CITRUS GROVES: Orange farm sold for residential development."
        ]
    }
    
    # 100 documents per domain
    result = {}
    for domain, base_texts in domains.items():
        # Repeat the 10 base texts 10 times with a unique ID to make 100
        result[domain] = [base_texts[i % 10] + f" [Doc #{i+1}]" for i in range(100)]
        
    return result

def get_massive_documents() -> list[Document]:
    """Returns 1000 Document objects across 10 industries (100 per industry)."""
    data = _generate_massive_texts()
    docs = []
    doc_id = 1
    for industry, texts in data.items():
        for text in texts:
            docs.append(Document(page_content=text, metadata={"doc_id": doc_id, "industry": industry}))
            doc_id += 1
    return docs
