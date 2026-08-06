"""Documents dataset generator for 200 multi-industry macro benchmark documents."""
from langchain_core.documents import Document


def _generate_industry_texts() -> dict[str, list[str]]:
    """Returns 50 realistic texts per industry (200 total) with complex acronyms & homonyms."""
    legal_base = [
        "ASSET PURCHASE AGREEMENT: Alphabet Inc. (Google) acquires TechCorp assets. Latham & Watkins LLP acts as counsel, while Kirkland & Ellis LLP represents TechCorp before the DOJ.",
        "MASTER SERVICES AGREEMENT: Apple Inc. contracts with CyberDyne Systems under California Law and General Data Protection Regulation (GDPR) compliance.",
        "MUTUAL NON-DISCLOSURE AGREEMENT: OpenAI and BioSynth Global execute an NDA regarding AI model architecture supervised by Federal Trade Commission (FTC).",
        "PATENT LICENSING DEED: Microsoft Corporation grants patent license to Apex Legal Group under WIPO and EU Patent Law.",
        "LITIGATION BRIEF: Supreme Court of the United States (SCOTUS) reviewed ACLU vs FBI surveillance protocols.",
        "DATA PROTECTION ADDENDUM: Meta Platforms Inc. submitted DPA to European Data Protection Board (EDPB) ensuring GDPR compliance.",
        "DEFENSE CONTRACT: Lockheed Martin and SpaceX secured satellite launch contract supervised by Skadden Arps Slate Meagher & Flom.",
        "CUSTODY AGREEMENT: J.P. Morgan Chase & Co. and Goldman Sachs Group finalized digital asset custody protocols.",
        "ANTITRUST CONSENT DECREE: Department of Justice (DOJ) and FTC filed enforcement decree involving Amazon.com Inc. in federal court.",
        "RESTRUCTURING DEED: Skadden filed corporate reorganization plan for Apex Holdings before Delaware Bankruptcy Court.",
    ]
    tech_base = [
        "CLOUD DEPLOYMENT: Amazon Web Services (AWS) launched new Kubernetes (K8s) clusters for enterprise Docker containerized workloads.",
        "SOFTWARE ARCHITECTURE: Microsoft Corporation integrated Copilot AI into Azure Kubernetes Service (AKS) for CI/CD pipelines.",
        "VECTOR SEARCH SPECIFICATION: Elastic N.V. (Elasticsearch) benchmarked vector database indexing performance for RAG workloads.",
        "AI MODEL RELEASE: OpenAI and Anthropic announced safety evaluations for Claude 3.5 Sonnet and GPT-4o LLM architectures.",
        "SDK DOCUMENTATION: Google LLC (Alphabet) updated Android SDK enabling native REST API endpoint routing.",
        "ENTERPRISE SECURITY: Meta Platforms Inc. implemented Role-Based Access Control (RBAC) across IAM identity infrastructure.",
        "MICROSERVICES BENCHMARK: Docker Inc. released Desktop update optimizing Linux container memory allocation.",
        "DATA PIPELINE ENGINEERING: Databricks partnered with Snowflake Inc. for multi-cloud data warehouse interoperability.",
        "OPEN SOURCE LLM RELEASE: Mistral AI launched new open weights model in Paris France for PyTorch framework.",
        "DEVOPS INFRASTRUCTURE: Cloudflare expanded content delivery network nodes across Tokyo Metropolis for edge computing.",
    ]
    ins_base = [
        "UNDERWRITING POLICY: Allianz SE issued Directors and Officers Liability (D&O) coverage for GlobalCorp enterprise board.",
        "CLAIMS ASSESSMENT: AXA SA processed Errors and Omissions Insurance (E&O) claim submitted by Apex Tech Solutions.",
        "COMMERCIAL REINSURANCE: Berkshire Hathaway Specialty Insurance (BHSI) underwrote Commercial General Liability (CGL) coverage.",
        "PROPERTY RISK AUDIT: Ping An Insurance evaluated Property and Casualty Insurance (P&C) exposure for commercial real estate.",
        "THIRD PARTY CLAIMS: Prudential Financial retained Third Party Administrator (TPA) for disability benefits administration.",
        "WORKERS COMPENSATION AUDIT: Chubb Limited audited Allocated Loss Adjustment Expenses (ALAE) for manufacturing plant.",
        "REPLACEMENT COST VALUATION: Travelers Companies calculated Actual Cash Value (ACV) and Replacement Cost Value (RCV) for commercial loss.",
        "AUTO LIABILITY CLAIM: Progressive Casualty Insurance reviewed Uninsured Motorist (UM/UIM) coverage limits.",
        "CERTIFICATE OF INSURANCE: Liberty Mutual issued Certificate of Insurance (COI) confirming general liability limits.",
        "CYBER RISK POLICY: AIG underwrote specialized ransomware protection policy for healthcare provider network.",
    ]
    fin_base = [
        "FINANCIAL EARNINGS REPORT: JP Morgan Chase & Co. (JPM) reported record quarterly EBITDA earnings on Wall Street.",
        "M&A INVESTMENT MEMO: Goldman Sachs Group (GS) advised BlackRock Inc. on $5B renewable energy infrastructure acquisition.",
        "CREDIT RISK ASSESSMENT: Morgan Stanley reviewed market exposure to Bitcoin (BTC) and Ethereum (ETH) crypto assets.",
        "EUROPEAN BANKING AUDIT: Société Générale (SocGen) and BNP Paribas completed ECB stress test capital compliance.",
        "REGULATORY FILING: Securities and Exchange Commission (SEC) enforced Know Your Customer and Anti-Money Laundering (KYC/AML) rules.",
        "BENCHMARK RATE TRANSITION: FINRA published guidance transitioning corporate loans from LIBOR to SOFR reference rate.",
        "PRIVATE EQUITY BUYOUT: KKR & Co. finalized Leveraged Buyout (LBO) of medical devices manufacturer.",
        "INITIAL PUBLIC OFFERING: Credit Suisse structured IPO registration statement for fintech startup listing.",
        "ASSET MANAGEMENT AUDIT: Vanguard Group reported Assets Under Management (AUM) growth across index fund portfolio.",
        "VALUATION MODELING: Lazard calculated Discounted Cash Flow (DCF) and Compound Annual Growth Rate (CAGR) projections.",
    ]

    return {
        "Legal": [legal_base[i % 10] + f" [Case #{i+1}]" for i in range(50)],
        "Tech": [tech_base[i % 10] + f" [Spec #{i+1}]" for i in range(50)],
        "Insurance": [ins_base[i % 10] + f" [Claim #{i+1}]" for i in range(50)],
        "Finance": [fin_base[i % 10] + f" [Filing #{i+1}]" for i in range(50)],
    }


def get_macro_documents() -> list[Document]:
    """Returns 200 Document objects across 4 industries (50 Legal, 50 Tech, 50 Insurance, 50 Finance)."""
    data = _generate_industry_texts()
    docs = []
    doc_id = 1
    for industry, texts in data.items():
        for text in texts:
            docs.append(Document(page_content=text, metadata={"doc_id": doc_id, "industry": industry}))
            doc_id += 1
    return docs
