"""Legal documents dataset and existing Knowledge Graph nodes for the Legal Team RAG benchmark."""
from langchain_core.documents import Document

from autograft.models.entities import ExistingNode


def build_legal_existing_nodes() -> list[ExistingNode]:
    """Returns existing legal Knowledge Graph nodes in Neo4j."""
    return [
        ExistingNode(node_id="l1", canonical_name="Latham & Watkins LLP", type="LawFirm", aliases=["Latham & Watkins", "Latham"]),
        ExistingNode(node_id="l2", canonical_name="Kirkland & Ellis LLP", type="LawFirm", aliases=["Kirkland & Ellis", "Kirkland"]),
        ExistingNode(node_id="l3", canonical_name="Skadden, Arps, Slate, Meagher & Flom LLP", type="LawFirm", aliases=["Skadden", "Skadden Arps"]),
        ExistingNode(node_id="l4", canonical_name="Department of Justice", type="LegalInstitution", aliases=["DOJ", "U.S. Department of Justice"]),
        ExistingNode(node_id="l5", canonical_name="Federal Trade Commission", type="LegalInstitution", aliases=["FTC"]),
        ExistingNode(node_id="l6", canonical_name="Supreme Court of the United States", type="LegalInstitution", aliases=["Supreme Court", "SCOTUS"]),
        ExistingNode(node_id="l7", canonical_name="General Data Protection Regulation", type="LegalConcept", aliases=["GDPR"]),
        ExistingNode(node_id="l8", canonical_name="Alphabet Inc.", type="Company", aliases=["Google", "Alphabet"]),
        ExistingNode(node_id="l9", canonical_name="Apple Inc.", type="Company", aliases=["Apple"]),
        ExistingNode(node_id="l10", canonical_name="Microsoft Corporation", type="Company", aliases=["Microsoft"]),
        ExistingNode(node_id="l11", canonical_name="Meta Platforms Inc.", type="Company", aliases=["Meta", "Facebook"]),
        ExistingNode(node_id="l12", canonical_name="Amazon.com Inc.", type="Company", aliases=["Amazon"]),
        ExistingNode(node_id="l13", canonical_name="OpenAI", type="Company"),
        ExistingNode(node_id="l14", canonical_name="SpaceX", type="Company", aliases=["Space Exploration Technologies"]),
        ExistingNode(node_id="l15", canonical_name="JP Morgan Chase & Co.", type="Company", aliases=["J.P. Morgan"]),
    ]


def get_legal_documents() -> list[Document]:
    """Returns 10 realistic legal documents for RAG processing."""
    texts = [
        "ASSET PURCHASE AGREEMENT: Alphabet Inc. (Google) agrees to acquire technology assets from TechCorp. Latham & Watkins LLP serves as legal counsel for Alphabet Inc., while Kirkland & Ellis LLP represents TechCorp before the DOJ.",
        "MASTER SERVICES AGREEMENT: Apple Inc. enters into a software development contract with CyberDyne Systems. This agreement is governed by California Law and subject to General Data Protection Regulation (GDPR) privacy compliance.",
        "MUTUAL NON-DISCLOSURE AGREEMENT: OpenAI and BioSynth Global execute an NDA regarding proprietary AI model architecture. Regulatory oversight is maintained by the Federal Trade Commission (FTC).",
        "INTELLECTUAL PROPERTY LICENSING AGREEMENT: Microsoft Corporation grants a non-exclusive patent license to Apex Legal Group. Legal disputes shall be adjudicated under EU Patent Law.",
        "LITIGATION BRIEF: The Supreme Court of the United States (SCOTUS) reviewed civil liberties arguments in ACLU vs FBI. Attorney General held a press conference regarding law enforcement protocols.",
        "DATA PROTECTION ADDENDUM: Meta Platforms Inc. (Meta) submitted a DPA to the European Data Protection Board (EDPB) ensuring compliance with GDPR data transfer regulations.",
        "DEFENSE PROCUREMENT CONTRACT: Lockheed Martin and SpaceX secured a joint satellite launch agreement with NASA and the Pentagon, represented by Skadden Arps Slate Meagher & Flom.",
        "ESCROW & ASSET CUSTODY AGREEMENT: J.P. Morgan Chase & Co. and Goldman Sachs Group finalized digital asset custody protocols for Bitcoin (BTC) and Ethereum (ETH).",
        "ANTITRUST CONSENT DECREE: The Department of Justice (DOJ) and Federal Trade Commission (FTC) filed an antitrust enforcement decree involving Amazon.com Inc. in federal court.",
        "BANKRUPTCY RESTRUCTURING DEED: Skadden, Arps, Slate, Meagher & Flom LLP filed a corporate reorganization plan for Apex Holdings before the Delaware Bankruptcy Court.",
    ]
    return [Document(page_content=t, metadata={"doc_id": i + 1, "source": f"legal_doc_{i+1}.pdf"}) for i, t in enumerate(texts)]
