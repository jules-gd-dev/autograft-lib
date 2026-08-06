"""Existing Knowledge Graph nodes for 4-industry macro benchmark."""
from autograft.models.entities import ExistingNode


def build_macro_existing_nodes() -> list[ExistingNode]:
    """Returns 50 existing Knowledge Graph nodes across Legal, Tech, Insurance, and Finance."""
    return [
        # Legal & Regulatory (15 nodes)
        ExistingNode(node_id="m1", canonical_name="Latham & Watkins LLP", type="LawFirm", aliases=["Latham & Watkins", "Latham"]),
        ExistingNode(node_id="m2", canonical_name="Kirkland & Ellis LLP", type="LawFirm", aliases=["Kirkland & Ellis", "Kirkland"]),
        ExistingNode(node_id="m3", canonical_name="Skadden, Arps, Slate, Meagher & Flom LLP", type="LawFirm", aliases=["Skadden"]),
        ExistingNode(node_id="m4", canonical_name="Department of Justice", type="LegalInstitution", aliases=["DOJ", "U.S. Department of Justice"]),
        ExistingNode(node_id="m5", canonical_name="Federal Trade Commission", type="LegalInstitution", aliases=["FTC"]),
        ExistingNode(node_id="m6", canonical_name="Supreme Court of the United States", type="LegalInstitution", aliases=["SCOTUS", "Supreme Court"]),
        ExistingNode(node_id="m7", canonical_name="General Data Protection Regulation", type="LegalConcept", aliases=["GDPR"]),
        ExistingNode(node_id="m8", canonical_name="Securities and Exchange Commission", type="LegalInstitution", aliases=["SEC"]),
        ExistingNode(node_id="m9", canonical_name="World Intellectual Property Organization", type="LegalInstitution", aliases=["WIPO"]),
        ExistingNode(node_id="m10", canonical_name="European Data Protection Board", type="LegalInstitution", aliases=["EDPB"]),
        # Tech & Enterprise Software (12 nodes)
        ExistingNode(node_id="m11", canonical_name="Alphabet Inc.", type="Company", aliases=["Google", "Google LLC"]),
        ExistingNode(node_id="m12", canonical_name="Apple Inc.", type="Company", aliases=["Apple"]),
        ExistingNode(node_id="m13", canonical_name="Microsoft Corporation", type="Company", aliases=["Microsoft", "MSFT"]),
        ExistingNode(node_id="m14", canonical_name="Amazon Web Services", type="Company", aliases=["AWS", "Amazon"]),
        ExistingNode(node_id="m15", canonical_name="Kubernetes", type="Software", aliases=["K8s"]),
        ExistingNode(node_id="m16", canonical_name="Docker Inc.", type="Software", aliases=["Docker"]),
        ExistingNode(node_id="m17", canonical_name="OpenAI", type="Company"),
        ExistingNode(node_id="m18", canonical_name="Anthropic", type="Company"),
        ExistingNode(node_id="m19", canonical_name="Meta Platforms Inc.", type="Company", aliases=["Meta", "Facebook"]),
        ExistingNode(node_id="m20", canonical_name="Elastic N.V.", type="Company", aliases=["Elasticsearch"]),
        # Insurance & Risk (12 nodes)
        ExistingNode(node_id="m21", canonical_name="Allianz SE", type="Company", aliases=["Allianz"]),
        ExistingNode(node_id="m22", canonical_name="AXA SA", type="Company", aliases=["AXA"]),
        ExistingNode(node_id="m23", canonical_name="Ping An Insurance", type="Company", aliases=["Ping An"]),
        ExistingNode(node_id="m24", canonical_name="Prudential Financial", type="Company", aliases=["Prudential"]),
        ExistingNode(node_id="m25", canonical_name="Berkshire Hathaway Specialty Insurance", type="Company", aliases=["BHSI"]),
        ExistingNode(node_id="m26", canonical_name="Directors and Officers Liability", type="InsuranceType", aliases=["D&O"]),
        ExistingNode(node_id="m27", canonical_name="Errors and Omissions Insurance", type="InsuranceType", aliases=["E&O"]),
        ExistingNode(node_id="m28", canonical_name="Commercial General Liability", type="InsuranceType", aliases=["CGL", "GL"]),
        ExistingNode(node_id="m29", canonical_name="Property and Casualty Insurance", type="InsuranceType", aliases=["P&C"]),
        ExistingNode(node_id="m30", canonical_name="Third Party Administrator", type="Role", aliases=["TPA"]),
        # Finance & Investment Banking (11 nodes)
        ExistingNode(node_id="m31", canonical_name="JP Morgan Chase & Co.", type="Company", aliases=["J.P. Morgan", "JPM"]),
        ExistingNode(node_id="m32", canonical_name="Goldman Sachs Group", type="Company", aliases=["Goldman Sachs", "GS"]),
        ExistingNode(node_id="m33", canonical_name="Morgan Stanley", type="Company"),
        ExistingNode(node_id="m34", canonical_name="BlackRock Inc.", type="Company", aliases=["BlackRock"]),
        ExistingNode(node_id="m35", canonical_name="Société Générale", type="Company", aliases=["SocGen"]),
        ExistingNode(node_id="m36", canonical_name="BNP Paribas", type="Company"),
        ExistingNode(node_id="m37", canonical_name="Earnings Before Interest Taxes Depreciation Amortization", type="FinancialConcept", aliases=["EBITDA"]),
        ExistingNode(node_id="m38", canonical_name="Know Your Customer and Anti-Money Laundering", type="ComplianceConcept", aliases=["KYC/AML", "KYC", "AML"]),
        ExistingNode(node_id="m39", canonical_name="Initial Public Offering", type="FinancialEvent", aliases=["IPO"]),
        ExistingNode(node_id="m40", canonical_name="Secured Overnight Financing Rate", type="FinancialConcept", aliases=["SOFR"]),
    ]
