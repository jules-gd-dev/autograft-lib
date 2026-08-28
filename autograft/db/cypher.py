"""Cypher identifier quoting for dynamically built Neo4j queries."""


def quote_label(label: str) -> str:
    """Wrap an identifier in backticks, doubling any embedded backtick.

    Labels come from untrusted LLM extractions (``entity.type``). Doubling
    backticks keeps the value a single identifier instead of letting it break
    out of the label context in queries like ``MATCH (n:`{type}`)``.
    """
    return f"`{label.replace('`', '``')}`"
