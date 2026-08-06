"""End-to-end integration demo connecting LangChain LLMGraphTransformer to AutoGraft."""
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

from autograft import Entity, ExistingNode, resolve_and_generate_cypher

load_dotenv()


def main() -> None:
    """Demonstrates seamless integration between LangChain entity extraction and AutoGraft ER."""
    # 1. Initialize LangChain LLM (connected to Groq / OpenRouter)
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("AUTOGRRAFT_LLM_MODEL", "groq/llama-3.3-70b-versatile")
    base_url = (
        "https://api.groq.com/openai/v1"
        if os.getenv("GROQ_API_KEY")
        else "https://openrouter.ai/api/v1"
    )

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )

    # 2. Simulate Existing Knowledge Graph Data in Neo4j
    existing_nodes = [
        ExistingNode(
            node_id="a1",
            canonical_name="Apple Inc.",
            type="Company",
            aliases=["Apple"],
        ),
        ExistingNode(
            node_id="j1",
            canonical_name="Jean Dupont",
            type="Person",
            aliases=["J. Dupont"],
        ),
    ]

    # 3. Define Source Text for Extraction
    source_text = (
        "J. Dupont has been promoted to CTO of Apple. He replaced John Smith."
    )

    print("=== End-to-End RAG Integration (LangChain -> AutoGraft) ===")
    print(f"Source Text: \"{source_text}\"\n")
    print(f"Loaded {len(existing_nodes)} existing nodes from Neo4j.\n")

    # 4. Extract Graph Entities via LangChain LLMGraphTransformer
    print("Running LangChain LLMGraphTransformer extraction...")
    transformer = LLMGraphTransformer(llm=llm)
    documents = [Document(page_content=source_text)]
    graph_documents = transformer.convert_to_graph_documents(documents)

    extracted_nodes = graph_documents[0].nodes if graph_documents else []
    print(
        f"LangChain extracted {len(extracted_nodes)} raw nodes: "
        f"{[f'{n.id} ({n.type})' for n in extracted_nodes]}\n"
    )

    # 5. Intercept & Resolve with AutoGraft
    print("--- AutoGraft Resolution & Cypher Generation ---")
    for node in extracted_nodes:
        entity = Entity(canonical_name=str(node.id), type=str(node.type))
        cypher = resolve_and_generate_cypher(entity, existing_nodes)

        print(f"Entity: '{node.id}' (Type: {node.type})")
        print(f"Generated Cypher:\n  {cypher}\n")


if __name__ == "__main__":
    main()
