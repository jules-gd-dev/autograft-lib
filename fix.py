
def main():
    with open("autograft/integrations/langchain.py", "r") as f:
        content = f.read()
    
    # Fix exception
    content = content.replace("try:", "import contextlib\n                with contextlib.suppress(Exception):")
    content = content.replace("                except Exception:\n                    pass  # If query fails, cache remains empty for this label", "                    pass")
    
    # Fix Type Checking imports
    content = content.replace("try:\n    from langchain_community.graphs import Neo4jGraph\n    from langchain_community.graphs.graph_document import GraphDocument\nexcept ImportError:\n    Neo4jGraph = Any\n    GraphDocument = Any", """from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_community.graphs import Neo4jGraph
    from langchain_community.graphs.graph_document import GraphDocument
else:
    Neo4jGraph = Any
    GraphDocument = Any""")
    
    # Fix node.id type mismatch
    content = content.replace("node.id = match_result.matched_node_id", "node.id = str(match_result.matched_node_id)")
    content = content.replace("rel.source.id = id_mapping[rel.source.id]", "rel.source.id = str(id_mapping[rel.source.id])")
    content = content.replace("rel.target.id = id_mapping[rel.target.id]", "rel.target.id = str(id_mapping[rel.target.id])")
    
    with open("autograft/integrations/langchain.py", "w") as f:
        f.write(content)

    with open("autograft/integrations/llamaindex.py", "r") as f:
        content = f.read()
    
    # Fix exception
    content = content.replace("try:", "import contextlib\n                with contextlib.suppress(Exception):")
    content = content.replace("                except Exception:\n                    pass  # If query fails, cache remains empty for this label", "                    pass")
    
    # Fix Type Checking imports
    content = content.replace("try:\n    from llama_index.core.graph_stores.types import EntityNode, Relation\n    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore\nexcept ImportError:\n    EntityNode = Any\n    Relation = Any\n    Neo4jPropertyGraphStore = Any", """from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from llama_index.core.graph_stores.types import EntityNode, Relation
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
else:
    EntityNode = Any
    Relation = Any
    Neo4jPropertyGraphStore = Any""")
    
    content = content.replace("node.name = match_result.matched_node_id", "node.name = str(match_result.matched_node_id)")

    with open("autograft/integrations/llamaindex.py", "w") as f:
        f.write(content)

    with open("autograft/layers/llm_arbiter.py", "r") as f:
        content = f.read()
        content = content.replace("            raise err", "            raise")
        content = content.replace("    except Exception:", "    except Exception as e:\n        import logging\n        logging.debug(e)")
    with open("autograft/layers/llm_arbiter.py", "w") as f:
        f.write(content)
        
    with open("tests/test_integrations.py", "r") as f:
        content = f.read()
        content = content.replace('    node_apple = MockEntityNode(name="Apple", label="Company")\n', '')
        content = content.replace('    node_iphone = MockEntityNode(name="iPhone", label="Product")\n', '')
    with open("tests/test_integrations.py", "w") as f:
        f.write(content)

main()
