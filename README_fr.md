<div align="center">
  
# Autograft

**Le middleware d'Entity Resolution économique pour GraphRAG.**

[English](README.md) | [Français](README_fr.md) | [中文](README_zh.md)

[![PyPI - Version](https://img.shields.io/pypi/v/autograft?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/autograft/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/autograft?logo=python&logoColor=white)](https://pypi.org/project/autograft/)
</div>

Arrêtez de dupliquer les entités dans votre graphe de connaissances Neo4j. AutoGraft intercepte les entités extraites par LangChain ou LlamaIndex, utilise une approche hybride à 3 couches (Déterministe -> Vectoriel -> LLM) pour fusionner les doublons, et génère des requêtes Cypher propres.

## Installation

```bash
pip install autograft
```
*(Pour utiliser les intégrations, installez avec `pip install autograft[langchain]` ou `pip install autograft[llamaindex]`)*

## Pourquoi AutoGraft ?

- **Agnostique aux LLM** : Fonctionne avec OpenAI, Groq, Ollama, OpenRouter via litellm.
- **Réduction majeure des coûts** : Résout ~99.8% des entités localement, réduisant le coût de l'ER LLM d'environ 99.8%.
- **Plug & Play** : Remplacement direct avant votre base Neo4j (1 ligne de code).
- **Extrêmement Rapide** : RapidFuzz (C/C++) et NumPy (moyenne 0.3 ms/entité).

> **Embeddings :** AutoGraft *consomme* les embeddings, il ne les *crée* pas.
> Fournissez des vecteurs pré-calculés (ex. `sentence-transformers`, OpenAI) via les
> propriétés des nœuds. Voir `AutoGraftConfig.embedding_attr`.

---

## Benchmark de Performance (500 Documents / 10 Industries — Exécution Réelle)

Évalué sur **500 documents** contenant **3 000 mentions d'entités** de **63 identités
réelles** sur 10 secteurs, avec de vraies ambiguïtés (abréviations, tickers boursiers,
homonymes). Aucun facteur d'échelle — l'exécution complète sur 500 docs a été faite.

*Méthodologie :* vrais embeddings `all-MiniLM-L6-v2`, vrais appels LLM Groq
(`groq/llama-3.1-8b-instant`), vrais prix `litellm`. Précision mesurée contre un
corpus de vérité-terrain. Voir [BENCHMARK.md](BENCHMARK.md) pour les détails.

| Métrique | Naïf (sans ER) | Full LLM ER | AutoGraft (hybride) |
| :--- | :---: | :---: | :---: |
| Mentions d'entités | 3000 | 3000 | 3000 |
| Appels API LLM ER | 0 | 3000 | **7** (99.8% local) |
| Tokens consommés | 0 | 661 714 | **1 566** |
| Coût LLM ER | $0.000 | $0.0334 | **$0.00008** |
| Nœuds finaux du graphe | 3000 | 63 | 81 |
| Précision / Rappel | — | — | **100% / 99.4%** |

---

### Figure 1.1 : Métriques de Performance Brutes (500 docs réels)
![Figure 1.1](benchmark/assets/real_metrics.png)

### Figure 1.2 : Répartition par Couche de Résolution
![Figure 1.2](benchmark/assets/real_layers.png)

- `deterministic_match` : 2869 (95.6%) — correspondance exacte/alias/lexical, 0 token
- `semantic_match` : 3 (0.1%) — similarité cosinus ≥ 0.85, 0 token
- `llm_merge` : 7 (0.2%) — fusion confirmée par LLM
- `no_match_declined` : 121 (4.0%) — nouveau nœud unique ou fusion manquée

### Figure 1.3 : Évolution des Coûts (coût mesuré par doc, projection linéaire)
Pour 1 000 000 de documents, AutoGraft maintient le coût ER LLM autour de **0,16 $**
contre **66 800 $** pour une approche full-LLM (~99.999% d'économie).

![Figure 1.3](benchmark/assets/real_cost_scaling.png)

### Figure 1.4 : Précision vs Vérité-Terrain
![Figure 1.4](benchmark/assets/real_accuracy.png)

*Pour la méthodologie complète et les limites honnêtes (les ~18 fusions manquées
sans alias_map, ~4 avec alias_map), voir [BENCHMARK.md](BENCHMARK.md).*

---

## Configuration & Clés API

AutoGraft peut être configuré via des variables d'environnement (`.env`) ou dynamiquement via la classe `AutoGraftConfig`.

```python
from autograft import AutoGraftConfig
from autograft.integrations import AutoGraftNeo4jMiddleware

config = AutoGraftConfig(
    model="openai/gpt-4o",
    api_key="sk-...",
    api_base="https://custom.endpoint/v1",  # Optionnel : proxy, Azure, Ollama
    match_threshold=0.85,
    id_attr="id",
    aliases_attr="aliases",
    matching_algorithm="token_sort_ratio"
)
autograft_graph = AutoGraftNeo4jMiddleware(graph, config=config)
```

---

## Architecture (Court-Circuit à 4 Couches)

1. **Couche 1 (Déterministe)** : Correspondance exacte & alias via rapidfuzz (0 token, 0.1ms).
2. **Couche 1.5 (Lexicale)** : Suppression de suffixes + détection d'acronymes (0 token, 0.1ms).
3. **Couche 2 (Sémantique)** : Similarité cosinus vectorielle via numpy (0 token, 0.5ms).
4. **Couche 3 (Arbitre LLM)** : Appel LiteLLM UNIQUEMENT pour les cas ambigus résiduels.

---

## Intégrations Plug & Play

AutoGraft fournit des intégrations natives en 1 ligne pour **LangChain** et **LlamaIndex**. 

### LangChain

```python
from langchain_community.graphs import Neo4jGraph
from autograft.integrations import AutoGraftNeo4jMiddleware

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
autograft_graph = AutoGraftNeo4jMiddleware(graph)

# AutoGraft déduplique tout silencieusement et localement !
autograft_graph.add_graph_documents(extracted_graph_documents)
```

### LlamaIndex

```python
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from autograft.integrations import AutoGraftLlamaIndexMiddleware

store = Neo4jPropertyGraphStore(username="neo4j", password="password", url="bolt://localhost:7687")
autograft_store = AutoGraftLlamaIndexMiddleware(store)

# Utilisez dans votre pipeline LlamaIndex
# index = PropertyGraphIndex.from_documents(documents, property_graph_store=autograft_store)
```

---

## Licence
MIT
