<div align="center">
  
# Autograft

**Le middleware d'Entity Resolution économique pour GraphRAG.**

[English](README.md) | [Français](README_fr.md) | [中文](README_zh.md)

[![PyPI](https://img.shields.io/pypi/v/autograft)](https://pypi.org/project/autograft/)
</div>

Arrêtez de dupliquer les entités dans votre graphe de connaissances Neo4j. AutoGraft intercepte les entités extraites par LangChain ou LlamaIndex, utilise une approche hybride à 3 couches (Déterministe -> Vectoriel -> LLM) pour fusionner les doublons, et génère des requêtes Cypher propres.

## Installation

```bash
pip install autograft
```
*(Pour utiliser les intégrations, installez avec `pip install autograft[langchain]` ou `pip install autograft[llamaindex]`)*

## Pourquoi AutoGraft ?

- **Agnostique aux LLM** : Fonctionne avec OpenAI, Groq, Ollama, OpenRouter via litellm.
- **Réduction massive des coûts** : Réduit les coûts de tokens de l'Entity Resolution jusqu'à 100% en résolvant localement.
- **Plug & Play** : Remplacement direct avant votre base de données Neo4j (en 1 ligne de code).
- **Extrêmement Rapide** : C/C++ (RapidFuzz) et correspondance locale NumPy.

---

## Benchmark de Performance (200 Documents / 4 Industries)

Évalué sur **200 documents d'entreprise réels** couvrant 4 scénarios clés : **Juridique & Conformité**, **Tech & Logiciel d'Entreprise**, **Assurance & Gestion des Risques**, et **Finance & Banque d'Investissement**.

*Moteur LLM configuré* : **`groq/llama-3.1-8b-instant`** pour l'extraction & l'arbitrage, et **`groq/llama-3.3-70b-versatile`** pour l'audit de précision.

| Métrique | LangChain Naïf (Pas d'ER) | LangChain + Full LLM ER | LangChain + AutoGraft Hybride ER |
| :--- | :---: | :---: | :---: |
| Documents Traités | 200 documents | 200 documents | 200 documents |
| Entités Extraites | 742 entités | 742 entités | 742 entités |
| Appels API LLM ER | 0 appels | 742 appels | 0 appels *(Court-circuit local à 100%)* |
| Tokens Consommés | 0 tokens | 207,760 tokens | 0 tokens *(100% d'Économie)* |
| Doublons Créés | 188 doublons | 0 doublons | 0 doublons |
| Doublons Évités (`MATCH`) | 0 requêtes | 188 requêtes | 188 requêtes |
| Nouvelles Entités Créées (`MERGE`) | 742 requêtes | 554 requêtes | 554 requêtes |
| Coût LLM ER | $0.00000 | $0.04155 | $0.00000 |
| Qualité du Knowledge Graph | Pollué de Doublons | Dédupliqué (Coûteux) | Dédupliqué & Gratuit |

---

### Figure 1.1: Métriques de Performance RAG d'Entreprise
![Figure 1.1](benchmark/assets/macro_benchmark_metrics.png)

### Figure 1.2: Évolution des Coûts (Jusqu'à 1,000,000 de Documents)
Pour 1,000,000 de documents, AutoGraft maintient **$0.00** de coût LLM API pour l'Entity Resolution tout en garantissant un Graphe de Connaissances propre à 100%.

![Figure 1.2](benchmark/assets/macro_cost_scaling_1m.png)

### Figure 1.3: Précision par Secteur (100.0% Global)
![Figure 1.3](benchmark/assets/macro_accuracy_by_industry.png)

---

## Architecture (Court-Circuit à 3 Couches)

1. **Couche 1 (Déterministe)** : Correspondance exacte & alias via rapidfuzz (0 token, 0.1ms).
2. **Couche 2 (Sémantique)** : Similarité cosinus vectorielle via numpy (0 token, 0.5ms).
3. **Couche 3 (Arbitre LLM)** : Appel LiteLLM UNIQUEMENT pour les cas ambigus résiduels.

---

## Intégrations Plug & Play

AutoGraft fournit des intégrations natives en 1 ligne pour **LangChain** et **LlamaIndex**. 

### 🦜🔗 LangChain

```python
from langchain_community.graphs import Neo4jGraph
from autograft.integrations import AutoGraftNeo4jMiddleware

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
autograft_graph = AutoGraftNeo4jMiddleware(graph)

# AutoGraft déduplique tout silencieusement et localement !
autograft_graph.add_graph_documents(extracted_graph_documents)
```

### 🦙 LlamaIndex

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
