<div align="center">
  
# Autograft

**适用于 GraphRAG 的高性价比实体解析中间件。**

[English](README.md) | [Français](README_fr.md) | [中文](README_zh.md)

[![PyPI - Version](https://img.shields.io/pypi/v/autograft?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/autograft/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/autograft?logo=python&logoColor=white)](https://pypi.org/project/autograft/)
</div>

停止在您的 Neo4j 知识图谱中产生重复实体。AutoGraft 会拦截 LangChain 或 LlamaIndex 提取的实体，使用 3 层混合方法（确定性 -> 向量 -> LLM）来合并重复项，并生成干净的 Cypher 查询。

## 安装

```bash
pip install autograft
```
*(要使用框架集成，请使用 `pip install autograft[langchain]` 或 `pip install autograft[llamaindex]` 进行安装)*

## 为什么选择 AutoGraft？

- **与 LLM 无关**：通过 litellm 支持 OpenAI、Groq、Ollama、OpenRouter 等。
- **大幅降低成本**：在本地解析约 99.8% 的实体，将 LLM 实体解析成本降低约 99.8%。
- **即插即用**：直接放在您的 Neo4j 数据库之前作为替代方案（只需 1 行代码）。
- **极速性能**：RapidFuzz (C/C++) 和 NumPy 本地匹配（平均 0.3 毫秒/实体）。

> **嵌入向量：** AutoGraft *消费* 嵌入向量，而不会*生成*它们。请通过节点属性
> 提供预计算的向量（如 `sentence-transformers`、OpenAI）。参见 `AutoGraftConfig.embedding_attr`。

---

## 性能基准测试 (500 份文档 / 10 个行业 — 真实运行)

在 **500 份文档**上评估，包含 **3 000 条实体提及**，涵盖 10 个行业的 **63 个真实
身份**，带有真正的歧义（缩写、股票代码、同音异义词）。无缩放系数 — 完整的 500 份
文档运行已执行。

*方法论：* 真实的 `all-MiniLM-L6-v2` 嵌入，真实的 Groq LLM 仲裁调用
(`groq/llama-3.1-8b-instant`)，真实的 `litellm` 定价。精确度对照真值语料库测量。
详见 [BENCHMARK.md](BENCHMARK.md)。

| 指标 | 原始 (无 ER) | 全 LLM ER | AutoGraft (混合) |
| :--- | :---: | :---: | :---: |
| 实体提及 | 3000 | 3000 | 3000 |
| LLM ER API 调用 | 0 | 3000 | **7** (99.8% 本地) |
| 消耗的 Tokens | 0 | 661,714 | **1,566** |
| LLM ER 成本 | $0.000 | $0.0334 | **$0.00008** |
| 最终图节点 | 3000 | 63 | 81 |
| 精确率 / 召回率 | — | — | **100% / 99.4%** |

---

### 图 1.1: 原始性能指标 (500 份真实文档)
![Figure 1.1](benchmark/assets/real_metrics.png)

### 图 1.2: 解析层分布
![Figure 1.2](benchmark/assets/real_layers.png)

- `deterministic_match`：2869 (95.6%) — 精确/别名/词汇重匹配，0 tokens
- `semantic_match`：3 (0.1%) — 余弦相似度 ≥ 0.85，0 tokens
- `llm_merge`：7 (0.2%) — LLM 确认合并
- `no_match_declined`：121 (4.0%) — 新唯一节点或遗漏合并

### 图 1.3: 成本扩展 (实测每文档成本，线性投影)
对于 1,000,000 份文档，AutoGraft 将 LLM 实体解析成本保持在约 **$0.16**，
而全 LLM 方法为 **$66,800**（节省约 99.999%）。

![Figure 1.3](benchmark/assets/real_cost_scaling.png)

### 图 1.4: 精确率 vs 真值
![Figure 1.4](benchmark/assets/real_accuracy.png)

*完整的评估方法论和诚实的局限性说明（无 alias_map 时约 18 个漏合并，
有 alias_map 时约 4 个），请参见 [BENCHMARK.md](BENCHMARK.md)。*

---

## 配置与 API 密钥

AutoGraft 可以通过环境变量 (`.env`) 或以编程方式通过 `AutoGraftConfig` 类进行配置。

```python
from autograft import AutoGraftConfig
from autograft.integrations import AutoGraftNeo4jMiddleware

config = AutoGraftConfig(
    model="openai/gpt-4o",
    api_key="sk-...",
    api_base="https://custom.endpoint/v1",  # 可选：用于代理、Azure 或本地 LLM
    match_threshold=0.85,
    id_attr="id",
    aliases_attr="aliases",
    matching_algorithm="token_sort_ratio"
)
autograft_graph = AutoGraftNeo4jMiddleware(graph, config=config)
```

---

## 架构 (4 层短路机制)

1. **第 1 层 (确定性)**: 通过 rapidfuzz 进行精确字符串及别名匹配 (0 token, 0.1ms)。
2. **第 1.5 层 (词汇)**: 后缀剥离核心匹配 (fuzz≥90) + 精确首字母缩略词检测 (0 token, 0.1ms)。
3. **第 2 层 (语义)**: 通过 numpy 计算向量余弦相似度 (0 token, 0.5ms)。
4. **第 3 层 (LLM 仲裁)**: 仅在处理残留的歧义情况时调用 LiteLLM。

---

## 即插即用集成

AutoGraft 为 **LangChain** 和 **LlamaIndex** 提供原生的 1 行集成代码。

### LangChain

```python
from langchain_community.graphs import Neo4jGraph
from autograft.integrations import AutoGraftNeo4jMiddleware

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
autograft_graph = AutoGraftNeo4jMiddleware(graph)

# AutoGraft 会在本地静默完成所有去重工作！
autograft_graph.add_graph_documents(extracted_graph_documents)
```

### LlamaIndex

```python
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from autograft.integrations import AutoGraftLlamaIndexMiddleware

store = Neo4jPropertyGraphStore(username="neo4j", password="password", url="bolt://localhost:7687")
autograft_store = AutoGraftLlamaIndexMiddleware(store)

# 在您的 LlamaIndex 管道中使用
# index = PropertyGraphIndex.from_documents(documents, property_graph_store=autograft_store)
```

---

## 许可证
MIT
