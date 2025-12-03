# PrismRAG: Multi-Perspective Query Decomposition for Adaptive Graph-Based Retrieval

**PrismRAG** refracts queries like light through a prism—decomposing them into three complementary perspectives (semantic traversal, global importance, and structural similarity), then recomposing the scattered insights into comprehensive answers. Built on [LightRAG](https://github.com/HKUDS/LightRAG), PrismRAG extends graph-based retrieval with adaptive schema discovery and intelligent entity merging.

## 🚀 Key Contributions

### 🔍 **Three-Perspective Expansion Architecture**

* **Problem**: LightRAG primarily supports one-hop expansion, limiting indirect but meaningful reasoning.
* **Solution**: PrismRAG employs a **three-perspective expansion approach** that processes queries from complementary angles: semantic multi-hop traversal, global importance ranking via Query-aware Personalized PageRank, and structural similarity through content-aware FastRP embeddings with edge weights.
* *Currently implemented for the NanoVectorDB backend.*

**🏗️ Architecture Overview:**

The system processes each query through three independent perspectives and merges results:

1. **Multi-hop Semantic Expansion**: Traditional graph traversal based on query-entity similarity
2. **Query-aware Global Importance Analysis**: Enhanced PageRank with query-sensitive seed weighting and edge reweighting
3. **Adaptive Structural Analysis**: Enhanced FastRP with edge weights for structure-aware embeddings

**Mathematical Framework:**

$$R_{final} = \text{merge}(R_{multihop}, R_{ppr}, R_{fastrp})$$

**🔄 Multi-hop Expansion:**

Graph traversal that expands from seed entities through neighbors, scoring candidates by entity-query similarity and relationship-query similarity.

$$S(e, q, h)
= \alpha \cdot \mathrm{sim}_{\text{entity}}(e, q)
+ \beta \cdot \mathrm{sim}_{\text{relation}}(r, q)
+ \gamma \cdot \delta^{h-1}$$

where:
- $\alpha = 0.4$ (entity similarity weight)
- $\beta = 0.5$ (relationship similarity weight)
- $\gamma = 0.1$ (distance decay weight)
- $\delta = 0.8$ (decay factor per hop)
- $h$ = hop count (distance from seed entities)

**🎯 Query-aware Personalized PageRank:**

$$PPR_{\text{query}}(v; S, W, E)
= (1 - d)\, p_{S,W}(v) + d \sum_{u \rightarrow v}
\frac{
PPR_{\text{query}}(u; S, W, E)\, e_{\text{weight}}(u,v)
}{
\sum_{x} e_{\text{weight}}(u,x)
}$$

where:
- $S$ = seed entities, $W$ = query-aware seed weights
- $E$ = query-aware edge weights, $d = 0.85$ (damping factor)
- $p_{S,W}(v)$ = personalization vector with weighted seeds
- $e_{weight}(u,v)$ = temporarily adjusted edge weights based on query-relation similarity

**🧬 Content-aware FastRP with Edge Weights:**

$$X = \sum_{k=0}^{K} w_k \cdot D^r \cdot S^k \cdot R$$

where $S = D^{-1/2}A_{weight}D^{-1/2}$, $A_{weight}$ uses co-occurrence-based edge weights

### 🧠 **Adaptive Entity Type Discovery**

* **Problem**: Predefined static schemas (e.g., Person, Organization, Category) often misclassify domain-specific metrics (e.g., EPS, Operating Margin).
* **Solution**: PrismRAG dynamically suggests and augments **domain-specific entity types** during extraction, improving recognition accuracy and retrieval relevance.

**Usage Examples:**

```python
# Option 1: Analyze documents from a folder (e.g., PDF files converted to .txt)
result = rag.entity_type_aug(input_folder="./financial_reports/")

# Option 2: Analyze text content directly (recommended for in-memory processing)
documents = [
    "Apple Q3 2024 Earnings Call - iPhone revenue $39.3B, up 1.5% YoY...",
    "Microsoft Azure grew 29% to $25.4B with AI integration..."
]
result = rag.entity_type_aug(texts=documents)

# Option 3: Force refresh to reprocess already-analyzed content
result = rag.entity_type_aug(texts=documents, force_refresh=True)
```


### ⚡ **Agentic Entity Merging**

* **Problem**: Duplicate entities scatter knowledge and reduce retrieval precision.
* **Solution**: PrismRAG combines **vector similarity pre-filtering** with **LLM-based reasoning** for intelligent entity deduplication. The system now supports **automatic merging during insertion**.
* *Currently implemented for the NanoVectorDB backend.*

**Mathematical Framework:**

- **Vector Similarity Pre-filtering:**

$$C = \{(e_i, e_j) | cosine(emb(e_i), emb(e_j)) > \theta, i < j\}$$

where $\theta$ = similarity threshold (e.g., 0.8)

- **LLM Decision Function:**

$$M(e_i, e_j) = \mathbb{I}[LLM_{confidence}("Should\ merge?", context(e_i, e_j)) > 0.95]$$

- **Merge Operation:**

$$entity_{final} = Merge(e_i, e_j) \text{ if } M(e_i, e_j) = \text{True}$$

### 🎯 **Context Recognition Filtering**

* **Problem**: Retrieved entities and relations often contain noise irrelevant to the query, wasting tokens and diluting LLM focus on critical information.
* **Solution**: PrismRAG employs **LLM-based relevance filtering** that removes irrelevant entities/relations, while preserving multi-hop reasoning paths and contextual information.

**Mathematical Framework:**

- **Removal Decision Function:**

$$R(e_i) = \mathbb{I}[LLM("Irrelevant\ to\ query?", context(q, e_i))]$$

where $q$ = user query, $e_i$ = entity/relation

- **Filter Operation:**

$$E_{filtered} = \{e_i \in E_{retrieved} | R(e_i) = \text{False}\}$$

### ⏳ **Automatic Timestamp Integration**

* **Problem**: Time-sensitive documents lose context when temporal information is inconsistent.
* **Solution**: PrismRAG automatically prefixes user-provided timestamps to entity and relation descriptions, ensuring consistent temporal alignment without relying on LLM memory.

**Mathematical Framework:**

$$Entity'(e, t) = T(t) \oplus Entity(e)$$
$$Relation'(r, t) = T(t) \oplus Relation(r)$$

where $T(t)$ is the timestamp prefix function, $\oplus$ denotes string concatenation

**Real-World Example:**

**Single Document Processing:**

*Original Entity Description:*
> Apple is a technology company that reported strong iPhone sales performance

*After Automatic Timestamp Prefixing (timestamp: "2024-Q3"):*
> 2024-Q3: Apple is a technology company that reported strong iPhone sales performance

**Multi-Document Merging:**

*After inserting Q2 and Q3 documents:*
> 2024-Q2: Apple reported iPhone revenue of $45.9 billion and Services revenue of $23.9 billion
>
> 2024-Q3: Apple reported iPhone revenue of $39.3 billion and Services revenue of $24.2 billion

**Token-Limit Exceeded with LLM Summary:**

When accumulated descriptions exceed n tokens (e.g., after inserting 10+ quarterly documents), the system automatically triggers summarization:

*LLM-Generated Summary:*
> Apple demonstrated consistent performance across 2024 quarters. Key trends: iPhone revenue peaked at $45.9B (Q2), declined to $39.3B (Q3); Services grew from $23.9B to $24.2B; strong expansion in emerging markets; iPhone 16 series launched Q4; maintained 44-46% gross margins despite supply chain challenges.

### 📊 **Table-Aware Document Processing**

* **Problem**: HTML tables in documents get fragmented during traditional chunking, causing loss of structural relationships and data integrity that are crucial for understanding financial reports, research papers, and structured documents.
* **Solution**: PrismRAG implements intelligent **table-aware chunking** that preserves complete `<table>...</table>` structures within single chunks, automatically includes surrounding context, and employs **table-enhanced entity extraction** with specialized prompts for tabular data.

---

## 🛠️ Installation

```bash
git clone https://github.com/lucien2712/PrismRAG.git
cd PrismRAG
pip install -e .
```

---

## 🚀 Quick Start

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

async def main():
    # Initialize PrismRAG
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
    )
    await rag.initialize_storages()
    
    # Adaptive entity type discovery from text content
    documents = ["Apple Q3 2024 earnings...", "Microsoft Azure revenue..."]
    rag.entity_type_aug(texts=documents)  # Or use input_folder="./reports/"

    # Insert with automatic timestamp integration and agentic merging
    await rag.insert(
        input=[document],
        timestamps=["2024-Q3"],  # Auto-prefixed to descriptions
        file_paths=["apple_q3_2024.pdf"],
        agentic_merging=True,           # Enable automatic entity merging
        agentic_merging_threshold=0.8   # Similarity threshold for merging
    )
    
    # Query with three-perspective expansion
    response = rag.query(
        "Apple iPhone revenue trends and supplier relationships",
        param=QueryParam(
            mode="hybrid", 
            max_hop=2,              # Multi-hop traversal depth
            top_neighbors=30,       # Max neighbor per node
            top_ppr_nodes=5,        # Top PageRank entities
            top_fastrp_nodes=5      # Top Content-aware FastRP structural entities
        )
    )

if __name__ == "__main__":
    asyncio.run(main())
```
---

## 🏗️ Workflow

```python
# 1. Discover entity types for your domain
documents = ["Apple Q3 2024 earnings...", "Microsoft Azure revenue..."]
rag.entity_type_aug(texts=documents)  # Or: input_folder="./reports/"

# 2. Insert documents with timestamps (tables processed automatically)
rag.insert(docs_with_tables, timestamps=["2024-Q3"], file_paths=["report.pdf"], agentic_merging=True, agentic_merging_threshold=0.8)
# Note: HTML tables are automatically detected and processed with context preservation

# 3. Query with three-perspective expansion
response = rag.query(query, param=QueryParam(
    max_hop=2,           # Multi-hop traversal
    top_neighbors=30,    # Top neighbor per node
    top_ppr_nodes=5,     # Personalized PageRank entities
    top_fastrp_nodes=5   # Content-aware FastRP structural entities
))
```

---

## 🤝 Contributing

PrismRAG extends **LightRAG** with:

1. **Three-Perspective Expansion** – Multi-hop traversal, Query-aware Personalized PageRank, and Content-aware FastRP structural similarity providing complementary retrieval perspectives.
2. **Adaptive Entity Type Discovery** – Dynamic schema induction for domain-specific contexts.
3. **Agentic Entity Merging** – Hybrid vector+LLM pipeline with cosine similarity pre-filtering and confidence thresholding.
4. **Context Recognition Filtering** – LLM-based relevance filtering to remove query-irrelevant entities/relations.
5. **Automatic Timestamp Integration** – Consistent temporal metadata injection with mathematical prefix functions.
6. **Table-Aware Document Processing** – Intelligent table detection, structure preservation, and context-aware chunking with specialized entity extraction for tabular data.

---

## 🙏 Acknowledgments

PrismRAG builds upon [LightRAG](https://github.com/HKUDS/LightRAG) by the HKUDS team (MIT License). We deeply appreciate their foundational work in graph-based RAG systems. We extend it with **Three-Perspective Expansion Architecture, Adaptive Entity Type Discovery, Agentic Entity Merging, Context Recognition Filtering, Automatic Timestamp Integration, Table-Aware Document Processing**
