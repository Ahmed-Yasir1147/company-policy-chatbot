# CompanyPolicyChatbot

CompanyPolicyChatbot is a chatbot designed to provide information about company policies and guidelines using an **Advanced RAG (Retrieval-Augmented Generation)** pipeline.

The project is built using dummy company policy documents (generated via LLM). The main purpose of the project is to study different Advanced RAG techniques and evaluate how they improve the performance of RAG compared to a naive RAG pipeline.

## Tech Stack

* **LangChain** — RAG pipeline implementation
* **Gemini API** — Response generation and query transformation
* **RAGAS** — RAG evaluation
* **Qwen** — LLM used for RAGAS evaluation

## RAG Architecture

The project consists of many versions where in each version an advanced RAG technique is added and results are observed.

**Naive RAG → Semantic Chunking → Hybrid Retrieval → Query Transformation + Reranking**

### RAG0 — Naive RAG

The baseline RAG pipeline uses recursive chunking and dense retrieval.

```text
Documents
    ↓
Document Loader
    ↓
Recursive Chunker
    ↓
Embedding
    ↓
Dense Retriever
    ↓
Generator
    ↓
Answer
```

### RAG1 — Semantic Chunking

RAG1 replaces the recursive chunker with a semantic chunker to create chunks based on the meaning and structure of the document.

```text
Documents
    ↓
Document Loader
    ↓
Semantic Chunker
    ↓
Embedding
    ↓
Dense Retriever
    ↓
Generator
    ↓
Answer
```

### RAG2 — Hybrid Retrieval

RAG2 builds on RAG1 by replacing dense retrieval with hybrid retrieval, combining semantic and lexical retrieval.

```text
Documents
    ↓
Document Loader
    ↓
Semantic Chunker
    ↓
Embedding
    ↓
Hybrid Retriever
    ↓
Generator
    ↓
Answer
```

### RAG3 — Query Transformation + Reranking

RAG3 builds on RAG2 by adding query transformation before retrieval and a reranker after retrieval.

```text
Documents
    ↓
Document Loader
    ↓
Semantic Chunker
    ↓
Embedding
    ↓
Query Transformer
    ↓
Hybrid Retriever
    ↓
Reranker
    ↓
Generator
    ↓
Answer
```

## Results

The different RAG versions were evaluated using RAGAS.

| Version | Answer Relevancy | Faithfulness | Context Precision | Context Recall |
| ------- | ---------------: | -----------: | ----------------: | -------------: |
| RAG0    |           0.7178 |       0.7222 |            0.9699 |         1.0000 |
| RAG1    |           0.7275 |       0.8750 |            0.9699 |         1.0000 |
| RAG2    |           0.7703 |       0.9583 |            0.8380 |         0.9306 |
| RAG3    |       **0.7923** |       0.7976 |        **1.0000** |     **1.0000** |

## Conclusion

The experiment shows that progressively adding Advanced RAG techniques can improve the overall performance of the system.

- RAG0 showed decent performance.
- RAG1 (Semantic Chunking) improved Faithfulness.
- RAG2 (Hybrid retrieval) further improved Faithfulness and Answer Relevancy but decreased Precision and Recall
- RAG3 (Query transformation + Reranking) gave overall best performance except for Faithfulness which is observed to be lower than RAG1

This shows although Advanced RAG techniques improves performance incrementally but it's not a certainty.


