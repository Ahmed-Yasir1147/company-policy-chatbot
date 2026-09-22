# This file runs the various versions of RAG pipeline 
# & compare the evaluation results

from versions.rag_naive_v0 import RAGNaiveV0
from versions.rag_semantic_chunker_v1 import RAGSemanticChunkerV1
from versions.rag_hybrid_retriever_v2 import RAGHybridRetrieverV2
from versions.rag_query_transformer_v3 import RAGQueryTransformerV3
from versions.rag_naive_v0 import RAGNaiveV0
import os
from dotenv import load_dotenv
from google import genai
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from components.testing_dataset import dataset

# Gemini Client
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# Ollama model
llm = ChatOllama(model="qwen2.5:1.5b")
embeddings = OllamaEmbeddings(model="embeddinggemma")

rag0 = RAGNaiveV0(client)
rag1 = RAGSemanticChunkerV1(client)
rag2 = RAGHybridRetrieverV2(client)
rag3 = RAGQueryTransformerV3(client)

rags = {
    "RAG0": rag0,
    "RAG1": rag1,
    "RAG2": rag2,
    "RAG3": rag3 
}

for version, rag in rags.items():
    rag.ingestion_pipeline()
    evaluation_results = rag.evaluation_pipeline(dataset, llm, embeddings)
    print(f"{version}: {evaluation_results}")

# RAG0: {'answer_relevancy': 0.7178, 'faithfulness': 0.7222, 'context_precision': 0.9699, 'context_recall': 1.0000}
# RAG1: {'answer_relevancy': 0.7275, 'faithfulness': 0.8750, 'context_precision': 0.9699, 'context_recall': 1.0000}
# RAG2: {'answer_relevancy': 0.7703, 'faithfulness': 0.9583, 'context_precision': 0.8380, 'context_recall': 0.9306}
# RAG3: {'answer_relevancy': 0.7923, 'faithfulness': 0.7976, 'context_precision': 1.0000, 'context_recall': 1.0000}