from components.document_loader import PdfLoader
from components.chunker import Chunker
from components.retrievers import HybridRetriever
from components.embedder import Embedder
from components.output_generator import OutputGenerator
from components.rag_evaluator import RAGEvaluator
from google import genai
import os
from typing import List, Dict
from ragas.llms import llm_factory

class RAGHybridRetrieverV2:

    def __init__(self, client : genai.Client):
        self.pdf_loader = PdfLoader()
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.hybrid_retriever = HybridRetriever()
        self.output_generator = OutputGenerator()
        self.evaluator = RAGEvaluator()
        self.client = client

    def ingestion_pipeline(self):
        docs = self.pdf_loader.load_pdfs()
        chunks = self.chunker.split_semantically(docs)
        embeddings = self.embedder.generate_embeddings([chunk.page_content for chunk in chunks])
        self.hybrid_retriever.add_docs(chunks, embeddings)

    def generation_pipeline(self, query):
        retrieved_docs = self.hybrid_retriever.retrieve(query, 4)
        response = self.output_generator.generate_output(query, retrieved_docs, self.client)
        return (response, retrieved_docs)

    def evaluation_pipeline(self, dataset: Dict[str, List[str]], llm: llm_factory, llm_embeddings):
        evaluation_dataset = []
        for query, references in dataset.items():
            response, curr_retrieved_docs = self.generation_pipeline(query)
            context = [doc.page_content for doc in curr_retrieved_docs]
            evaluation_dataset.append({
                "user_input": query,
                "reference": " ".join(references),
                "response": response,
                "retrieved_contexts": context
            })
        print(evaluation_dataset)
        return self.evaluator.evaluate(
            llm,
            llm_embeddings,
            evaluation_dataset
        )

        