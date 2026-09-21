from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self):
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, docs : List[Document], query : str, top_k=4):
        pairs = []
        for doc in docs:
            pairs.append((query, doc.page_content))
        scores = self.cross_encoder.predict(pairs)
        scored_docs = zip(docs, scores)
        sorted_scored_docs = sorted(scored_docs, reverse=True, key=lambda doc: doc[1])
        return [doc[0] for doc in sorted_scored_docs][:top_k]

