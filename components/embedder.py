from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

class Embedder:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
            self.model_name = model_name
            self.embedder = HuggingFaceEmbeddings(model_name=model_name)
    
    def generate_embeddings(self, texts : List[str]):
        return self.embedder.embed_documents(texts)