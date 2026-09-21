from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import SentenceTransformerEmbeddings


class Chunker:

    def __init__(self, sentence_transformer_model="sentence-transformers/all-MiniLM-L6-v2"):
        embedder = SentenceTransformerEmbeddings(model_name=sentence_transformer_model)
        self.semantic_chunker = SemanticChunker(
            embedder,
            breakpoint_threshold_type="percentile"
        )


    def split_recursively(self, docs : List[Document], chunk_size=100, chunk_overlap=10):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(
            docs
        )

    def split_semantically(self, docs : List[Document]):
        return self.semantic_chunker.split_documents(docs)

    