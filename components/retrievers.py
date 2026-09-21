from typing import List
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
import chromadb
import os
import uuid
from langchain_huggingface import HuggingFaceEmbeddings

class SparseRetriever:

    def __init__(self):
        self.retriever = None

    def add_docs(self, docs):
        if self.retriever:
            docs = self.retriever.docs + docs # combine previous docs
        self.retriever = BM25Retriever.from_documents(docs)

    def retrieve(self, query, top_k: int = 4):
        return self.retriever.invoke(query)[:top_k]

class DenseRetriever:

    def __init__(self, persistent_directory: str = "data/vector_store", collection_name: str = "company_pdfs"):
        self.persistent_directory =persistent_directory
        self.collection_name = collection_name
        os.makedirs(persistent_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persistent_directory
        )
        self.collection = self.client.get_or_create_collection(
            collection_name
        )
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)
        

    def add_docs(self, docs: List[Document], embeddings):
        if (len(docs) != len(embeddings)):
            raise Exception("Length of documents is not same as length of embeddings.")

        ids = []
        metadatas = []
        doc_contents = []

        for i, doc in enumerate(docs):
            id = f"doc_{uuid.uuid4()}"
            ids.append(id)
            metadatas.append(doc.metadata)
            doc_contents.append(doc.page_content)

        self.collection.add(
            ids=ids, 
            metadatas=metadatas, 
            documents=doc_contents, 
            embeddings=embeddings
        )

    def retrieve(self, query_embeddings, top_k=4):
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k
        )

        retrieved_docs = []
        if results["documents"] and results["documents"][0]:
            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            documents = results["documents"][0]
            distances = results["distances"][0]

            for i, (id, metadata, document, distance) in enumerate(zip(ids, metadatas, documents, distances)):
                similarity_score = 1 - distance

                retrieved_docs.append(Document(
                    id=id,
                    page_content=document,
                    metadata=metadata,
                ))

        return retrieved_docs


class HybridRetriever:

    def __init__(self, embedder_name="all-MiniLM-L6-v2", persistent_directory: str="data/vector_store", collection_name: str ="company_pdfs"):
        self.dense_retriever = DenseRetriever(persistent_directory=persistent_directory, collection_name=collection_name)
        embedder = HuggingFaceEmbeddings(model_name=embedder_name)
        self.ensemble_dense_retriever = Chroma(
            persist_directory=self.dense_retriever.persistent_directory,
            collection_name=self.dense_retriever.collection_name,
            embedding_function=embedder
        ).as_retriever()
        self.sparse_retriever = SparseRetriever()


    def add_docs(self, docs: List[Document], embeddings):
        if (len(docs) != len(embeddings)):
            raise Exception("Length of documents is not same as length of embeddings.")

        ids = []
        metadatas = []
        doc_contents = []

        for i, doc in enumerate(docs):
            # for dense retriever => each unit consists of id, metadata, page_content, embedding
            id = f"doc_{uuid.uuid4()}"
            ids.append(id)
            metadatas.append(doc.metadata)
            doc_contents.append(doc.page_content)

            # for sparse retriever => each unit is a doc having id, metadata, page_content
            docs[i].id = id

        self.dense_retriever.collection.add(
            ids=ids, 
            metadatas=metadatas, 
            documents=doc_contents, 
            embeddings=embeddings
        )
        self.sparse_retriever.add_docs(docs)

    def retrieve(self, query: str, top_k: int = 10):
        self.ensemble_dense_retriever.search_kwargs={"k": top_k - int(top_k / 2)}
        self.sparse_retriever.retriever.k = int(top_k / 2)
        ensemble_retriever = EnsembleRetriever(retrievers=[
            self.sparse_retriever.retriever,
            self.ensemble_dense_retriever
        ])
        return ensemble_retriever.invoke(query)




        




    
        