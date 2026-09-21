from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
import os

class PdfLoader:

    def load_pdfs(self, directory_path: str = "dataset"):
        docs = []
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if filename.endswith(".pdf"): 
                pdf_loader = PyMuPDFLoader(file_path)
                curr_docs = pdf_loader.load()
                docs.extend(curr_docs)
        return docs




