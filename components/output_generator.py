from typing import List
from langchain_core.documents import Document
from google import genai

class OutputGenerator:

    def generate_output(self, query: str, retrieved_docs: List[Document], client: genai.Client, model_name: str = "gemini-3.1-flash-lite"):
        context = " ".join([doc.page_content for doc in retrieved_docs])
        prompt = f"""Using given context, generate the answer. If there is no context, then apologize that you don't have enough info.
                    Context: {context}
                    Query: {query}
                 """
        response = client.interactions.create(
            input=prompt,
            model=model_name
        )
        return response.output_text