import json
from typing import List
from google import genai
from pydantic import BaseModel, Field

class MultiQueries(BaseModel):
    queries: List[str] = Field(description="Collection of alternate queries.")

class QueryTransformer:

    def multi_query(self, query: str, client: genai.Client, queries_num=3, model_name="gemini-3.1-flash-lite"):
        prompt = """
            Generate exactly {queries_num} alternative versions of the following query, preserving its meaning while using different wording; output only the 3 rewritten queries separatt.ed by newlines, with no numbering, explanations, or repetition of the original prompt.
            Query: {query}
        """
        response = client.interactions.create(
            input=prompt,
            model=model_name,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": MultiQueries.model_json_schema()
            }
        )
        output = json.loads(response.output_text)
        queries = output["queries"]
        queries.append(query)
        return queries
