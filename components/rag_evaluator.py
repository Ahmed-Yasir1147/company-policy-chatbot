# ragas dependency issue fix
import sys
import types

dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat

from typing import List, Dict
from ragas import evaluate, EvaluationDataset
from ragas.llms import llm_factory
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas.run_config import RunConfig
import langchain_community.llms
langchain_community.llms.VertexAI = type("VertexAI", (object,), {})


class RAGEvaluator:

    def evaluate(self, llm: llm_factory, llm_embeddings, evaluation_dataset: List[Dict]):
        dataset = EvaluationDataset.from_dict(evaluation_dataset)
        results = evaluate(
            dataset=dataset,
            llm=llm,
            embeddings=llm_embeddings,
            metrics=[
                answer_relevancy,
                faithfulness,
                context_precision, 
                context_recall
            ],
            run_config=RunConfig(
                max_workers=1,
                timeout=1000,
                max_retries=2
            )   
        )
        return results

