import numpy as np
from typing import Optional
from src.utils.embeddings import embed_text


class SemanticCache:
    """
    L2 Cache: semantic similarity cache.
    Uses embeddings to detect similar queries.
    """

    def __init__(self, similarity_threshold: float = 0.90):
        self.cache = []
        self.threshold = similarity_threshold

    def _cosine(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def search(self, query: str) -> Optional[str]:

        query_embedding = embed_text(query)

        for item in self.cache:
            sim = self._cosine(query_embedding, item["embedding"])

            if sim >= self.threshold:
                return item["answer"]

        return None

    def store(self, query: str, answer: str):

        query_embedding = embed_text(query)

        self.cache.append(
            {"embedding": query_embedding, "answer": answer, "query": query}
        )
