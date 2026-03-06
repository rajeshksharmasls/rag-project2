from typing import List, Dict, Optional
import hashlib


class RetrievalCache:
    """
    L3 Cache: cache retrieved document chunks.
    Avoids repeated vector DB searches.
    """

    def __init__(self):
        self.cache = {}

    def _hash_query(self, query: str):

        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[List[Dict]]:

        key = self._hash_query(query)
        return self.cache.get(key)

    def store(self, query: str, chunks: List[Dict]):

        key = self._hash_query(query)
        self.cache[key] = chunks

    def exists(self, query: str) -> bool:

        key = self._hash_query(query)
        return key in self.cache
