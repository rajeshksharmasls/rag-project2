import hashlib
from typing import Optional


class ExactCache:
    """
    L1 Cache: Exact query match.
    Stores question → answer mapping.
    """

    def __init__(self):
        self.cache = {}

    def _hash_query(self, query: str) -> str:
        """Create deterministic key."""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        key = self._hash_query(query)
        return self.cache.get(key)

    def store(self, query: str, answer: str):
        key = self._hash_query(query)
        self.cache[key] = answer

    def exists(self, query: str) -> bool:
        key = self._hash_query(query)
        return key in self.cache
