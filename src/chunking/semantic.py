import numpy as np
import re
from typing import List, Dict
from src.utils.embeddings import embed_text


class SemanticChunker:
    """
    Semantic Chunking based on embedding similarity.

    Steps:
    1. Split document into sentences
    2. Generate embeddings
    3. Compute similarity between adjacent sentences
    4. Detect semantic boundaries
    5. Form chunks based on similarity threshold
    """

    def __init__(
        self, similarity_threshold: float = 0.75, max_chunk_sentences: int = 8
    ):
        self.similarity_threshold = similarity_threshold
        self.max_chunk_sentences = max_chunk_sentences

    # ---------------------------------------------------
    # Sentence Splitter
    # ---------------------------------------------------
    def split_sentences(self, text: str) -> List[str]:
        """
        Basic sentence splitter using regex.
        """
        sentences = re.split(r"(?<=[.!?]) +", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    # ---------------------------------------------------
    # Cosine Similarity
    # ---------------------------------------------------
    def cosine_similarity(self, vec1, vec2) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # ---------------------------------------------------
    # Chunking Logic
    # ---------------------------------------------------
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:

        metadata = metadata or {}

        sentences = self.split_sentences(text)

        if len(sentences) <= 1:
            return [{"text": text, **metadata}]

        # Generate embeddings
        embeddings = [embed_text(sentence) for sentence in sentences]

        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):

            sim = self.cosine_similarity(embeddings[i - 1], embeddings[i])

            # boundary detection
            if (
                sim < self.similarity_threshold
                or len(current_chunk) >= self.max_chunk_sentences
            ):
                chunks.append(" ".join(current_chunk))
                current_chunk = []

            current_chunk.append(sentences[i])

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # attach metadata
        result = []
        for idx, chunk in enumerate(chunks):
            result.append({"chunk_id": idx, "text": chunk, **metadata})

        return result
