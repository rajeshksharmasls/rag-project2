from typing import List, Dict


class ParentChildChunker:
    """
    Parent-Child Chunking Strategy

    Parent chunks: large context chunks
    Child chunks: smaller retrieval units

    Parent size  = 1500 characters
    Child size   = 300 characters
    """

    def __init__(self, parent_size: int = 1500, child_size: int = 300):
        self.parent_size = parent_size
        self.child_size = child_size

    # ---------------------------------------------------
    # Main chunking function
    # ---------------------------------------------------
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:

        metadata = metadata or {}

        parents = self._create_parent_chunks(text)

        chunks = []

        for parent_idx, parent_text in enumerate(parents):

            children = self._create_child_chunks(parent_text)

            for child_idx, child_text in enumerate(children):

                chunks.append(
                    {
                        "parent_id": f"parent_{parent_idx}",
                        "child_id": f"parent_{parent_idx}_child_{child_idx}",
                        "chunk_text": child_text,
                        "parent_text": parent_text,
                        **metadata,
                    }
                )

        return chunks

    # ---------------------------------------------------
    # Parent chunk creation
    # ---------------------------------------------------
    def _create_parent_chunks(self, text: str) -> List[str]:

        parents = []

        for i in range(0, len(text), self.parent_size):
            parent = text[i : i + self.parent_size]
            parents.append(parent.strip())

        return parents

    # ---------------------------------------------------
    # Child chunk creation
    # ---------------------------------------------------
    def _create_child_chunks(self, parent_text: str) -> List[str]:

        children = []

        for i in range(0, len(parent_text), self.child_size):
            child = parent_text[i : i + self.child_size]
            children.append(child.strip())

        return children
