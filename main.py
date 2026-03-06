#!/usr/bin/env python3
"""
RAG Project 2 - Main Application
A complete Retrieval-Augmented Generation system with multi-tenant support,
semantic chunking, caching, and citation-aware generation.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Local imports
from src.utils.logger import setup_logging, get_logger
from src.utils.config import (
    PINECONE_API_KEY,
    OPENAI_API_KEY,
    PINECONE_INDEX_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    RERANK_TOP_N,
)
from src.utils.embeddings import upsert_document
from src.chunking.semantic import SemanticChunker
from src.chunking.parent_child import ParentChildChunker
from src.retrieval.retreiver import search
from src.retrieval.reranker import rerank_results
from src.generation.generator import generate_answer
from src.caching.exact_cache import ExactCache
from src.caching.semantic_cache import SemanticCache
from src.caching.retreival_cache import RetrievalCache

# Setup logging
setup_logging()
logger = get_logger(__name__)


class RAGApplication:
    """Main RAG application orchestrating all components."""

    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.logger = get_logger(f"{__class__.__name__}.{user_id}")

        # Initialize caches
        self.exact_cache = ExactCache()
        self.semantic_cache = SemanticCache()
        self.retrieval_cache = RetrievalCache()

        # Initialize chunkers
        self.semantic_chunker = SemanticChunker()
        self.parent_child_chunker = ParentChildChunker(
            parent_chunk_size=CHUNK_SIZE * 2,
            child_chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        self.logger.info("RAG Application initialized")

    def ingest_document(
        self,
        document_path: str,
        document_id: str,
        use_semantic_chunking: bool = True,
        version: str = "v1",
    ) -> bool:
        """
        Ingest a document into the RAG system.

        Args:
            document_path: Path to the document file
            document_id: Unique identifier for the document
            use_semantic_chunking: Whether to use semantic chunking
            version: Document version

        Returns:
            Success status
        """
        try:
            self.logger.info(f"Ingesting document: {document_path}")

            # Read document
            with open(document_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Choose chunking strategy
            if use_semantic_chunking:
                chunks = self.semantic_chunker.chunk(content)
            else:
                chunks = self.parent_child_chunker.chunk(content)

            # Upsert to vector store
            success = upsert_document(
                user_id=self.user_id,
                document_id=document_id,
                chunks=chunks,
                version=version,
            )

            if success:
                self.logger.info(f"Successfully ingested {len(chunks)} chunks")
            else:
                self.logger.error("Failed to ingest document")

            return success

        except Exception as e:
            self.logger.error(f"Error ingesting document: {e}")
            return False

    def query(
        self, question: str, use_cache: bool = True, use_reranking: bool = True
    ) -> Dict:
        """
        Answer a question using the RAG pipeline.

        Args:
            question: The question to answer
            use_cache: Whether to use caching
            use_reranking: Whether to use reranking

        Returns:
            Response dictionary with answer and metadata
        """
        try:
            self.logger.info(f"Processing query: {question}")

            # Check exact cache first
            if use_cache:
                cached_answer = self.exact_cache.get(question)
                if cached_answer:
                    self.logger.info("Answer found in exact cache")
                    return {
                        "answer": cached_answer,
                        "source": "exact_cache",
                        "confidence": 1.0,
                    }

            # Check semantic cache
            if use_cache:
                cached_answer = self.semantic_cache.get(question)
                if cached_answer:
                    self.logger.info("Answer found in semantic cache")
                    return {
                        "answer": cached_answer,
                        "source": "semantic_cache",
                        "confidence": 0.9,
                    }

            # Retrieve relevant chunks
            search_results = search(user_id=self.user_id, query=question, top_k=TOP_K)

            if not search_results:
                self.logger.warning("No search results found")
                return {
                    "answer": "I don't have enough information to answer that question.",
                    "source": "no_results",
                    "confidence": 0.0,
                }

            # Cache retrieval results
            if use_cache:
                self.retrieval_cache.store(question, search_results)

            # Rerank if enabled
            if use_reranking and len(search_results) > RERANK_TOP_N:
                search_results = rerank_results(
                    question, search_results, top_n=RERANK_TOP_N
                )

            # Generate answer
            answer = generate_answer(
                user_id=self.user_id, question=question, chunks=search_results
            )

            # Cache the answer
            if use_cache:
                self.exact_cache.store(question, answer)

            self.logger.info("Answer generated successfully")

            return {
                "answer": answer,
                "source": "generated",
                "confidence": 0.8,
                "num_chunks": len(search_results),
            }

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "answer": "An error occurred while processing your question.",
                "source": "error",
                "confidence": 0.0,
                "error": str(e),
            }


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description="RAG Project 2 - Retrieval Augmented Generation"
    )
    parser.add_argument(
        "--user-id", default="default_user", help="User ID for multi-tenancy"
    )
    parser.add_argument(
        "--mode",
        choices=["ingest", "query", "interactive"],
        default="interactive",
        help="Operation mode",
    )

    # Ingest options
    parser.add_argument("--document", help="Document path for ingestion")
    parser.add_argument("--doc-id", help="Document ID for ingestion")
    parser.add_argument(
        "--chunking",
        choices=["semantic", "parent_child"],
        default="semantic",
        help="Chunking strategy",
    )

    # Query options
    parser.add_argument("--question", help="Question to answer")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")

    args = parser.parse_args()

    # Validate API keys
    if not PINECONE_API_KEY or not OPENAI_API_KEY:
        logger.error(
            "Missing required API keys. Please set PINECONE_API_KEY and OPENAI_API_KEY environment variables."
        )
        sys.exit(1)

    # Initialize application
    app = RAGApplication(user_id=args.user_id)

    if args.mode == "ingest":
        if not args.document or not args.doc_id:
            logger.error("Document path and ID required for ingestion")
            sys.exit(1)

        success = app.ingest_document(
            document_path=args.document,
            document_id=args.doc_id,
            use_semantic_chunking=(args.chunking == "semantic"),
        )

        if success:
            logger.info("Document ingested successfully")
        else:
            logger.error("Document ingestion failed")
            sys.exit(1)

    elif args.mode == "query":
        if not args.question:
            logger.error("Question required for query mode")
            sys.exit(1)

        result = app.query(
            question=args.question,
            use_cache=not args.no_cache,
            use_reranking=not args.no_rerank,
        )

        print(f"Answer: {result['answer']}")
        print(f"Source: {result['source']}")
        print(f"Confidence: {result['confidence']}")

    elif args.mode == "interactive":
        print("RAG Project 2 - Interactive Mode")
        print("Commands:")
        print("  ingest <doc_path> <doc_id> [semantic|parent_child]")
        print("  query <question>")
        print("  quit")
        print()

        while True:
            try:
                command = input("rag> ").strip()
                if not command:
                    continue

                parts = command.split()
                cmd = parts[0].lower()

                if cmd == "quit":
                    break

                elif cmd == "ingest":
                    if len(parts) < 3:
                        print("Usage: ingest <doc_path> <doc_id> [chunking_type]")
                        continue

                    doc_path = parts[1]
                    doc_id = parts[2]
                    chunking = parts[3] if len(parts) > 3 else "semantic"

                    if not Path(doc_path).exists():
                        print(f"Document not found: {doc_path}")
                        continue

                    success = app.ingest_document(
                        document_path=doc_path,
                        document_id=doc_id,
                        use_semantic_chunking=(chunking == "semantic"),
                    )

                    print("✓ Document ingested" if success else "✗ Ingestion failed")

                elif cmd == "query":
                    if len(parts) < 2:
                        print("Usage: query <question>")
                        continue

                    question = " ".join(parts[1:])
                    result = app.query(question)

                    print(f"Answer: {result['answer']}")
                    print(f"Source: {result['source']}")

                else:
                    print(f"Unknown command: {cmd}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Interactive mode error: {e}")
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
