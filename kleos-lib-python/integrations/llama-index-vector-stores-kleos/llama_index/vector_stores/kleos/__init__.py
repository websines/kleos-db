"""LlamaIndex vector store integration for kleos."""

from llama_index.vector_stores.kleos.base import KleosRetriever, KleosVectorStore

__all__ = ["KleosVectorStore", "KleosRetriever"]
__version__ = "0.2.1"
