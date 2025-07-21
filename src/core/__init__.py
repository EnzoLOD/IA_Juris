# This file marks the core directory as a Python package.
"""
Core module para JurisOracle
Componentes fundamentais do sistema de análise jurídica com IA
"""

from .hyde_retriever import HydeRetriever
from .embeddings import EmbeddingManager
from ..services.document_processor import DocumentProcessor
from .vector_store import VectorStore

__all__ = [
    'HydeRetriever',
    'EmbeddingManager',
    'DocumentProcessor',
    'VectorStore'
]

__version__ = "1.0.0"
