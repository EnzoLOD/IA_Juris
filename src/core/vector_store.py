"""
Advanced Vector Store implementation for JurisOracle.
Supports multiple backends with optimized semantic search for legal documents.
"""

import asyncio
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
import faiss
import numpy as np
import pandas as pd
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field

from ..config.settings import get_settings
from ..utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)
settings = get_settings()

# ========================================================================================
# MODELS AND CONFIGURATIONS
# ========================================================================================

@dataclass
class VectorStoreConfig:
    """Configuration for vector store operations."""
    backend: str = "chroma"  # chroma, faiss, pinecone
    collection_name: str = "juris_oracle_docs"
    dimension: int = 384  # sentence-transformers dimension
    distance_metric: str = "cosine"  # cosine, euclidean, dot_product
    
    # Chroma specific
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_persist_directory: str = "./data/chroma_db"
    
    # FAISS specific
    faiss_index_type: str = "IndexFlatIP"  # IndexFlatIP, IndexIVFFlat, IndexHNSW
    faiss_nlist: int = 100  # for IVF indices
    faiss_nprobe: int = 10  # for search
    
    # Performance
    batch_size: int = 1000
    max_results: int = 50
    similarity_threshold: float = 0.7
    
    # Cache
    enable_cache: bool = True
    cache_size: int = 10000

class DocumentMetadata(BaseModel):
    """Metadata structure for legal documents."""
    document_id: str
    title: str = ""
    content_type: str = "legal_document"  # jurisprudencia, lei, artigo, etc.
    tribunal: str = ""
    numero_processo: str = ""
    data_julgamento: Optional[str] = None
    relator: str = ""
    classe_processual: str = ""
    assunto: str = ""
    ementa: str = ""
    texto_completo: str = ""
    chunk_index: int = 0
    total_chunks: int = 1
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = Field(default_factory=list)
    confidence_score: float = 1.0

class SearchResult(BaseModel):
    """Result from vector search."""
    document_id: str
    content: str
    metadata: DocumentMetadata
    similarity_score: float
    rank: int

class SearchQuery(BaseModel):
    """Search query with filters."""
    query_text: str
    filters: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 10
    similarity_threshold: float = 0.7
    include_metadata: bool = True

# ========================================================================================
# ABSTRACT BASE CLASS
# ========================================================================================

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    async def add_documents(self, documents: List[str], embeddings: List[List[float]], 
                          metadata: List[DocumentMetadata]) -> List[str]:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: List[float], 
                    query: SearchQuery) -> List[SearchResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the store."""
        pass
    
    @abstractmethod
    async def update_document(self, document_id: str, content: str, 
                            embedding: List[float], metadata: DocumentMetadata) -> bool:
        """Update a document."""
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[SearchResult]:
        """Get a specific document."""
        pass

# ========================================================================================
# CHROMA IMPLEMENTATION
# ========================================================================================

class ChromaVectorStore(BaseVectorStore):
    """ChromaDB implementation of vector store."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.client = None
        self.collection = None
        self.metrics = MetricsCollector()
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client."""
        try:
            # Setup for persistent storage
            if self.config.chroma_persist_directory:
                os.makedirs(self.config.chroma_persist_directory, exist_ok=True)
                
                chroma_settings = Settings(
                    persist_directory=self.config.chroma_persist_directory,
                    anonymized_telemetry=False
                )
                self.client = chromadb.PersistentClient(settings=chroma_settings)
            else:
                # In-memory client
                self.client = chromadb.Client()
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name
                )
                logger.info(f"Connected to existing collection: {self.config.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    metadata={"description": "JurisOracle legal documents collection"}
                )
                logger.info(f"Created new collection: {self.config.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    async def add_documents(self, documents: List[str], embeddings: List[List[float]], 
                          metadata: List[DocumentMetadata]) -> List[str]:
        """Add documents to ChromaDB."""
        try:
            start_time = datetime.now()
            
            if len(documents) != len(embeddings) != len(metadata):
                raise ValueError("Documents, embeddings, and metadata lists must have same length")
            
            # Generate IDs if not provided
            ids = [meta.document_id if meta.document_id else str(uuid.uuid4()) 
                   for meta in metadata]
            
            # Convert metadata to dict format
            metadata_dicts = [meta.dict() for meta in metadata]
            
            # Add to collection in batches
            batch_size = self.config.batch_size
            added_ids = []
            
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                
                batch_ids = ids[i:batch_end]
                batch_documents = documents[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                batch_metadata = metadata_dicts[i:batch_end]
                
                # Execute in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self.collection.add(
                        ids=batch_ids,
                        documents=batch_documents,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadata
                    )
                )
                
                added_ids.extend(batch_ids)
                logger.info(f"Added batch {i//batch_size + 1}, total docs: {len(added_ids)}")
            
            # Track metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.record_operation("add_documents", duration, len(documents))
            
            logger.info(f"Successfully added {len(added_ids)} documents to ChromaDB")
            return added_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {str(e)}")
            raise
    
    async def search(self, query_embedding: List[float], 
                    query: SearchQuery) -> List[SearchResult]:
        """Search ChromaDB for similar documents."""
        try:
            start_time = datetime.now()
            
            # Prepare where clause for filtering
            where_clause = {}
            if query.filters:
                # Convert filters to ChromaDB format
                for key, value in query.filters.items():
                    if isinstance(value, list):
                        where_clause[key] = {"$in": value}
                    else:
                        where_clause[key] = value
            
            # Execute search in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=query.top_k,
                    where=where_clause if where_clause else None,
                    include=["documents", "metadatas", "distances"]
                )
            )
            
            # Process results
            search_results = []
            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                
                for i, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances)):
                    # Convert distance to similarity score (cosine similarity)
                    similarity_score = 1 - distance
                    
                    # Filter by similarity threshold
                    if similarity_score >= query.similarity_threshold:
                        search_result = SearchResult(
                            document_id=meta.get("document_id", f"doc_{i}"),
                            content=doc,
                            metadata=DocumentMetadata(**meta),
                            similarity_score=similarity_score,
                            rank=i + 1
                        )
                        search_results.append(search_result)
            
            # Track metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.record_operation("search", duration, len(search_results))
            
            logger.info(f"ChromaDB search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {str(e)}")
            raise
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from ChromaDB."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.collection.delete(ids=document_ids)
            )
            
            logger.info(f"Deleted {len(document_ids)} documents from ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {str(e)}")
            return False
    
    async def update_document(self, document_id: str, content: str, 
                            embedding: List[float], metadata: DocumentMetadata) -> bool:
        """Update a document in ChromaDB."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.collection.update(
                    ids=[document_id],
                    documents=[content],
                    embeddings=[embedding],
                    metadatas=[metadata.dict()]
                )
            )
            
            logger.info(f"Updated document {document_id} in ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document in ChromaDB: {str(e)}")
            return False
    
    async def get_document(self, document_id: str) -> Optional[SearchResult]:
        """Get a specific document from ChromaDB."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.collection.get(
                    ids=[document_id],
                    include=["documents", "metadatas"]
                )
            )
            
            if result["documents"] and result["documents"][0]:
                doc = result["documents"][0]
                meta = result["metadatas"][0]
                
                return SearchResult(
                    document_id=document_id,
                    content=doc,
                    metadata=DocumentMetadata(**meta),
                    similarity_score=1.0,
                    rank=1
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document from ChromaDB: {str(e)}")
            return None

# ========================================================================================
# FAISS IMPLEMENTATION
# ========================================================================================

class FAISSVectorStore(BaseVectorStore):
    """FAISS implementation of vector store."""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.index = None
        self.document_store = {}  # Store documents and metadata
        self.id_to_index = {}  # Map document IDs to FAISS indices
        self.index_to_id = {}  # Map FAISS indices to document IDs
        self.metrics = MetricsCollector()
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index."""
        try:
            if self.config.faiss_index_type == "IndexFlatIP":
                # Inner product (for cosine similarity with normalized vectors)
                self.index = faiss.IndexFlatIP(self.config.dimension)
            elif self.config.faiss_index_type == "IndexFlatL2":
                # L2 distance
                self.index = faiss.IndexFlatL2(self.config.dimension)
            elif self.config.faiss_index_type == "IndexIVFFlat":
                # IVF with flat quantizer
                quantizer = faiss.IndexFlatIP(self.config.dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, self.config.dimension, self.config.faiss_nlist
                )
            elif self.config.faiss_index_type == "IndexHNSW":
                # HNSW index for fast approximate search
                self.index = faiss.IndexHNSWFlat(self.config.dimension, 32)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 100
            else:
                raise ValueError(f"Unsupported FAISS index type: {self.config.faiss_index_type}")
            
            logger.info(f"Initialized FAISS index: {self.config.faiss_index_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {str(e)}")
            raise
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)
    
    async def add_documents(self, documents: List[str], embeddings: List[List[float]], 
                          metadata: List[DocumentMetadata]) -> List[str]:
        """Add documents to FAISS index."""
        try:
            start_time = datetime.now()
            
            if len(documents) != len(embeddings) != len(metadata):
                raise ValueError("Documents, embeddings, and metadata lists must have same length")
            
            # Convert embeddings to numpy array
            embedding_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize for cosine similarity
            if self.config.distance_metric == "cosine":
                embedding_array = self._normalize_embeddings(embedding_array)
            
            # Train index if needed (for IVF indices)
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                if embedding_array.shape[0] >= self.config.faiss_nlist:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.index.train, embedding_array
                    )
                    logger.info("FAISS index trained")
            
            # Get starting index for new documents
            start_idx = self.index.ntotal
            
            # Add embeddings to index
            await asyncio.get_event_loop().run_in_executor(
                None, self.index.add, embedding_array
            )
            
            # Store documents and metadata
            added_ids = []
            for i, (doc, meta) in enumerate(zip(documents, metadata)):
                doc_id = meta.document_id if meta.document_id else str(uuid.uuid4())
                faiss_idx = start_idx + i
                
                self.document_store[doc_id] = {
                    'content': doc,
                    'metadata': meta,
                    'faiss_index': faiss_idx
                }
                
                self.id_to_index[doc_id] = faiss_idx
                self.index_to_id[faiss_idx] = doc_id
                added_ids.append(doc_id)
            
            # Track metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.record_operation("add_documents", duration, len(documents))
            
            logger.info(f"Successfully added {len(added_ids)} documents to FAISS")
            return added_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {str(e)}")
            raise
    
    async def search(self, query_embedding: List[float], 
                    query: SearchQuery) -> List[SearchResult]:
        """Search FAISS index for similar documents."""
        try:
            start_time = datetime.now()
            
            # Convert query embedding to numpy array
            query_array = np.array([query_embedding], dtype=np.float32)
            
            # Normalize for cosine similarity
            if self.config.distance_metric == "cosine":
                query_array = self._normalize_embeddings(query_array)
            
            # Set search parameters for IVF indices
            if hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.config.faiss_nprobe
            
            # Perform search
            scores, indices = await asyncio.get_event_loop().run_in_executor(
                None, self.index.search, query_array, query.top_k * 2  # Get more to allow filtering
            )
            
            # Process results
            search_results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for missing results
                    continue
                
                # Get document ID
                doc_id = self.index_to_id.get(idx)
                if not doc_id:
                    continue
                
                # Get document data
                doc_data = self.document_store.get(doc_id)
                if not doc_data:
                    continue
                
                # Convert score to similarity
                if self.config.distance_metric == "cosine":
                    similarity_score = float(score)  # Already similarity for normalized vectors
                else:
                    similarity_score = 1.0 / (1.0 + float(score))  # Convert distance to similarity
                
                # Apply similarity threshold
                if similarity_score < query.similarity_threshold:
                    continue
                
                # Apply metadata filters
                if query.filters and not self._matches_filters(doc_data['metadata'], query.filters):
                    continue
                
                search_result = SearchResult(
                    document_id=doc_id,
                    content=doc_data['content'],
                    metadata=doc_data['metadata'],
                    similarity_score=similarity_score,
                    rank=len(search_results) + 1
                )
                search_results.append(search_result)
                
                # Stop when we have enough results
                if len(search_results) >= query.top_k:
                    break
            
            # Track metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.record_operation("search", duration, len(search_results))
            
            logger.info(f"FAISS search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching FAISS: {str(e)}")
            raise
    
    def _matches_filters(self, metadata: DocumentMetadata, filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the given filters."""
        for key, value in filters.items():
            meta_value = getattr(metadata, key, None)
            if meta_value is None:
                return False
            
            if isinstance(value, list):
                if meta_value not in value:
                    return False
            elif meta_value != value:
                return False
        
        return True
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from FAISS store."""
        try:
            # FAISS doesn't support deletion, so we mark as deleted
            deleted_count = 0
            for doc_id in document_ids:
                if doc_id in self.document_store:
                    faiss_idx = self.document_store[doc_id]['faiss_index']
                    
                    # Remove from our mappings
                    del self.document_store[doc_id]
                    del self.id_to_index[doc_id]
                    if faiss_idx in self.index_to_id:
                        del self.index_to_id[faiss_idx]
                    
                    deleted_count += 1
            
            logger.info(f"Marked {deleted_count} documents as deleted in FAISS")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents from FAISS: {str(e)}")
            return False
    
    async def update_document(self, document_id: str, content: str, 
                            embedding: List[float], metadata: DocumentMetadata) -> bool:
        """Update a document in FAISS store."""
        try:
            if document_id not in self.document_store:
                return False
            
            # Update document store
            self.document_store[document_id]['content'] = content
            self.document_store[document_id]['metadata'] = metadata
            
            # For FAISS, we would need to rebuild the index to update embeddings
            # For now, just update the stored data
            logger.info(f"Updated document {document_id} in FAISS store")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document in FAISS: {str(e)}")
            return False
    
    async def get_document(self, document_id: str) -> Optional[SearchResult]:
        """Get a specific document from FAISS store."""
        try:
            doc_data = self.document_store.get(document_id)
            if not doc_data:
                return None
            
            return SearchResult(
                document_id=document_id,
                content=doc_data['content'],
                metadata=doc_data['metadata'],
                similarity_score=1.0,
                rank=1
            )
            
        except Exception as e:
            logger.error(f"Error getting document from FAISS: {str(e)}")
            return None

# ========================================================================================
# UNIFIED VECTOR STORE
# ========================================================================================

class VectorStore:
    """Unified vector store interface supporting multiple backends."""
    
    def __init__(self, config: VectorStoreConfig = None):
        self.config = config or VectorStoreConfig()
        self.store: BaseVectorStore = None
        self.metrics = MetricsCollector()
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the appropriate vector store backend."""
        try:
            if self.config.backend.lower() == "chroma":
                self.store = ChromaVectorStore(self.config)
            elif self.config.backend.lower() == "faiss":
                self.store = FAISSVectorStore(self.config)
            else:
                raise ValueError(f"Unsupported vector store backend: {self.config.backend}")
            
            logger.info(f"Initialized {self.config.backend} vector store")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise
    
    async def add_documents(self, documents: List[str], embeddings: List[List[float]], 
                          metadata: List[DocumentMetadata]) -> List[str]:
        """Add documents to the vector store."""
        return await self.store.add_documents(documents, embeddings, metadata)
    
    async def search(self, query_embedding: List[float], 
                    query: SearchQuery) -> List[SearchResult]:
        """Search for similar documents."""
        return await self.store.search(query_embedding, query)
    
    async def hybrid_search(self, query_text: str, query_embedding: List[float],
                          filters: Dict[str, Any] = None, top_k: int = 10) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword search."""
        try:
            # Perform semantic search
            search_query = SearchQuery(
                query_text=query_text,
                filters=filters or {},
                top_k=top_k,
                similarity_threshold=self.config.similarity_threshold
            )
            
            semantic_results = await self.search(query_embedding, search_query)
            
            # TODO: Add keyword search and combine results
            # For now, return semantic results
            return semantic_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from the store."""
        return await self.store.delete_documents(document_ids)
    
    async def update_document(self, document_id: str, content: str, 
                            embedding: List[float], metadata: DocumentMetadata) -> bool:
        """Update a document."""
        return await self.store.update_document(document_id, content, embedding, metadata)
    
    async def get_document(self, document_id: str) -> Optional[SearchResult]:
        """Get a specific document."""
        return await self.store.get_document(document_id)
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            if isinstance(self.store, ChromaVectorStore):
                # Get ChromaDB stats
                count = self.store.collection.count()
                return {
                    "backend": "chroma",
                    "total_documents": count,
                    "collection_name": self.config.collection_name
                }
            elif isinstance(self.store, FAISSVectorStore):
                # Get FAISS stats
                return {
                    "backend": "faiss",
                    "total_documents": len(self.store.document_store),
                    "index_size": self.store.index.ntotal,
                    "dimension": self.config.dimension
                }
            
            return {"backend": self.config.backend}
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
    
    async def optimize_index(self) -> bool:
        """Optimize the vector index for better performance."""
        try:
            if isinstance(self.store, FAISSVectorStore):
                # For FAISS, we could rebuild with optimized parameters
                logger.info("FAISS index optimization not implemented yet")
                return True
            
            logger.info(f"Index optimization not needed for {self.config.backend}")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing index: {str(e)}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics.get_metrics()

# ========================================================================================
# UTILITY FUNCTIONS
# ========================================================================================

def create_vector_store(backend: str = "chroma", **kwargs) -> VectorStore:
    """Factory function to create a vector store."""
    config = VectorStoreConfig(backend=backend, **kwargs)
    return VectorStore(config)

async def migrate_vector_store(source_store: VectorStore, target_store: VectorStore,
                             batch_size: int = 1000) -> bool:
    """Migrate data from one vector store to another."""
    try:
        logger.info("Starting vector store migration...")
        
        # This would require implementing a way to export all data from source
        # and import to target. Implementation depends on specific backends.
        
        logger.warning("Vector store migration not fully implemented yet")
        return False
        
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        return False

# ========================================================================================
# EXAMPLE USAGE
# ========================================================================================

if __name__ == "__main__":
    async def main():
        # Initialize vector store
        config = VectorStoreConfig(
            backend="chroma",
            collection_name="test_collection",
            dimension=384
        )
        
        store = VectorStore(config)
        
        # Sample data
        documents = [
            "Artigo 5º da Constituição Federal garante direitos fundamentais.",
            "O Código Civil brasileiro foi instituído pela Lei 10.406/2002.",
            "Habeas corpus é um remédio constitucional previsto no art. 5º, LXVIII."
        ]
        
        # Mock embeddings (in real usage, these would come from an embedding model)
        embeddings = [
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384
        ]
        
        metadata = [
            DocumentMetadata(
                document_id="doc1",
                title="Direitos Fundamentais",
                content_type="lei",
                assunto="direitos_fundamentais"
            ),
            DocumentMetadata(
                document_id="doc2", 
                title="Código Civil",
                content_type="lei",
                assunto="direito_civil"
            ),
            DocumentMetadata(
                document_id="doc3",
                title="Habeas Corpus",
                content_type="jurisprudencia",
                assunto="direito_processual"
            )
        ]
        
        # Add documents
        ids = await store.add_documents(documents, embeddings, metadata)
        print(f"Added documents: {ids}")
        
        # Search
        query = SearchQuery(
            query_text="direitos fundamentais",
            top_k=2,
            filters={"content_type": "lei"}
        )
        
        results = await store.search([0.15] * 384, query)
        print(f"Search results: {len(results)}")
        
        for result in results:
            print(f"- {result.document_id}: {result.similarity_score:.3f}")
        
        # Get stats
        stats = await store.get_collection_stats()
        print(f"Collection stats: {stats}")
    
    asyncio.run(main())