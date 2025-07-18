"""
Modelos para documentos jurídicos.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from .base import BaseEntity, BaseSchema


class DocumentType(str, Enum):
    """Tipos de documentos suportados."""
    PROCESSO = "processo"
    SENTENCA = "sentenca"
    ACORDAO = "acordao"
    PETICAO = "peticao"
    CONTRATO = "contrato"
    LEI = "lei"
    DECRETO = "decreto"
    OUTROS = "outros"


class ProcessingStatus(str, Enum):
    """Status de processamento do documento."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(BaseEntity):
    """Modelo de documento no banco de dados."""
    __tablename__ = "documents"
    
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)
    
    title = Column(String(500))
    document_type = Column(String(50), nullable=False, default=DocumentType.OUTROS)
    content = Column(Text)
    summary = Column(Text)
    
    processing_status = Column(String(50), default=ProcessingStatus.PENDING)
    processing_error = Column(Text)
    
    # Metadados extraídos
    metadata = Column(JSON)
    
    # Informações de embedding
    embedding_model = Column(String(100))
    chunk_count = Column(Integer, default=0)
    
    # Relacionamentos
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(BaseEntity):
    """Modelo para chunks de documentos."""
    __tablename__ = "document_chunks"
    
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    
    # Embedding vector (armazenado como JSON para compatibilidade)
    embedding = Column(JSON)
    
    # Metadados do chunk
    start_char = Column(Integer)
    end_char = Column(Integer)
    page_number = Column(Integer)
    
    # Relacionamentos
    document = relationship("Document", back_populates="chunks")


# Schemas Pydantic

class DocumentCreateSchema(BaseSchema):
    """Schema para criação de documento."""
    title: Optional[str] = None
    document_type: DocumentType = DocumentType.OUTROS
    metadata: Optional[Dict[str, Any]] = None


class DocumentUpdateSchema(BaseSchema):
    """Schema para atualização de documento."""
    title: Optional[str] = None
    document_type: Optional[DocumentType] = None
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentChunkSchema(BaseSchema):
    """Schema para chunk de documento."""
    id: str
    chunk_index: int
    content: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    page_number: Optional[int] = None
    
    class Config:
        from_attributes = True


class DocumentSchema(BaseSchema):
    """Schema completo do documento."""
    id: str
    filename: str
    original_filename: str
    file_size: int
    mime_type: str
    title: Optional[str] = None
    document_type: DocumentType
    content: Optional[str] = None
    summary: Optional[str] = None
    processing_status: ProcessingStatus
    processing_error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embedding_model: Optional[str] = None
    chunk_count: int
    created_at: datetime
    updated_at: datetime
    chunks: Optional[List[DocumentChunkSchema]] = None
    
    class Config:
        from_attributes = True


class DocumentListSchema(BaseSchema):
    """Schema para listagem de documentos."""
    id: str
    filename: str
    original_filename: str
    title: Optional[str] = None
    document_type: DocumentType
    processing_status: ProcessingStatus
    file_size: int
    chunk_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentUploadResponse(BaseSchema):
    """Schema para resposta de upload."""
    document_id: str
    filename: str
    message: str = "Documento enviado com sucesso"


class DocumentProcessingRequest(BaseSchema):
    """Schema para requisição de processamento."""
    document_id: str
    force_reprocess: bool = False
    extract_metadata: bool = True
    generate_summary: bool = True


class DocumentSearchRequest(BaseSchema):
    """Schema para busca de documentos."""
    query: str = Field(..., min_length=3, max_length=500)
    document_types: Optional[List[DocumentType]] = None
    limit: int = Field(10, ge=1, le=50)
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query não pode estar vazia')
        return v.strip()


class DocumentSearchResult(BaseSchema):
    """Schema para resultado de busca."""
    document_id: str
    title: Optional[str] = None
    filename: str
    document_type: DocumentType
    similarity_score: float
    relevant_chunks: List[DocumentChunkSchema]
    summary: Optional[str] = None