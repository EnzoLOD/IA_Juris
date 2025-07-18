"""
Modelos para queries e respostas do sistema.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, String, Text, Float, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from .base import BaseEntity, BaseSchema


class QueryType(str, Enum):
    """Tipos de query suportados."""
    QUESTION_ANSWERING = "qa"
    SUMMARIZATION = "summarization"
    SEARCH = "search"
    ANALYSIS = "analysis"


class QueryStatus(str, Enum):
    """Status da query."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Query(BaseEntity):
    """Modelo para queries no banco."""
    __tablename__ = "queries"
    
    user_id = Column(String(100))  # Para futura implementação de usuários
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50), nullable=False)
    
    # Configurações da query
    parameters = Column(JSON)
    
    # Resultados
    response_text = Column(Text)
    confidence_score = Column(Float)
    processing_time = Column(Float)  # em segundos
    
    # Status
    status = Column(String(50), default=QueryStatus.PENDING)
    error_message = Column(Text)
    
    # Metadados
    metadata = Column(JSON)
    
    # Relacionamentos
    query_documents = relationship("QueryDocument", back_populates="query")


class QueryDocument(BaseEntity):
    """Relacionamento entre queries e documentos utilizados."""
    __tablename__ = "query_documents"
    
    query_id = Column(UUID(as_uuid=True), ForeignKey("queries.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    relevance_score = Column(Float)
    chunks_used = Column(JSON)  # IDs dos chunks utilizados
    
    # Relacionamentos
    query = relationship("Query", back_populates="query_documents")


# Schemas Pydantic

class QueryRequest(BaseSchema):
    """Schema para requisição de query."""
    query: str = Field(..., min_length=3, max_length=2000)
    query_type: QueryType = QueryType.QUESTION_ANSWERING
    document_ids: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query não pode estar vazia')
        return v.strip()


class HyDEQueryRequest(QueryRequest):
    """Schema específico para queries com HyDE."""
    use_hyde: bool = True
    hyde_iterations: int = Field(1, ge=1, le=3)
    hyde_temperature: float = Field(0.7, ge=0.0, le=2.0)


class QueryResponse(BaseSchema):
    """Schema para resposta de query."""
    query_id: str
    response: str
    confidence_score: Optional[float] = None
    processing_time: float
    sources: List[Dict[str, Any]] = []
    metadata: Optional[Dict[str, Any]] = None


class QueryAnalytics(BaseSchema):
    """Schema para analytics de queries."""
    total_queries: int
    avg_processing_time: float
    success_rate: float
    most_common_types: List[Dict[str, Any]]
    recent_queries: List[Dict[str, Any]]


class SummarizationRequest(BaseSchema):
    """Schema para requisição de resumo."""
    document_ids: List[str] = Field(..., min_items=1)
    summary_type: str = Field("comprehensive", regex="^(brief|comprehensive|detailed)$")
    max_length: int = Field(500, ge=100, le=2000)
    focus_areas: Optional[List[str]] = None


class SummarizationResponse(BaseSchema):
    """Schema para resposta de resumo."""
    summary: str
    document_count: int
    key_points: List[str]
    confidence_score: float
    processing_time: float


class AnalysisRequest(BaseSchema):
    """Schema para requisição de análise."""
    document_ids: List[str] = Field(..., min_items=1)
    analysis_type: str = Field("legal", regex="^(legal|sentiment|entities|topics)$")
    parameters: Optional[Dict[str, Any]] = None


class AnalysisResponse(BaseSchema):
    """Schema para resposta de análise."""
    analysis_type: str
    results: Dict[str, Any]
    insights: List[str]
    confidence_score: float
    processing_time: float


class QueryHistoryRequest(BaseSchema):
    """Schema para histórico de queries."""
    user_id: Optional[str] = None
    query_type: Optional[QueryType] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(50, ge=1, le=200)


class QueryHistoryItem(BaseSchema):
    """Item do histórico de queries."""
    id: str
    query_text: str
    query_type: QueryType
    status: QueryStatus
    confidence_score: Optional[float] = None
    processing_time: Optional[float] = None
    created_at: datetime
    
    class Config:
        from_attributes = True