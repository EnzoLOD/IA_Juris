"""
Modelos base para o sistema JurisOracle.
"""
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, DateTime, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class TimestampMixin:
    """Mixin para adicionar timestamps aos modelos."""
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class BaseEntity(Base, TimestampMixin):
    """Classe base para todas as entidades do banco."""
    __abstract__ = True
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class BaseSchema(BaseModel):
    """Schema base para validação de dados."""
    
    class Config:
        orm_mode = True
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True


class ResponseSchema(BaseSchema):
    """Schema padrão para respostas da API."""
    success: bool = True
    message: str = "Operação realizada com sucesso"
    data: Optional[Any] = None
    errors: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginationSchema(BaseSchema):
    """Schema para paginação."""
    page: int = Field(1, ge=1, description="Número da página")
    size: int = Field(10, ge=1, le=100, description="Itens por página")
    total: Optional[int] = Field(None, description="Total de itens")
    pages: Optional[int] = Field(None, description="Total de páginas")


class ErrorSchema(BaseSchema):
    """Schema para erros."""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
