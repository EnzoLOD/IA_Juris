"""
documents.py - Módulo de Gestão de Documentos Jurídicos

Este módulo fornece funcionalidades completas para o gerenciamento de documentos
jurídicos no sistema JurisOracle, incluindo operações CRUD, processamento de texto,
análise de conteúdo e integração com IA para extração de insights jurídicos.

Funcionalidades principais:
- Gestão completa de documentos jurídicos (CRUD)
- Processamento e análise de conteúdo
- Extração de metadados jurídicos
- Integração com sistema de embeddings
- Versionamento de documentos
- Auditoria e logs de atividades

Dependências:
- SQLAlchemy (ORM)
- FastAPI (Framework web)
- Pydantic (Validação de dados)
- PyPDF2/pdfplumber (Processamento PDF)
- python-magic (Detecção de tipo de arquivo)

Autor: Sistema JurisOracle
Versão: 1.0.0
Data: 2025-01-11
"""

import os
import hashlib
import mimetypes
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path
import logging
import asyncio
from contextlib import asynccontextmanager

# FastAPI e Pydantic
from fastapi import HTTPException, UploadFile, status
from pydantic import BaseModel, Field, validator, root_validator

# SQLAlchemy
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float,
    ForeignKey, Index, event, and_, or_, func, desc, asc
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session, sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

# Utilitários
import uuid
import json
import magic
import PyPDF2
import pdfplumber
from io import BytesIO

# Configuração de logging
logger = logging.getLogger(__name__)

# ================================
# MODELOS DE DADOS (SQLAlchemy)
# ================================

Base = declarative_base()

class Document(Base):
    """
    Modelo SQLAlchemy para documentos jurídicos.
    
    Representa um documento no sistema com todos os metadados,
    conteúdo processado e informações de auditoria.
    """
    __tablename__ = "documents"
    
    # Campos principais
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False, index=True)
    description = Column(Text)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True, index=True)
    mime_type = Column(String(100), nullable=False)
    
    # Conteúdo e processamento
    content_text = Column(Text)
    content_summary = Column(Text)
    page_count = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    
    # Metadados jurídicos
    document_type = Column(String(100), index=True)  # Petição, Sentença, Contrato, etc.
    legal_area = Column(String(100), index=True)     # Civil, Penal, Trabalhista, etc.
    court_level = Column(String(50))                 # STF, STJ, TRF, etc.
    case_number = Column(String(100), index=True)
    parties_involved = Column(ARRAY(String))
    keywords = Column(ARRAY(String))
    legal_topics = Column(ARRAY(String))
    
    # Status e processamento
    processing_status = Column(String(50), default="pending", index=True)
    is_processed = Column(Boolean, default=False, index=True)
    is_active = Column(Boolean, default=True, index=True)
    is_public = Column(Boolean, default=False, index=True)
    
    # Embeddings e IA
    embedding_vector = Column(ARRAY(Float))
    ai_analysis = Column(JSONB)
    confidence_score = Column(Float, default=0.0)
    
    # Auditoria
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    updated_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Versionamento
    version = Column(Integer, default=1)
    parent_document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"))
    
    # Relacionamentos
    creator = relationship("User", foreign_keys=[created_by], back_populates="created_documents")
    updater = relationship("User", foreign_keys=[updated_by])
    versions = relationship("Document", remote_side=[id], back_populates="parent_document")
    parent_document = relationship("Document", remote_side=[id], back_populates="versions")
    
    # Índices compostos para otimização
    __table_args__ = (
        Index('idx_doc_type_area', 'document_type', 'legal_area'),
        Index('idx_doc_status_active', 'processing_status', 'is_active'),
        Index('idx_doc_created_date', 'created_at'),
        Index('idx_doc_search', 'title', 'description', 'keywords'),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title[:50]}...', type='{self.document_type}')>"

class DocumentVersion(Base):
    """Modelo para controle de versões de documentos."""
    __tablename__ = "document_versions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    changes_description = Column(Text)
    previous_content = Column(JSONB)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    document = relationship("Document", back_populates="document_versions")
    creator = relationship("User")

class DocumentActivity(Base):
    """Modelo para auditoria de atividades em documentos."""
    __tablename__ = "document_activities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    activity_type = Column(String(50), nullable=False)  # CREATE, UPDATE, DELETE, VIEW, DOWNLOAD
    activity_description = Column(Text)
    metadata = Column(JSONB)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    
    document = relationship("Document")
    user = relationship("User")

# ================================
# MODELOS PYDANTIC (Validação)
# ================================

class DocumentBase(BaseModel):
    """Schema base para documentos."""
    title: str = Field(..., min_length=1, max_length=500, description="Título do documento")
    description: Optional[str] = Field(None, max_length=2000, description="Descrição do documento")
    document_type: Optional[str] = Field(None, max_length=100, description="Tipo do documento jurídico")
    legal_area: Optional[str] = Field(None, max_length=100, description="Área jurídica")
    court_level: Optional[str] = Field(None, max_length=50, description="Instância do tribunal")
    case_number: Optional[str] = Field(None, max_length=100, description="Número do processo")
    parties_involved: Optional[List[str]] = Field(default_factory=list, description="Partes envolvidas")
    keywords: Optional[List[str]] = Field(default_factory=list, description="Palavras-chave")
    legal_topics: Optional[List[str]] = Field(default_factory=list, description="Tópicos jurídicos")
    is_public: bool = Field(default=False, description="Documento público")
    
    @validator('title')
    def validate_title(cls, v):
        if not v or not v.strip():
            raise ValueError('Título não pode estar vazio')
        return v.strip()
    
    @validator('keywords', 'legal_topics', 'parties_involved')
    def validate_arrays(cls, v):
        if v is None:
            return []
        return [item.strip() for item in v if item and item.strip()]

class DocumentCreate(DocumentBase):
    """Schema para criação de documentos."""
    pass

class DocumentUpdate(BaseModel):
    """Schema para atualização de documentos."""
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=2000)
    document_type: Optional[str] = Field(None, max_length=100)
    legal_area: Optional[str] = Field(None, max_length=100)
    court_level: Optional[str] = Field(None, max_length=50)
    case_number: Optional[str] = Field(None, max_length=100)
    parties_involved: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    legal_topics: Optional[List[str]] = None
    is_public: Optional[bool] = None

class DocumentResponse(DocumentBase):
    """Schema para resposta de documentos."""
    id: uuid.UUID
    file_name: str
    file_size: int
    mime_type: str
    page_count: int
    word_count: int
    processing_status: str
    is_processed: bool
    is_active: bool
    confidence_score: float
    version: int
    created_at: datetime
    updated_at: datetime
    created_by: Optional[uuid.UUID]
    
    class Config:
        from_attributes = True

class DocumentSearchFilters(BaseModel):
    """Filtros para busca de documentos."""
    query: Optional[str] = None
    document_type: Optional[str] = None
    legal_area: Optional[str] = None
    court_level: Optional[str] = None
    case_number: Optional[str] = None
    keywords: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    is_public: Optional[bool] = None
    is_processed: Optional[bool] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
class DocumentSearchResponse(BaseModel):
    """Resposta para busca de documentos."""
    documents: List[DocumentResponse]
    total: int
    page: int
    per_page: int
    total_pages: int

# ================================
# EXCEÇÕES CUSTOMIZADAS
# ================================

class DocumentError(Exception):
    """Exceção base para erros de documento."""
    pass

class DocumentNotFoundError(DocumentError):
    """Documento não encontrado."""
    pass

class DocumentProcessingError(DocumentError):
    """Erro no processamento do documento."""
    pass

class DocumentValidationError(DocumentError):
    """Erro de validação do documento."""
    pass

class DocumentStorageError(DocumentError):
    """Erro no armazenamento do documento."""
    pass

# ================================
# UTILITÁRIOS DE PROCESSAMENTO
# ================================

class DocumentProcessor:
    """Classe para processamento de documentos."""
    
    SUPPORTED_MIME_TYPES = {
        'application/pdf': 'pdf',
        'text/plain': 'txt',
        'application/msword': 'doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'text/html': 'html',
        'application/rtf': 'rtf'
    }
    
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    @staticmethod
    def validate_file(file: UploadFile) -> Tuple[bool, str]:
        """
        Valida se o arquivo é suportado.
        
        Args:
            file: Arquivo para validação
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # Verificar tamanho
            if hasattr(file, 'size') and file.size > DocumentProcessor.MAX_FILE_SIZE:
                return False, f"Arquivo muito grande. Máximo permitido: {DocumentProcessor.MAX_FILE_SIZE // (1024*1024)}MB"
            
            # Verificar tipo MIME
            if file.content_type not in DocumentProcessor.SUPPORTED_MIME_TYPES:
                return False, f"Tipo de arquivo não suportado: {file.content_type}"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Erro na validação do arquivo: {str(e)}")
            return False, "Erro na validação do arquivo"
    
    @staticmethod
    def calculate_file_hash(content: bytes) -> str:
        """Calcula hash SHA-256 do conteúdo do arquivo."""
        return hashlib.sha256(content).hexdigest()
    
    @staticmethod
    def extract_text_from_pdf(content: bytes) -> Tuple[str, int]:
        """
        Extrai texto de arquivo PDF.
        
        Args:
            content: Conteúdo do arquivo PDF
            
        Returns:
            Tuple[str, int]: (texto_extraido, numero_paginas)
        """
        try:
            text_content = ""
            page_count = 0
            
            # Tentar com pdfplumber primeiro (melhor para tabelas)
            try:
                with pdfplumber.open(BytesIO(content)) as pdf:
                    page_count = len(pdf.pages)
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
            except Exception:
                # Fallback para PyPDF2
                pdf_reader = PyPDF2.PdfReader(BytesIO(content))
                page_count = len(pdf_reader.pages)
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
            
            return text_content.strip(), page_count
            
        except Exception as e:
            logger.error(f"Erro na extração de texto do PDF: {str(e)}")
            raise DocumentProcessingError(f"Erro ao processar PDF: {str(e)}")
    
    @staticmethod
    def extract_text_from_txt(content: bytes) -> str:
        """Extrai texto de arquivo TXT."""
        try:
            # Tentar diferentes encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    return content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            
            # Se nenhum encoding funcionar, usar utf-8 com ignore
            return content.decode('utf-8', errors='ignore')
            
        except Exception as e:
            logger.error(f"Erro na extração de texto TXT: {str(e)}")
            raise DocumentProcessingError(f"Erro ao processar arquivo de texto: {str(e)}")
    
    @staticmethod
    def count_words(text: str) -> int:
        """Conta palavras no texto."""
        if not text:
            return 0
        return len(text.split())
    
    @staticmethod
    def generate_summary(text: str, max_length: int = 500) -> str:
        """
        Gera resumo simples do texto.
        
        Args:
            text: Texto para resumir
            max_length: Comprimento máximo do resumo
            
        Returns:
            str: Resumo do texto
        """
        if not text or len(text) <= max_length:
            return text
        
        # Resumo simples: primeiras sentenças até o limite
        sentences = text.split('.')
        summary = ""
        
        for sentence in sentences:
            if len(summary + sentence + ".") <= max_length:
                summary += sentence + "."
            else:
                break
        
        return summary.strip()

# ================================
# SERVIÇO PRINCIPAL DE DOCUMENTOS
# ================================

class DocumentService:
    """
    Serviço principal para gestão de documentos jurídicos.
    
    Fornece todas as operações CRUD e funcionalidades avançadas
    para documentos no sistema JurisOracle.
    """
    
    def __init__(self, db_session: Session, storage_path: str = "storage/documents"):
        """
        Inicializa o serviço de documentos.
        
        Args:
            db_session: Sessão do banco de dados
            storage_path: Caminho para armazenamento de arquivos
        """
        self.db = db_session
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.processor = DocumentProcessor()
    
    async def create_document(
        self,
        file: UploadFile,
        document_data: DocumentCreate,
        user_id: Optional[uuid.UUID] = None
    ) -> DocumentResponse:
        """
        Cria um novo documento no sistema.
        
        Args:
            file: Arquivo enviado
            document_data: Dados do documento
            user_id: ID do usuário criador
            
        Returns:
            DocumentResponse: Documento criado
            
        Raises:
            DocumentValidationError: Se a validação falhar
            DocumentStorageError: Se houver erro no armazenamento
        """
        try:
            # Validar arquivo
            is_valid, error_msg = self.processor.validate_file(file)
            if not is_valid:
                raise DocumentValidationError(error_msg)
            
            # Ler conteúdo do arquivo
            content = await file.read()
            file_hash = self.processor.calculate_file_hash(content)
            
            # Verificar se documento já existe (por hash)
            existing_doc = self.db.query(Document).filter(
                Document.file_hash == file_hash
            ).first()
            
            if existing_doc:
                raise DocumentValidationError("Documento já existe no sistema")
            
            # Processar conteúdo baseado no tipo
            text_content = ""
            page_count = 0
            
            if file.content_type == 'application/pdf':
                text_content, page_count = self.processor.extract_text_from_pdf(content)
            elif file.content_type == 'text/plain':
                text_content = self.processor.extract_text_from_txt(content)
                page_count = 1
            
            word_count = self.processor.count_words(text_content)
            content_summary = self.processor.generate_summary(text_content)
            
            # Gerar caminho único para o arquivo
            file_extension = Path(file.filename).suffix
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = self.storage_path / unique_filename
            
            # Salvar arquivo no disco
            try:
                with open(file_path, 'wb') as f:
                    f.write(content)
            except Exception as e:
                logger.error(f"Erro ao salvar arquivo: {str(e)}")
                raise DocumentStorageError(f"Erro ao salvar arquivo: {str(e)}")
            
            # Criar registro no banco
            document = Document(
                title=document_data.title,
                description=document_data.description,
                file_name=file.filename,
                file_path=str(file_path),
                file_size=len(content),
                file_hash=file_hash,
                mime_type=file.content_type,
                content_text=text_content,
                content_summary=content_summary,
                page_count=page_count,
                word_count=word_count,
                document_type=document_data.document_type,
                legal_area=document_data.legal_area,
                court_level=document_data.court_level,
                case_number=document_data.case_number,
                parties_involved=document_data.parties_involved,
                keywords=document_data.keywords,
                legal_topics=document_data.legal_topics,
                is_public=document_data.is_public,
                processing_status="completed",
                is_processed=True,
                created_by=user_id,
                updated_by=user_id
            )
            
            self.db.add(document)
            self.db.commit()
            self.db.refresh(document)
            
            # Registrar atividade
            await self._log_activity(
                document_id=document.id,
                user_id=user_id,
                activity_type="CREATE",
                description=f"Documento '{document.title}' criado"
            )
            
            logger.info(f"Documento criado com sucesso: {document.id}")
            return DocumentResponse.from_orm(document)
            
        except (DocumentValidationError, DocumentStorageError):
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Erro inesperado ao criar documento: {str(e)}")
            raise DocumentError(f"Erro ao criar documento: {str(e)}")
    
    def get_document(self, document_id: uuid.UUID, user_id: Optional[uuid.UUID] = None) -> DocumentResponse:
        """
        Obtém um documento por ID.
        
        Args:
            document_id: ID do documento
            user_id: ID do usuário (para auditoria)
            
        Returns:
            DocumentResponse: Documento encontrado
            
        Raises:
            DocumentNotFoundError: Se o documento não for encontrado
        """
        try:
            document = self.db.query(Document).filter(
                Document.id == document_id,
                Document.is_active == True
            ).first()
            
            if not document:
                raise DocumentNotFoundError(f"Documento {document_id} não encontrado")
            
            # Registrar visualização
            asyncio.create_task(self._log_activity(
                document_id=document.id,
                user_id=user_id,
                activity_type="VIEW",
                description=f"Documento '{document.title}' visualizado"
            ))
            
            return DocumentResponse.from_orm(document)
            
        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Erro ao obter documento {document_id}: {str(e)}")
            raise DocumentError(f"Erro ao obter documento: {str(e)}")
    
    def update_document(
        self,
        document_id: uuid.UUID,
        document_data: DocumentUpdate,
        user_id: Optional[uuid.UUID] = None
    ) -> DocumentResponse:
        """
        Atualiza um documento existente.
        
        Args:
            document_id: ID do documento
            document_data: Dados para atualização
            user_id: ID do usuário
            
        Returns:
            DocumentResponse: Documento atualizado
            
        Raises:
            DocumentNotFoundError: Se o documento não for encontrado
        """
        try:
            document = self.db.query(Document).filter(
                Document.id == document_id,
                Document.is_active == True
            ).first()
            
            if not document:
                raise DocumentNotFoundError(f"Documento {document_id} não encontrado")
            
            # Salvar estado anterior para versionamento
            previous_data = {
                'title': document.title,
                'description': document.description,
                'document_type': document.document_type,
                'legal_area': document.legal_area,
                'keywords': document.keywords,
                'legal_topics': document.legal_topics
            }
            
            # Atualizar campos fornecidos
            update_data = document_data.dict(exclude_unset=True)
            changes = []
            
            for field, value in update_data.items():
                if hasattr(document, field) and getattr(document, field) != value:
                    changes.append(f"{field}: {getattr(document, field)} → {value}")
                    setattr(document, field, value)
            
            if changes:
                document.updated_by = user_id
                document.updated_at = datetime.utcnow()
                document.version += 1
                
                # Criar versão
                version = DocumentVersion(
                    document_id=document.id,
                    version_number=document.version - 1,
                    changes_description="; ".join(changes),
                    previous_content=previous_data,
                    created_by=user_id
                )
                self.db.add(version)
                
                self.db.commit()
                self.db.refresh(document)
                
                # Registrar atividade
                asyncio.create_task(self._log_activity(
                    document_id=document.id,
                    user_id=user_id,
                    activity_type="UPDATE",
                    description=f"Documento atualizado: {'; '.join(changes)}"
                ))
                
                logger.info(f"Documento {document_id} atualizado com sucesso")
            
            return DocumentResponse.from_orm(document)
            
        except DocumentNotFoundError:
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Erro ao atualizar documento {document_id}: {str(e)}")
            raise DocumentError(f"Erro ao atualizar documento: {str(e)}")
    
    def delete_document(
        self,
        document_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
        hard_delete: bool = False
    ) -> bool:
        """
        Remove um documento do sistema.
        
        Args:
            document_id: ID do documento
            user_id: ID do usuário
            hard_delete: Se True, remove permanentemente
            
        Returns:
            bool: True se removido com sucesso
            
        Raises:
            DocumentNotFoundError: Se o documento não for encontrado
        """
        try:
            document = self.db.query(Document).filter(
                Document.id == document_id
            ).first()
            
            if not document:
                raise DocumentNotFoundError(f"Documento {document_id} não encontrado")
            
            if hard_delete:
                # Remover arquivo físico
                try:
                    if os.path.exists(document.file_path):
                        os.remove(document.file_path)
                except Exception as e:
                    logger.warning(f"Erro ao remover arquivo físico: {str(e)}")
                
                # Remover do banco
                self.db.delete(document)
                activity_description = f"Documento '{document.title}' removido permanentemente"
            else:
                # Soft delete
                document.is_active = False
                document.updated_by = user_id
                document.updated_at = datetime.utcnow()
                activity_description = f"Documento '{document.title}' desativado"
            
            self.db.commit()
            
            # Registrar atividade
            asyncio.create_task(self._log_activity(
                document_id=document.id,
                user_id=user_id,
                activity_type="DELETE",
                description=activity_description
            ))
            
            logger.info(f"Documento {document_id} removido com sucesso")
            return True
            
        except DocumentNotFoundError:
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Erro ao remover documento {document_id}: {str(e)}")
            raise DocumentError(f"Erro ao remover documento: {str(e)}")
    
    def search_documents(
        self,
        filters: DocumentSearchFilters,
        page: int = 1,
        per_page: int = 20,
        user_id: Optional[uuid.UUID] = None
    ) -> DocumentSearchResponse:
        """
        Busca documentos com filtros avançados.
        
        Args:
            filters: Filtros de busca
            page: Página atual
            per_page: Itens por página
            user_id: ID do usuário
            
        Returns:
            DocumentSearchResponse: Resultados da busca
        """
        try:
            query = self.db.query(Document).filter(Document.is_active == True)
            
            # Aplicar filtros
            if filters.query:
                search_term = f"%{filters.query}%"
                query = query.filter(
                    or_(
                        Document.title.ilike(search_term),
                        Document.description.ilike(search_term),
                        Document.content_text.ilike(search_term),
                        Document.case_number.ilike(search_term)
                    )
                )
            
            if filters.document_type:
                query = query.filter(Document.document_type == filters.document_type)
            
            if filters.legal_area:
                query = query.filter(Document.legal_area == filters.legal_area)
            
            if filters.court_level:
                query = query.filter(Document.court_level == filters.court_level)
            
            if filters.case_number:
                query = query.filter(Document.case_number.ilike(f"%{filters.case_number}%"))
            
            if filters.keywords:
                for keyword in filters.keywords:
                    query = query.filter(Document.keywords.any(keyword))
            
            if filters.date_from:
                query = query.filter(Document.created_at >= filters.date_from)
            
            if filters.date_to:
                query = query.filter(Document.created_at <= filters.date_to)
            
            if filters.is_public is not None:
                query = query.filter(Document.is_public == filters.is_public)
            
            if filters.is_processed is not None:
                query = query.filter(Document.is_processed == filters.is_processed)
            
            if filters.min_confidence is not None:
                query = query.filter(Document.confidence_score >= filters.min_confidence)
            
            # Contar total
            total = query.count()
            
            # Aplicar paginação e ordenação
            documents = query.order_by(desc(Document.created_at)).offset(
                (page - 1) * per_page
            ).limit(per_page).all()
            
            # Calcular páginas
            total_pages = (total + per_page - 1) // per_page
            
            return DocumentSearchResponse(
                documents=[DocumentResponse.from_orm(doc) for doc in documents],
                total=total,
                page=page,
                per_page=per_page,
                total_pages=total_pages
            )
            
        except Exception as e:
            logger.error(f"Erro na busca de documentos: {str(e)}")
            raise DocumentError(f"Erro na busca: {str(e)}")
    
    def get_document_content(self, document_id: uuid.UUID) -> str:
        """
        Obtém o conteúdo textual de um documento.
        
        Args:
            document_id: ID do documento
            
        Returns:
            str: Conteúdo do documento
            
        Raises:
            DocumentNotFoundError: Se o documento não for encontrado
        """
        try:
            document = self.db.query(Document).filter(
                Document.id == document_id,
                Document.is_active == True
            ).first()
            
            if not document:
                raise DocumentNotFoundError(f"Documento {document_id} não encontrado")
            
            return document.content_text or ""
            
        except DocumentNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Erro ao obter conteúdo do documento {document_id}: {str(e)}")
            raise DocumentError(f"Erro ao obter conteúdo: {str(e)}")
    
    def get_document_versions(self, document_id: uuid.UUID) -> List[Dict[str, Any]]:
        """
        Obtém histórico de versões de um documento.
        
        Args:
            document_id: ID do documento
            
        Returns:
            List[Dict]: Lista de versões
        """
        try:
            versions = self.db.query(DocumentVersion).filter(
                DocumentVersion.document_id == document_id
            ).order_by(desc(DocumentVersion.version_number)).all()
            
            return [
                {
                    'version_number': v.version_number,
                    'changes_description': v.changes_description,
                    'created_at': v.created_at,
                    'created_by': v.created_by
                }
                for v in versions
            ]
            
        except Exception as e:
            logger.error(f"Erro ao obter versões do documento {document_id}: {str(e)}")
            raise DocumentError(f"Erro ao obter versões: {str(e)}")
    
    def get_document_activities(
        self,
        document_id: uuid.UUID,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Obtém atividades de um documento.
        
        Args:
            document_id: ID do documento
            limit: Limite de atividades
            
        Returns:
            List[Dict]: Lista de atividades
        """
        try:
            activities = self.db.query(DocumentActivity).filter(
                DocumentActivity.document_id == document_id
            ).order_by(desc(DocumentActivity.created_at)).limit(limit).all()
            
            return [
                {
                    'activity_type': a.activity_type,
                    'description': a.activity_description,
                    'created_at': a.created_at,
                    'user_id': a.user_id,
                    'ip_address': a.ip_address
                }
                for a in activities
            ]
            
        except Exception as e:
            logger.error(f"Erro ao obter atividades do documento {document_id}: {str(e)}")
            raise DocumentError(f"Erro ao obter atividades: {str(e)}")
    
    async def _log_activity(
        self,
        document_id: uuid.UUID,
        user_id: Optional[uuid.UUID],
        activity_type: str,
        description: str,
        metadata: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Registra atividade de documento.
        
        Args:
            document_id: ID do documento
            user_id: ID do usuário
            activity_type: Tipo de atividade
            description: Descrição da atividade
            metadata: Metadados adicionais
            ip_address: Endereço IP
            user_agent: User agent
        """
        try:
            activity = DocumentActivity(
                document_id=document_id,
                user_id=user_id,
                activity_type=activity_type,
                activity_description=description,
                metadata=metadata,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self.db.add(activity)
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Erro ao registrar atividade: {str(e)}")

# ================================
# FUNÇÕES UTILITÁRIAS
# ================================

def get_document_service(db: Session) -> DocumentService:
    """
    Factory function para criar instância do DocumentService.
    
    Args:
        db: Sessão do banco de dados
        
    Returns:
        DocumentService: Instância do serviço
    """
    return DocumentService(db)

def validate_document_access(
    document: Document,
    user_id: Optional[uuid.UUID],
    required_permission: str = "read"
) -> bool:
    """
    Valida se usuário tem acesso ao documento.
    
    Args:
        document: Documento
        user_id: ID do usuário
        required_permission: Permissão necessária
        
    Returns:
        bool: True se tem acesso
    """
    # Documentos públicos são acessíveis a todos
    if document.is_public:
        return True
    
    # Criador sempre tem acesso
    if user_id and document.created_by == user_id:
        return True
    
    # Aqui você pode implementar lógica adicional de permissões
    # baseada em roles, grupos, etc.
    
    return False

# ================================
# EVENTOS SQLAlchemy
# ================================

@event.listens_for(Document, 'before_update')
def document_before_update(mapper, connection, target):
    """Evento executado antes de atualizar documento."""
    target.updated_at = datetime.utcnow()

@event.listens_for(Document, 'after_insert')
def document_after_insert(mapper, connection, target):
    """Evento executado após inserir documento."""
    logger.info(f"Novo documento criado: {target.id} - {target.title}")

@event.listens_for(Document, 'after_update')
def document_after_update(mapper, connection, target):
    """Evento executado após atualizar documento."""
    logger.info(f"Documento atualizado: {target.id} - {target.title}")

# ================================
# CONFIGURAÇÃO DE LOGGING
# ================================

# Configurar logging específico para documentos
document_logger = logging.getLogger('juris_oracle.documents')
document_logger.setLevel(logging.INFO)

# Handler para arquivo de log
file_handler = logging.FileHandler('logs/documents.log')
file_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)

document_logger.addHandler(file_handler)

# ================================
# EXPORTAÇÕES
# ================================

__all__ = [
    # Modelos SQLAlchemy
    'Document',
    'DocumentVersion',
    'DocumentActivity',
    
    # Schemas Pydantic
    'DocumentBase',
    'DocumentCreate',
    'DocumentUpdate',
    'DocumentResponse',
    'DocumentSearchFilters',
    'DocumentSearchResponse',
    
    # Exceções
    'DocumentError',
    'DocumentNotFoundError',
    'DocumentProcessingError',
    'DocumentValidationError',
    'DocumentStorageError',
    
    # Serviços
    'DocumentService',
    'DocumentProcessor',
    
    # Utilitários
    'get_document_service',
    'validate_document_access',
]

if __name__ == "__main__":
    # Código para testes rápidos durante desenvolvimento
    print("Módulo documents.py carregado com sucesso!")
    print(f"Tipos de arquivo suportados: {list(DocumentProcessor.SUPPORTED_MIME_TYPES.keys())}")
    print(f"Tamanho máximo de arquivo: {DocumentProcessor.MAX_FILE_SIZE // (1024*1024)}MB")