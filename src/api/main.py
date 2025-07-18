"""
main.py - Juris Oracle API Routes
Sistema RAG (Retrieval Augmented Generation) com HyDE (Hypothetical Document Embeddings)
para análise e processamento de documentos jurídicos.

Este módulo implementa a interface principal da API, orquestrando funcionalidades
avançadas de IA para busca semântica e geração de respostas aumentadas.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from pydantic.types import UUID4

# Mock imports - em produção, estes seriam importados dos módulos reais
from typing import Protocol

logger = logging.getLogger(__name__)

# ==================== PROTOCOLS & INTERFACES ====================

class EmbeddingService(Protocol):
    """Interface para serviços de embedding."""
    async def generate_embedding(self, text: str) -> List[float]:
        """Gera embedding para o texto fornecido."""
        ...

class VectorStoreService(Protocol):
    """Interface para serviços de vector store."""
    async def store_embeddings(self, embeddings: List[float], metadata: Dict[str, Any]) -> str:
        """Armazena embeddings com metadados."""
        ...
    
    async def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Busca por similaridade no vector store."""
        ...

class LLMService(Protocol):
    """Interface para serviços de LLM."""
    async def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Gera resposta usando LLM."""
        ...

# ==================== PYDANTIC MODELS ====================

class BaseResponse(BaseModel):
    """Modelo base para respostas da API."""
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class HealthResponse(BaseResponse):
    """Modelo de resposta para health check."""
    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    uptime_seconds: float

class DocumentIngestRequest(BaseModel):
    """Modelo de requisição para ingestão de documentos."""
    content: str = Field(..., min_length=10, max_length=100000, description="Conteúdo do documento")
    document_type: str = Field(..., description="Tipo do documento (e.g., 'jurisprudencia', 'lei', 'doutrina')")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadados adicionais")
    chunk_size: Optional[int] = Field(default=1000, ge=100, le=5000, description="Tamanho dos chunks")
    overlap: Optional[int] = Field(default=200, ge=0, le=1000, description="Sobreposição entre chunks")

    @validator('document_type')
    def validate_document_type(cls, v):
        allowed_types = ['jurisprudencia', 'lei', 'doutrina', 'parecer', 'contrato', 'peticao']
        if v.lower() not in allowed_types:
            raise ValueError(f'Tipo de documento deve ser um de: {allowed_types}')
        return v.lower()

class DocumentIngestResponse(BaseResponse):
    """Modelo de resposta para ingestão de documentos."""
    document_id: UUID4
    chunks_created: int
    processing_time_seconds: float
    vector_store_ids: List[str]

class QueryRequest(BaseModel):
    """Modelo de requisição para consultas RAG."""
    query: str = Field(..., min_length=5, max_length=2000, description="Consulta do usuário")
    max_results: Optional[int] = Field(default=5, ge=1, le=20, description="Número máximo de resultados")
    similarity_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0, description="Threshold de similaridade")
    document_types: Optional[List[str]] = Field(default=None, description="Filtrar por tipos de documento")
    include_metadata: Optional[bool] = Field(default=True, description="Incluir metadados nos resultados")

class RetrievedDocument(BaseModel):
    """Modelo para documentos recuperados."""
    document_id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    chunk_index: int

class QueryResponse(BaseResponse):
    """Modelo de resposta para consultas RAG."""
    query: str
    generated_response: str
    retrieved_documents: List[RetrievedDocument]
    processing_time_seconds: float
    total_tokens_used: int

class HydeGenerateRequest(BaseModel):
    """Modelo de requisição para geração HyDE."""
    query: str = Field(..., min_length=5, max_length=2000, description="Consulta para geração hipotética")
    num_hypothetical_docs: Optional[int] = Field(default=3, ge=1, le=10, description="Número de documentos hipotéticos")
    max_tokens_per_doc: Optional[int] = Field(default=500, ge=100, le=2000, description="Tokens máximos por documento")
    document_type_context: Optional[str] = Field(default="jurisprudencia", description="Contexto do tipo de documento")

class HypotheticalDocument(BaseModel):
    """Modelo para documentos hipotéticos gerados."""
    content: str
    embedding: List[float]
    generation_confidence: float

class HydeGenerateResponse(BaseResponse):
    """Modelo de resposta para geração HyDE."""
    original_query: str
    hypothetical_documents: List[HypotheticalDocument]
    enhanced_search_results: Optional[List[RetrievedDocument]] = None
    processing_time_seconds: float

class ErrorResponse(BaseModel):
    """Modelo padronizado para respostas de erro."""
    success: bool = False
    error_code: str
    error_message: str
    error_details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# ==================== MOCK SERVICES ====================

class MockEmbeddingService:
    """Serviço mock para geração de embeddings."""
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Simula geração de embedding."""
        logger.info(f"Gerando embedding para texto de {len(text)} caracteres")
        # Simula tempo de processamento
        await asyncio.sleep(0.1)
        # Retorna embedding mock de 384 dimensões
        import random
        return [random.uniform(-1, 1) for _ in range(384)]

class MockVectorStoreService:
    """Serviço mock para vector store."""
    
    def __init__(self):
        self.stored_vectors = {}
    
    async def store_embeddings(self, embeddings: List[float], metadata: Dict[str, Any]) -> str:
        """Simula armazenamento de embeddings."""
        vector_id = str(uuid.uuid4())
        self.stored_vectors[vector_id] = {
            "embedding": embeddings,
            "metadata": metadata,
            "timestamp": datetime.utcnow()
        }
        logger.info(f"Embedding armazenado com ID: {vector_id}")
        return vector_id
    
    async def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Simula busca por similaridade."""
        import random
        results = []
        
        # Simula resultados baseados nos vetores armazenados
        available_vectors = list(self.stored_vectors.values())[:top_k]
        
        for i, vector_data in enumerate(available_vectors):
            similarity_score = random.uniform(0.6, 0.95)  # Score mock
            results.append({
                "document_id": f"doc_{i+1}",
                "content": f"Conteúdo jurídico relevante {i+1}...",
                "similarity_score": similarity_score,
                "metadata": vector_data["metadata"],
                "chunk_index": i
            })
        
        # Ordena por score de similaridade
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results

class MockLLMService:
    """Serviço mock para LLM."""
    
    async def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Simula geração de resposta."""
        logger.info(f"Gerando resposta LLM para prompt de {len(prompt)} caracteres")
        await asyncio.sleep(0.5)  # Simula tempo de processamento
        
        if "hipotético" in prompt.lower() or "hypothetical" in prompt.lower():
            return """Com base na consulta jurídica apresentada, é relevante analisar 
            a jurisprudência consolidada do Superior Tribunal de Justiça que estabelece 
            precedentes importantes sobre a matéria. A doutrina majoritária converge 
            no sentido de que os princípios constitucionais devem ser interpretados 
            de forma sistemática e harmônica."""
        
        return """Baseado nos documentos recuperados e na análise jurídica realizada, 
        a resposta contempla os aspectos doutrinários e jurisprudenciais relevantes 
        para a questão apresentada. É importante considerar o contexto específico 
        e as particularidades do caso concreto."""

# ==================== DEPENDENCY INJECTION ====================

import asyncio

async def get_embedding_service() -> EmbeddingService:
    """Dependency injection para serviço de embedding."""
    return MockEmbeddingService()

async def get_vector_store_service() -> VectorStoreService:
    """Dependency injection para serviço de vector store."""
    return MockVectorStoreService()

async def get_llm_service() -> LLMService:
    """Dependency injection para serviço de LLM."""
    return MockLLMService()

# ==================== EXCEPTION HANDLERS ====================

class JurisOracleException(Exception):
    """Exceção base personalizada para a aplicação."""
    def __init__(self, message: str, error_code: str = "GENERIC_ERROR", details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class DocumentProcessingException(JurisOracleException):
    """Exceção para erros de processamento de documentos."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "DOCUMENT_PROCESSING_ERROR", details)

class EmbeddingGenerationException(JurisOracleException):
    """Exceção para erros de geração de embeddings."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "EMBEDDING_GENERATION_ERROR", details)

class VectorSearchException(JurisOracleException):
    """Exceção para erros de busca vetorial."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, "VECTOR_SEARCH_ERROR", details)

# ==================== UTILITY FUNCTIONS ====================

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Divide texto em chunks com sobreposição.
    
    Args:
        text: Texto a ser dividido
        chunk_size: Tamanho de cada chunk
        overlap: Sobreposição entre chunks
    
    Returns:
        Lista de chunks de texto
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end == len(text):
            break
            
        start = end - overlap
    
    return chunks

async def create_enhanced_prompt(query: str, retrieved_docs: List[RetrievedDocument], context_type: str = "jurisprudencia") -> str:
    """
    Cria prompt enriquecido com contexto recuperado.
    
    Args:
        query: Consulta original do usuário
        retrieved_docs: Documentos recuperados
        context_type: Tipo de contexto para personalização
    
    Returns:
        Prompt enriquecido para LLM
    """
    context_parts = []
    
    for i, doc in enumerate(retrieved_docs[:3], 1):  # Limita a 3 documentos principais
        context_parts.append(f"[DOCUMENTO {i}]\n{doc.content}\n")
    
    context = "\n".join(context_parts)
    
    prompt = f"""Você é um assistente jurídico especializado. Com base nos documentos fornecidos como contexto, 
responda à consulta do usuário de forma precisa e fundamentada.

CONTEXTO:
{context}

CONSULTA DO USUÁRIO:
{query}

INSTRUÇÕES:
- Baseie sua resposta exclusivamente nos documentos fornecidos como contexto
- Cite as fontes quando relevante
- Se a informação não estiver disponível no contexto, indique claramente
- Mantenha um tom profissional e técnico apropriado para o âmbito jurídico
- Estruture a resposta de forma clara e objetiva

RESPOSTA:"""

    return prompt

# ==================== ROUTER SETUP ====================

router = APIRouter(prefix="/api/v1", tags=["Juris Oracle API"])

# Variável global para tracking de uptime
app_start_time = time.time()

# ==================== ROUTES ====================

@router.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check() -> HealthResponse:
    """
    Endpoint de health check para monitoramento da API.
    
    Returns:
        Status de saúde da aplicação com informações de uptime
    """
    uptime = time.time() - app_start_time
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime_seconds=uptime
    )

@router.post("/documents/ingest", response_model=DocumentIngestResponse, summary="Ingestão de Documentos")
async def ingest_document(
    request: DocumentIngestRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store_service)
) -> DocumentIngestResponse:
    """
    Processa e indexa um novo documento para o sistema RAG.
    
    Este endpoint realiza:
    1. Chunking do documento em segmentos processáveis
    2. Geração de embeddings para cada chunk
    3. Armazenamento no vector store com metadados
    
    Args:
        request: Dados do documento a ser processado
        embedding_service: Serviço de geração de embeddings
        vector_store: Serviço de armazenamento vetorial
    
    Returns:
        Informações sobre o processamento do documento
    
    Raises:
        HTTPException: Para erros de processamento ou validação
    """
    start_time = time.time()
    
    try:
        # Gerar ID único para o documento
        document_id = uuid.uuid4()
        
        # Dividir documento em chunks
        chunks = chunk_text(
            request.content, 
            chunk_size=request.chunk_size, 
            overlap=request.overlap
        )
        
        logger.info(f"Documento {document_id} dividido em {len(chunks)} chunks")
        
        # Processar cada chunk
        vector_store_ids = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Gerar embedding para o chunk
                embedding = await embedding_service.generate_embedding(chunk)
                
                # Preparar metadados
                chunk_metadata = {
                    "document_id": str(document_id),
                    "chunk_index": i,
                    "chunk_content": chunk,
                    "document_type": request.document_type,
                    "created_at": datetime.utcnow().isoformat(),
                    **request.metadata
                }
                
                # Armazenar no vector store
                vector_id = await vector_store.store_embeddings(embedding, chunk_metadata)
                vector_store_ids.append(vector_id)
                
            except Exception as e:
                logger.error(f"Erro ao processar chunk {i}: {str(e)}")
                raise DocumentProcessingException(
                    f"Falha ao processar chunk {i}",
                    details={"chunk_index": i, "error": str(e)}
                )
        
        processing_time = time.time() - start_time
        
        logger.info(f"Documento {document_id} processado com sucesso em {processing_time:.2f}s")
        
        return DocumentIngestResponse(
            document_id=document_id,
            chunks_created=len(chunks),
            processing_time_seconds=processing_time,
            vector_store_ids=vector_store_ids
        )
        
    except DocumentProcessingException:
        raise
    except Exception as e:
        logger.error(f"Erro inesperado na ingestão: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno durante processamento: {str(e)}"
        )

@router.post("/query", response_model=QueryResponse, summary="Consulta RAG")
async def query_rag(
    request: QueryRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store_service),
    llm_service: LLMService = Depends(get_llm_service)
) -> QueryResponse:
    """
    Realiza consulta utilizando RAG (Retrieval Augmented Generation).
    
    O processo inclui:
    1. Geração de embedding da consulta
    2. Busca por similaridade no vector store
    3. Composição de prompt com contexto recuperado
    4. Geração de resposta pela LLM
    
    Args:
        request: Dados da consulta
        embedding_service: Serviço de embeddings
        vector_store: Serviço de busca vetorial
        llm_service: Serviço de LLM
    
    Returns:
        Resposta gerada com documentos de apoio
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processando consulta RAG: {request.query[:100]}...")
        
        # Gerar embedding da consulta
        query_embedding = await embedding_service.generate_embedding(request.query)
        
        # Buscar documentos similares
        search_results = await vector_store.similarity_search(
            query_embedding, 
            top_k=request.max_results
        )
        
        # Filtrar por threshold de similaridade
        filtered_results = [
            result for result in search_results 
            if result["similarity_score"] >= request.similarity_threshold
        ]
        
        # Filtrar por tipos de documento se especificado
        if request.document_types:
            filtered_results = [
                result for result in filtered_results
                if result["metadata"].get("document_type") in request.document_types
            ]
        
        # Converter para modelo Pydantic
        retrieved_docs = [
            RetrievedDocument(
                document_id=result["document_id"],
                content=result["content"],
                similarity_score=result["similarity_score"],
                metadata=result["metadata"] if request.include_metadata else {},
                chunk_index=result["chunk_index"]
            )
            for result in filtered_results
        ]
        
        # Criar prompt enriquecido
        enhanced_prompt = await create_enhanced_prompt(request.query, retrieved_docs)
        
        # Gerar resposta
        generated_response = await llm_service.generate_response(enhanced_prompt)
        
        processing_time = time.time() - start_time
        
        # Estimar tokens utilizados (mock)
        total_tokens = len(enhanced_prompt.split()) + len(generated_response.split())
        
        logger.info(f"Consulta processada em {processing_time:.2f}s, {len(retrieved_docs)} documentos recuperados")
        
        return QueryResponse(
            query=request.query,
            generated_response=generated_response,
            retrieved_documents=retrieved_docs,
            processing_time_seconds=processing_time,
            total_tokens_used=total_tokens
        )
        
    except Exception as e:
        logger.error(f"Erro na consulta RAG: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro durante processamento da consulta: {str(e)}"
        )

@router.post("/hyde/generate", response_model=HydeGenerateResponse, summary="Geração HyDE")
async def generate_hyde(
    request: HydeGenerateRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store_service),
    llm_service: LLMService = Depends(get_llm_service)
) -> HydeGenerateResponse:
    """
    Implementa HyDE (Hypothetical Document Embeddings) para melhorar recuperação.
    
    O processo HyDE:
    1. Gera documentos hipotéticos baseados na consulta
    2. Cria embeddings dos documentos hipotéticos
    3. Opcionalmente realiza busca aprimorada
    
    Args:
        request: Parâmetros para geração HyDE
        embedding_service: Serviço de embeddings
        vector_store: Serviço de busca vetorial
        llm_service: Serviço de LLM
    
    Returns:
        Documentos hipotéticos e resultados de busca aprimorada
    """
    start_time = time.time()
    
    try:
        logger.info(f"Iniciando geração HyDE para: {request.query[:100]}...")
        
        hypothetical_documents = []
        
        # Gerar documentos hipotéticos
        for i in range(request.num_hypothetical_docs):
            # Criar prompt para geração hipotética
            hyde_prompt = f"""Você é um especialista jurídico. Com base na consulta fornecida, 
gere um documento {request.document_type_context} hipotético que seria uma resposta ideal 
e completa para essa consulta.

CONSULTA: {request.query}

INSTRUÇÕES:
- Crie um documento técnico e preciso
- Use linguagem jurídica apropriada
- Inclua fundamentos legais relevantes
- Mantenha foco na consulta específica
- Limite-se a {request.max_tokens_per_doc} tokens

DOCUMENTO HIPOTÉTICO:"""

            # Gerar documento hipotético
            hypothetical_content = await llm_service.generate_response(
                hyde_prompt, 
                max_tokens=request.max_tokens_per_doc
            )
            
            # Gerar embedding do documento hipotético
            hypothetical_embedding = await embedding_service.generate_embedding(hypothetical_content)
            
            # Calcular confiança mock (em produção seria baseado em métricas reais)
            import random
            confidence = random.uniform(0.75, 0.95)
            
            hypothetical_documents.append(HypotheticalDocument(
                content=hypothetical_content,
                embedding=hypothetical_embedding,
                generation_confidence=confidence
            ))
        
        # Opcionalmente, realizar busca aprimorada usando os documentos hipotéticos
        enhanced_search_results = []
        
        if len(hypothetical_documents) > 0:
            # Usar o melhor documento hipotético para busca
            best_doc = max(hypothetical_documents, key=lambda x: x.generation_confidence)
            
            search_results = await vector_store.similarity_search(
                best_doc.embedding, 
                top_k=5
            )
            
            enhanced_search_results = [
                RetrievedDocument(
                    document_id=result["document_id"],
                    content=result["content"],
                    similarity_score=result["similarity_score"],
                    metadata=result["metadata"],
                    chunk_index=result["chunk_index"]
                )
                for result in search_results
            ]
        
        processing_time = time.time() - start_time
        
        logger.info(f"HyDE gerado em {processing_time:.2f}s, {len(hypothetical_documents)} documentos criados")
        
        return HydeGenerateResponse(
            original_query=request.query,
            hypothetical_documents=hypothetical_documents,
            enhanced_search_results=enhanced_search_results,
            processing_time_seconds=processing_time
        )
        
    except Exception as e:
        logger.error(f"Erro na geração HyDE: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro durante geração HyDE: {str(e)}"
        )

# ==================== EXCEPTION HANDLERS ====================

@router.exception_handler(JurisOracleException)
async def juris_oracle_exception_handler(request: Request, exc: JurisOracleException):
    """Handler para exceções customizadas da aplicação."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error_code=exc.error_code,
            error_message=exc.message,
            error_details=exc.details
        ).dict()
    )

@router.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler para exceções HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=f"HTTP_{exc.status_code}",
            error_message=exc.detail,
            error_details={"status_code": exc.status_code}
        ).dict()
    )

@router.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handler global para exceções não tratadas."""
    logger.error(f"Erro não tratado: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            error_message="Erro interno do servidor",
            error_details={"original_error": str(exc)}
        ).dict()
    )

# ==================== MIDDLEWARE E CONFIGURAÇÕES ====================

def setup_cors(app):
    """Configura CORS para a aplicação."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Em produção, especificar origins permitidas
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

@router.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware para adicionar tempo de processamento nos headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@router.middleware("http")
async def add_request_id_header(request: Request, call_next):
    """Middleware para adicionar ID único da requisição."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# ==================== LOGGING CONFIGURATION ====================

def setup_logging():
    """Configura logging para a aplicação."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('juris_oracle.log')
        ]
    )

# Configurar logging na inicialização
setup_logging()

# ==================== METADATA E DOCUMENTAÇÃO ====================

# Metadados para documentação da API
tags_metadata = [
    {
        "name": "Juris Oracle API",
        "description": "API principal para sistema RAG com HyDE especializado em documentos jurídicos",
    }
]

# Informações adicionais para OpenAPI
openapi_info = {
    "title": "Juris Oracle API",
    "description": """
    Sistema avançado de RAG (Retrieval Augmented Generation) com HyDE 
    (Hypothetical Document Embeddings) para análise jurídica.
    
    ## Funcionalidades Principais
    
    * **Ingestão de Documentos**: Processamento e indexação de documentos jurídicos
    * **Consulta RAG**: Busca semântica com geração de respostas aumentadas
    * **Geração HyDE**: Criação de documentos hipotéticos para melhor recuperação
    * **Health Check**: Monitoramento da saúde da aplicação
    
    ## Arquitetura
    
    O sistema utiliza embeddings vetoriais para busca semântica e LLMs para 
    geração de respostas contextualizadas, proporcionando análises jurídicas 
    precisas e fundamentadas.
    """,
    "version": "1.0.0",
    "contact": {
        "name": "Juris Oracle Team",
        "email": "dev@jurisoracle.com"
    }
}

# Exportar router para uso na aplicação principal
__all__ = ["router", "setup_cors", "tags_metadata", "openapi_info"]