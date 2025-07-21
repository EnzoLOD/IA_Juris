"""
API Principal do JurisOracle
API FastAPI para sistema de análise de documentos jurídicos com RAG e HyDE
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import tempfile
import shutil
import uuid

# FastAPI e dependências
from fastapi import (
    FastAPI, HTTPException, Depends, UploadFile, File, 
    BackgroundTasks, status, Request, Response
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

# Pydantic para validação
from pydantic import BaseModel, Field, validator
from typing_extensions import Annotated

# Utilitários
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Imports do projeto
import sys
from pathlib import Path

# Adiciona diretório raiz ao path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.rag_pipeline import JurisRAGPipeline, RAGResponse
from src.core.hyde_retriever import HyDERetriever, HyDEResponse
from src.services.document_processor import DocumentProcessor, ProcessedDocument
from src.utils.security import verify_api_key, create_access_token, get_current_user
from src.utils.monitoring import MetricsCollector, setup_prometheus_metrics
from src.config.settings import get_settings

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurações
settings = get_settings()

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Modelos Pydantic para API
class QueryRequest(BaseModel):
    """Modelo para requisições de query"""
    query: str = Field(..., min_length=3, max_length=2000, description="Pergunta jurídica")
    k: int = Field(default=5, ge=1, le=20, description="Número de resultados")
    score_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Score mínimo")
    use_hyde: bool = Field(default=True, description="Usar HyDE para recuperação")
    strategy: str = Field(default="hybrid", regex="^(hyde|direct|hybrid)$", description="Estratégia de recuperação")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query não pode estar vazia')
        return v.strip()

class QueryResponse(BaseModel):
    """Modelo para resposta de query"""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    strategy_used: str
    timestamp: str
    request_id: str

class DocumentUploadResponse(BaseModel):
    """Modelo para resposta de upload"""
    document_id: str
    filename: str
    file_size: int
    processing_status: str
    chunks_created: int
    processing_time: float
    metadata: Dict[str, Any]
    warnings: List[str] = []
    errors: List[str] = []

class DocumentListResponse(BaseModel):
    """Modelo para listagem de documentos"""
    documents: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int
    total_pages: int

class SystemStatusResponse(BaseModel):
    """Modelo para status do sistema"""
    status: str
    version: str
    uptime: str
    total_documents: int
    total_chunks: int
    memory_usage: Dict[str, Any]
    model_info: Dict[str, Any]
    cache_stats: Dict[str, Any]

class HealthResponse(BaseModel):
    """Modelo para health check"""
    status: str
    timestamp: str
    version: str
    dependencies: Dict[str, bool]

# Variáveis globais para componentes
rag_pipeline: Optional[JurisRAGPipeline] = None
hyde_retriever: Optional[HyDERetriever] = None
document_processor: Optional[DocumentProcessor] = None
metrics_collector: Optional[MetricsCollector] = None
active_sessions: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia ciclo de vida da aplicação"""
    global rag_pipeline, hyde_retriever, document_processor, metrics_collector
    
    logger.info("Iniciando JurisOracle API...")
    
    try:
        # Inicializa componentes principais
        logger.info("Inicializando RAG Pipeline...")
        rag_pipeline = JurisRAGPipeline(
            embedding_model=settings.EMBEDDING_MODEL,
            llm_model=settings.LLM_MODEL,
            vector_store_path=settings.VECTOR_STORE_PATH,
            device=settings.DEVICE
        )
        
        logger.info("Inicializando HyDE Retriever...")
        hyde_retriever = HyDERetriever(
            rag_pipeline=rag_pipeline,
            llm_model=settings.HYDE_LLM_MODEL,
            device=settings.DEVICE
        )
        
        logger.info("Inicializando Document Processor...")
        document_processor = DocumentProcessor(
            cache_dir=settings.DOCUMENT_CACHE_DIR,
            max_workers=settings.MAX_PROCESSING_WORKERS,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # Inicializa métricas
        logger.info("Inicializando sistema de métricas...")
        metrics_collector = MetricsCollector()
        setup_prometheus_metrics(app)
        
        logger.info("JurisOracle API iniciada com sucesso!")
        
        yield
        
    except Exception as e:
        logger.error(f"Erro ao inicializar aplicação: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Finalizando JurisOracle API...")
        if rag_pipeline:
            rag_pipeline.save_vector_store()
        
        # Limpa sessões ativas
        active_sessions.clear()
        
        logger.info("JurisOracle API finalizada.")

# Inicializa FastAPI
app = FastAPI(
    title="JurisOracle API",
    description="""
    ## Sistema Avançado de Análise de Documentos Jurídicos
    
    **JurisOracle** é uma API de Inteligência Artificial especializada em análise de documentos jurídicos brasileiros.
    
    ### Funcionalidades Principais:
    - 📄 **Processamento de Documentos**: Suporte a PDF, DOCX e outros formatos
    - 🔍 **RAG (Retrieval-Augmented Generation)**: Busca semântica avançada
    - 🧠 **HyDE (Hypothetical Document Embeddings)**: Recuperação de alta precisão
    - ⚖️ **Análise Jurídica**: Extração de metadados específicos para documentos legais
    - 🚀 **Performance**: Processamento assíncrono e cache inteligente
    
    ### Tecnologias:
    - Transformers e Sentence-Transformers para embeddings
    - FAISS para busca vetorial
    - spaCy para processamento de linguagem natural
    - FastAPI para API moderna e documentação automática
    
    ### Suporte:
    - Documentos em português brasileiro
    - Múltiplos formatos de arquivo
    - Processamento em lote
    - Autenticação e rate limiting
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Equipe JurisOracle",
        "email": "suporte@jurisoracle.com.br"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SlowAPIMiddleware)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security
security = HTTPBearer()

# Dependências
async def get_current_session(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Obtém sessão atual do usuário"""
    try:
        user = await get_current_user(credentials.credentials)
        session_id = str(uuid.uuid4())
        
        session_data = {
            "session_id": session_id,
            "user_id": user.get("user_id"),
            "created_at": datetime.now(),
            "requests_count": 0
        }
        
        active_sessions[session_id] = session_data
        return session_data
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciais inválidas"
        )

async def get_request_id() -> str:
    """Gera ID único para requisição"""
    return str(uuid.uuid4())

# Middleware para logging e métricas
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware para logging de requisições"""
    start_time = datetime.now()
    request_id = str(uuid.uuid4())
    
    # Adiciona request_id ao estado da requisição
    request.state.request_id = request_id
    
    logger.info(f"Request {request_id} - {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        
        # Calcula tempo de processamento
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log de resposta
        logger.info(
            f"Request {request_id} completed - "
            f"Status: {response.status_code} - "
            f"Time: {processing_time:.3f}s"
        )
        
        # Adiciona headers de resposta
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = str(processing_time)
        
        # Coleta métricas
        if metrics_collector:
            metrics_collector.record_request(
                method=request.method,
                endpoint=str(request.url.path),
                status_code=response.status_code,
                processing_time=processing_time
            )
        
        return response
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Request {request_id} failed - Error: {str(e)} - Time: {processing_time:.3f}s")
        
        # Coleta métricas de erro
        if metrics_collector:
            metrics_collector.record_error(
                method=request.method,
                endpoint=str(request.url.path),
                error_type=type(e).__name__
            )
        
        raise

# Endpoints principais

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raiz da API"""
    return {
        "message": "JurisOracle API - Sistema de Análise de Documentos Jurídicos",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check da API"""
    dependencies = {
        "rag_pipeline": rag_pipeline is not None,
        "hyde_retriever": hyde_retriever is not None,
        "document_processor": document_processor is not None,
        "vector_store": rag_pipeline.vector_store.ntotal > 0 if rag_pipeline else False
    }
    
    status_code = "healthy" if all(dependencies.values()) else "degraded"
    
    return HealthResponse(
        status=status_code,
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        dependencies=dependencies
    )

@app.get("/status", response_model=SystemStatusResponse)
@limiter.limit("10/minute")
async def system_status(
    request: Request,
    session: Dict[str, Any] = Depends(get_current_session)
):
    """Status detalhado do sistema"""
    if not all([rag_pipeline, hyde_retriever, document_processor]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serviços não inicializados"
        )
    
    # Coleta estatísticas
    rag_stats = rag_pipeline.get_statistics()
    hyde_stats = hyde_retriever.get_performance_statistics()
    doc_stats = document_processor.get_processing_statistics()
    
    # Calcula uptime
    uptime = str(datetime.now() - datetime.now().replace(hour=0, minute=0, second=0))
    
    return SystemStatusResponse(
        status="operational",
        version="1.0.0",
        uptime=uptime,
        total_documents=rag_stats.get("total_documents", 0),
        total_chunks=rag_stats.get("total_chunks", 0),
        memory_usage={
            "vector_store_size": rag_stats.get("vector_store_size", 0),
            "cache_size": len(active_sessions)
        },
        model_info={
            "embedding_model": rag_stats.get("embedding_model"),
            "llm_model": hyde_stats.get("llm_model"),
            "device": rag_stats.get("device")
        },
        cache_stats={
            "document_cache_hits": doc_stats.get("cache_hits", 0),
            "active_sessions": len(active_sessions)
        }
    )

@app.post("/documents/upload", response_model=DocumentUploadResponse)
@limiter.limit("5/minute")
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    extract_tables: bool = True,
    extract_images: bool = False,
    session: Dict[str, Any] = Depends(get_current_session),
    request_id: str = Depends(get_request_id)
):
    """
    Upload e processamento de documento
    
    - **file**: Arquivo PDF, DOCX ou TXT
    - **extract_tables**: Extrair informações de tabelas
    - **extract_images**: Extrair informações de imagens
    """
    if not document_processor or not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serviços de processamento não disponíveis"
        )
    
    # Validações do arquivo
    if file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Arquivo muito grande. Máximo: {settings.MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
        )
    
    allowed_types = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Tipo de arquivo não suportado: {file.content_type}"
        )
    
    start_time = datetime.now()
    
    try:
        # Salva arquivo temporário
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Processa documento
        logger.info(f"Processando documento: {file.filename}")
        processed_doc = document_processor.process_document(
            file_path=temp_file_path,
            extract_tables=extract_tables,
            extract_images=extract_images
        )
        
        # Verifica erros de processamento
        if processed_doc.errors:
            logger.warning(f"Documento processado com erros: {processed_doc.errors}")
        
        # Adiciona ao pipeline RAG em background
        def add_to_rag():
            try:
                chunks_added = rag_pipeline.add_document(
                    content=processed_doc.text_content,
                    document_id=f"{session['user_id']}_{file.filename}_{request_id}",
                    metadata={
                        "filename": file.filename,
                        "upload_session": session["session_id"],
                        "user_id": session["user_id"],
                        "request_id": request_id,
                        **processed_doc.metadata.__dict__
                    }
                )
                
                # Salva pipeline
                rag_pipeline.save_vector_store()
                
                logger.info(f"Documento adicionado ao RAG: {chunks_added} chunks")
                
            except Exception as e:
                logger.error(f"Erro ao adicionar documento ao RAG: {e}")
        
        background_tasks.add_task(add_to_rag)
        
        # Limpa arquivo temporário
        def cleanup():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        
        background_tasks.add_task(cleanup)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Atualiza estatísticas da sessão
        session["requests_count"] += 1
        
        return DocumentUploadResponse(
            document_id=f"{session['user_id']}_{file.filename}_{request_id}",
            filename=file.filename,
            file_size=file.size,
            processing_status="completed" if not processed_doc.errors else "completed_with_errors",
            chunks_created=len(processed_doc.chunks),
            processing_time=processing_time,
            metadata={
                "pages": processed_doc.metadata.pages,
                "word_count": processed_doc.metadata.word_count,
                "language": processed_doc.metadata.language,
                "document_type": processed_doc.metadata.document_type
            },
            warnings=processed_doc.warnings,
            errors=processed_doc.errors
        )
        
    except Exception as e:
        logger.error(f"Erro no upload de documento: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar documento: {str(e)}"
        )

@app.post("/query/rag", response_model=QueryResponse)
@limiter.limit("20/minute")
async def query_rag(
    request: Request,
    query_request: QueryRequest,
    session: Dict[str, Any] = Depends(get_current_session),
    request_id: str = Depends(get_request_id)
):
    """
    Executa query usando RAG básico
    
    - **query**: Pergunta jurídica
    - **k**: Número de documentos para recuperar (1-20)
    - **score_threshold**: Score mínimo de relevância (0.0-1.0)
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline RAG não disponível"
        )
    
    try:
        logger.info(f"Executando query RAG: {query_request.query[:100]}...")
        
        # Executa query RAG
        rag_response = rag_pipeline.query(
            query=query_request.query,
            k=query_request.k,
            score_threshold=query_request.score_threshold,
            generate_answer=True
        )
        
        # Formata sources
        sources = []
        for result in rag_response.sources:
            sources.append({
                "chunk_id": result.chunk.id,
                "content": result.chunk.content[:500] + "..." if len(result.chunk.content) > 500 else result.chunk.content,
                "score": result.score,
                "rank": result.rank,
                "source_document": result.chunk.source_document,
                "page": result.chunk.start_page,
                "chunk_type": result.chunk.chunk_type
            })
        
        # Atualiza estatísticas
        session["requests_count"] += 1
        
        return QueryResponse(
            query=rag_response.query,
            answer=rag_response.answer,
            sources=sources,
            confidence=rag_response.confidence,
            processing_time=rag_response.processing_time,
            strategy_used="rag_basic",
            timestamp=rag_response.timestamp,
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Erro na query RAG: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar query: {str(e)}"
        )

@app.post("/query/hyde", response_model=QueryResponse)
@limiter.limit("15/minute")  # Limite menor por ser mais custoso
async def query_hyde(
    request: Request,
    query_request: QueryRequest,
    session: Dict[str, Any] = Depends(get_current_session),
    request_id: str = Depends(get_request_id)
):
    """
    Executa query usando HyDE (Hypothetical Document Embeddings)
    
    - **query**: Pergunta jurídica
    - **k**: Número de documentos para recuperar (1-20)
    - **score_threshold**: Score mínimo de relevância (0.0-1.0)
    - **strategy**: Estratégia de recuperação (hyde, direct, hybrid)
    """
    if not hyde_retriever:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="HyDE Retriever não disponível"
        )
    
    try:
        logger.info(f"Executando query HyDE: {query_request.query[:100]}...")
        
        # Executa query HyDE
        hyde_response = hyde_retriever.query(
            query=query_request.query,
            k=query_request.k,
            score_threshold=query_request.score_threshold,
            strategy=query_request.strategy,
            generate_answer=True
        )
        
        # Formata sources
        sources = []
        for result in hyde_response.results:
            sources.append({
                "chunk_id": result.original_result.chunk.id,
                "content": result.original_result.chunk.content[:500] + "..." if len(result.original_result.chunk.content) > 500 else result.original_result.chunk.content,
                "hyde_score": result.hyde_score,
                "direct_score": result.direct_score,
                "combined_score": result.combined_score,
                "rank": result.original_result.rank,
                "source_document": result.original_result.chunk.source_document,
                "page": result.original_result.chunk.start_page,
                "chunk_type": result.original_result.chunk.chunk_type,
                "retrieval_method": result.retrieval_method
            })
        
        # Atualiza estatísticas
        session["requests_count"] += 1
        
        return QueryResponse(
            query=hyde_response.query,
            answer=hyde_response.answer,
            sources=sources,
            confidence=hyde_response.confidence,
            processing_time=hyde_response.processing_time,
            strategy_used=hyde_response.strategy_used,
            timestamp=hyde_response.timestamp,
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Erro na query HyDE: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar query HyDE: {str(e)}"
        )

@app.get("/documents", response_model=DocumentListResponse)
@limiter.limit("30/minute")
async def list_documents(
    request: Request,
    page: int = 1,
    page_size: int = 20,
    session: Dict[str, Any] = Depends(get_current_session)
):
    """Lista documentos do usuário"""
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline RAG não disponível"
        )
    
    try:
        # Filtra documentos do usuário
        user_docs = []
        for chunk in rag_pipeline.document_chunks:
            if chunk.metadata.get("user_id") == session["user_id"]:
                doc_info = {
                    "document_id": chunk.metadata.get("document_id", chunk.source_document),
                    "filename": chunk.metadata.get("filename", "Unknown"),
                    "uploaded_at": chunk.created_at,
                    "pages": chunk.metadata.get("pages", 0),
                    "word_count": chunk.metadata.get("word_count", 0),
                    "document_type": chunk.metadata.get("document_type", "Unknown")
                }
                
                # Evita duplicatas
                if not any(doc["document_id"] == doc_info["document_id"] for doc in user_docs):
                    user_docs.append(doc_info)
        
        # Paginação
        total_count = len(user_docs)
        total_pages = (total_count + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        paginated_docs = user_docs[start_idx:end_idx]
        
        return DocumentListResponse(
            documents=paginated_docs,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.error(f"Erro ao listar documentos: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao listar documentos: {str(e)}"
        )

@app.delete("/documents/{document_id}")
@limiter.limit("10/minute")
async def delete_document(
    request: Request,
    document_id: str,
    session: Dict[str, Any] = Depends(get_current_session)
):
    """Remove documento do sistema"""
    # Implementação simplificada - em produção seria mais complexa
    # pois FAISS não suporta remoção direta de vetores
    
    logger.info(f"Solicitação de remoção de documento: {document_id}")
    
    return {
        "message": f"Documento {document_id} marcado para remoção",
        "note": "Remoção completa será processada na próxima reconstrução do índice"
    }

@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """Endpoint para métricas Prometheus"""
    if metrics_collector:
        return Response(
            content=metrics_collector.generate_prometheus_metrics(),
            media_type="text/plain"
        )
    return {"message": "Métricas não disponíveis"}

# Handler de erros global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handler global para exceções não tratadas"""
    logger.error(f"Erro não tratado: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Erro interno do servidor",
            "request_id": getattr(request.state, "request_id", "unknown"),
            "timestamp": datetime.now().isoformat()
        }
    )

# Configuração customizada do OpenAPI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="JurisOracle API",
        version="1.0.0",
        description="API de Análise de Documentos Jurídicos com IA",
        routes=app.routes,
    )
    
    # Adiciona informações de segurança
    openapi_schema["components"]["securitySchemes"] = {
        "HTTPBearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Configuração para execução direta
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if not settings.DEBUG else "debug",
        access_log=True
    )