"""
routes/__init__.py - Configuração Central de Rotas

Este módulo centraliza e organiza todas as rotas da aplicação JurisOracle,
fornecendo um ponto único de registro e configuração para todos os endpoints
da API, middlewares de rota e tratamento de erros.

Estrutura:
- Importação automática de todos os módulos de rota
- Registro centralizado de APIRouters
- Tratamento global de erros HTTP
- Configuração de middleware de rota
- Validação e documentação automática

Autor: Sistema JurisOracle
Data: 2025
Versão: 1.0.0
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback
from datetime import datetime

# Importação de módulos de rota individuais
from .auth import router as auth_router
from .users import router as users_router
from .documents import router as documents_router
from .search import router as search_router
from .analysis import router as analysis_router
from .training import router as training_router
from .dashboard import router as dashboard_router

# Configuração de logging
logger = logging.getLogger(__name__)

# Router principal para o pacote routes
main_router = APIRouter(
    prefix="/api/v1",
    tags=["JurisOracle API"],
    responses={
        404: {"description": "Recurso não encontrado"},
        500: {"description": "Erro interno do servidor"},
        422: {"description": "Erro de validação de dados"}
    }
)

# Configuração de rotas por módulo com metadados
ROUTE_MODULES = [
    {
        "router": auth_router,
        "prefix": "/auth",
        "tags": ["Autenticação"],
        "description": "Endpoints de autenticação e autorização"
    },
    {
        "router": users_router,
        "prefix": "/users",
        "tags": ["Usuários"],
        "description": "Gerenciamento de usuários e perfis"
    },
    {
        "router": documents_router,
        "prefix": "/documents",
        "tags": ["Documentos"],
        "description": "Gestão de documentos jurídicos"
    },
    {
        "router": search_router,
        "prefix": "/search",
        "tags": ["Busca"],
        "description": "Busca e recuperação de informações"
    },
    {
        "router": analysis_router,
        "prefix": "/analysis",
        "tags": ["Análise"],
        "description": "Análise jurídica e extração de insights"
    },
    {
        "router": training_router,
        "prefix": "/training",
        "tags": ["Treinamento"],
        "description": "Treinamento e gestão de modelos IA"
    },
    {
        "router": dashboard_router,
        "prefix": "/dashboard",
        "tags": ["Dashboard"],
        "description": "Métricas e visualizações"
    }
]

# Registro automático de todos os routers
for route_config in ROUTE_MODULES:
    try:
        main_router.include_router(
            route_config["router"],
            prefix=route_config["prefix"],
            tags=route_config["tags"]
        )
        logger.info(
            f"Router registrado: {route_config['prefix']} - "
            f"{route_config['description']}"
        )
    except Exception as e:
        logger.error(
            f"Erro ao registrar router {route_config['prefix']}: {str(e)}"
        )
        raise


# Handlers de erro globais para o pacote routes
@main_router.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, 
    exc: StarletteHTTPException
) -> JSONResponse:
    """
    Handler global para exceções HTTP.
    
    Args:
        request: Objeto de requisição FastAPI
        exc: Exceção HTTP capturada
        
    Returns:
        JSONResponse: Resposta JSON estruturada com erro
    """
    logger.warning(
        f"HTTP Exception {exc.status_code}: {exc.detail} - "
        f"Path: {request.url.path} - Method: {request.method}"
    )
    
    error_responses = {
        400: "Requisição inválida",
        401: "Não autorizado - Credenciais inválidas",
        403: "Acesso negado - Permissões insuficientes",
        404: "Recurso não encontrado",
        405: "Método não permitido",
        409: "Conflito de dados",
        422: "Dados de entrada inválidos",
        429: "Muitas requisições - Limite excedido",
        500: "Erro interno do servidor",
        502: "Serviço indisponível",
        503: "Serviço temporariamente indisponível"
    }
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": error_responses.get(exc.status_code, exc.detail),
            "detail": exc.detail if exc.status_code != 500 else None,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path),
            "method": request.method
        }
    )


@main_router.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, 
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handler para erros de validação de dados Pydantic.
    
    Args:
        request: Objeto de requisição FastAPI
        exc: Exceção de validação capturada
        
    Returns:
        JSONResponse: Resposta JSON com detalhes da validação
    """
    logger.warning(
        f"Validation Error: {exc.errors()} - "
        f"Path: {request.url.path} - Method: {request.method}"
    )
    
    validation_errors = []
    for error in exc.errors():
        validation_errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": True,
            "status_code": 422,
            "message": "Erro de validação nos dados fornecidos",
            "validation_errors": validation_errors,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path),
            "method": request.method
        }
    )


@main_router.exception_handler(Exception)
async def general_exception_handler(
    request: Request, 
    exc: Exception
) -> JSONResponse:
    """
    Handler para exceções gerais não tratadas.
    
    Args:
        request: Objeto de requisição FastAPI
        exc: Exceção geral capturada
        
    Returns:
        JSONResponse: Resposta JSON genérica para erro interno
    """
    logger.error(
        f"Unhandled Exception: {str(exc)} - "
        f"Path: {request.url.path} - Method: {request.method}\n"
        f"Traceback: {traceback.format_exc()}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "status_code": 500,
            "message": "Erro interno do servidor",
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path),
            "method": request.method,
            "support_message": (
                "Se o problema persistir, entre em contato com o suporte "
                "técnico fornecendo o timestamp desta resposta."
            )
        }
    )


# Middleware de logging para todas as rotas
@main_router.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    Middleware para logging de requisições e respostas.
    
    Args:
        request: Objeto de requisição FastAPI
        call_next: Próximo middleware/handler na cadeia
        
    Returns:
        Response: Resposta processada
    """
    start_time = datetime.utcnow()
    
    # Log da requisição
    logger.info(
        f"Request: {request.method} {request.url.path} - "
        f"Client: {request.client.host if request.client else 'unknown'}"
    )
    
    # Processa a requisição
    try:
        response = await call_next(request)
        
        # Calcula tempo de processamento
        process_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Log da resposta
        logger.info(
            f"Response: {response.status_code} - "
            f"Time: {process_time:.3f}s - "
            f"Path: {request.url.path}"
        )
        
        # Adiciona header com tempo de processamento
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = (datetime.utcnow() - start_time).total_seconds()
        logger.error(
            f"Request failed: {request.method} {request.url.path} - "
            f"Error: {str(e)} - Time: {process_time:.3f}s"
        )
        raise


# Endpoint de health check para o pacote routes
@main_router.get(
    "/health",
    tags=["Sistema"],
    summary="Health Check da API",
    description="Verifica o status e saúde da API JurisOracle"
)
async def health_check() -> Dict[str, Any]:
    """
    Endpoint para verificação de saúde da API.
    
    Returns:
        Dict: Status da API e informações do sistema
    """
    return {
        "status": "healthy",
        "service": "JurisOracle API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "routes_loaded": len(ROUTE_MODULES),
        "environment": "production"
    }


# Endpoint de documentação das rotas
@main_router.get(
    "/routes-info",
    tags=["Sistema"],
    summary="Informações das Rotas",
    description="Lista todas as rotas registradas e suas configurações"
)
async def routes_info() -> Dict[str, Any]:
    """
    Endpoint que retorna informações sobre todas as rotas registradas.
    
    Returns:
        Dict: Informações detalhadas das rotas
    """
    routes_info_list = []
    
    for route_config in ROUTE_MODULES:
        routes_info_list.append({
            "prefix": route_config["prefix"],
            "tags": route_config["tags"],
            "description": route_config["description"],
            "full_path": f"/api/v1{route_config['prefix']}"
        })
    
    return {
        "total_route_modules": len(ROUTE_MODULES),
        "api_version": "v1",
        "base_prefix": "/api/v1",
        "routes": routes_info_list,
        "timestamp": datetime.utcnow().isoformat()
    }


# Exportação do router principal
router = main_router

# Lista de routers disponíveis para importação externa
__all__ = [
    "router",
    "main_router",
    "ROUTE_MODULES",
    "auth_router",
    "users_router", 
    "documents_router",
    "search_router",
    "analysis_router",
    "training_router",
    "dashboard_router"
]

# Log de inicialização do módulo
logger.info(
    f"Routes package initialized successfully. "
    f"Loaded {len(ROUTE_MODULES)} route modules."
)