"""
Configurações centralizadas do JurisOracle
Sistema de configuração baseado em Pydantic Settings com suporte a múltiplos ambientes
"""

import os
import secrets
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from functools import lru_cache

from pydantic import BaseSettings, Field, validator, root_validator
from pydantic.networks import AnyHttpUrl, PostgresDsn
from enum import Enum

# Diretório base do projeto
BASE_DIR = Path(__file__).parent.parent.parent

class Environment(str, Enum):
    """Ambientes suportados"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Níveis de log suportados"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class DeviceType(str, Enum):
    """Tipos de dispositivo para ML"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon

class Settings(BaseSettings):
    """
    Configurações principais do JurisOracle
    
    As configurações são carregadas na seguinte ordem de prioridade:
    1. Variáveis de ambiente
    2. Arquivo .env
    3. Valores padrão definidos aqui
    """
    
    # ==========================================
    # CONFIGURAÇÕES GERAIS DA APLICAÇÃO
    # ==========================================
    
    APP_NAME: str = Field(default="JurisOracle", description="Nome da aplicação")
    APP_VERSION: str = Field(default="1.0.0", description="Versão da aplicação")
    APP_DESCRIPTION: str = Field(
        default="Sistema de Análise de Documentos Jurídicos com IA",
        description="Descrição da aplicação"
    )
    
    ENVIRONMENT: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Ambiente de execução"
    )
    
    DEBUG: bool = Field(
        default=True,
        description="Modo debug (apenas para desenvolvimento)"
    )
    
    # ==========================================
    # CONFIGURAÇÕES DE REDE E API
    # ==========================================
    
    HOST: str = Field(default="0.0.0.0", description="Host da API")
    PORT: int = Field(default=8000, ge=1, le=65535, description="Porta da API")
    
    # URLs permitidas para CORS
    ALLOWED_ORIGINS: List[AnyHttpUrl] = Field(
        default=[
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000"
        ],
        description="URLs permitidas para CORS"
    )
    
    # Configurações de upload
    MAX_FILE_SIZE: int = Field(
        default=50 * 1024 * 1024,  # 50MB
        description="Tamanho máximo de arquivo em bytes"
    )
    
    ALLOWED_FILE_TYPES: List[str] = Field(
        default=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "application/msword"
        ],
        description="Tipos MIME permitidos para upload"
    )
    
    # ==========================================
    # CONFIGURAÇÕES DE SEGURANÇA
    # ==========================================
    
    # Chave secreta para JWT (deve ser definida via variável de ambiente em produção)
    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Chave secreta para JWT"
    )
    
    # Algoritmo JWT
    JWT_ALGORITHM: str = Field(default="HS256", description="Algoritmo JWT")
    
    # Tempo de expiração do token (em minutos)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60 * 24 * 7,  # 7 dias
        description="Tempo de expiração do token em minutos"
    )
    
    # API Keys para serviços externos
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="Chave da API OpenAI (opcional)"
    )
    
    HUGGINGFACE_API_KEY: Optional[str] = Field(
        default=None,
        description="Chave da API HuggingFace (opcional)"
    )
    
    # Configurações de rate limiting
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Habilitar rate limiting"
    )
    
    DEFAULT_RATE_LIMIT: str = Field(
        default="100/hour",
        description="Rate limit padrão"
    )
    
    # ==========================================
    # CONFIGURAÇÕES DE BANCO DE DADOS
    # ==========================================
    
    # PostgreSQL principal
    DB_HOST: str = Field(default="localhost", description="Host do banco de dados")
    DB_PORT: int = Field(default=5432, ge=1, le=65535, description="Porta do banco")
    DB_NAME: str = Field(default="jurisoracle_db", description="Nome do banco")
    DB_USER: str = Field(default="jurisoracle_user", description="Usuário do banco")
    DB_PASSWORD: str = Field(default="", description="Senha do banco")
    DB_SCHEMA: str = Field(default="public", description="Schema do banco")
    
    # Pool de conexões
    DB_POOL_SIZE: int = Field(default=10, ge=1, le=100, description="Tamanho do pool")
    DB_MAX_OVERFLOW: int = Field(default=20, ge=0, le=100, description="Overflow máximo")
    DB_POOL_TIMEOUT: int = Field(default=30, ge=1, description="Timeout do pool")
    
    # SSL
    DB_SSL_MODE: str = Field(default="prefer", description="Modo SSL do banco")
    
    @property
    def DATABASE_URL(self) -> str:
        """URL completa do banco de dados"""
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    # Redis para cache
    REDIS_HOST: str = Field(default="localhost", description="Host do Redis")
    REDIS_PORT: int = Field(default=6379, ge=1, le=65535, description="Porta do Redis")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Senha do Redis")
    REDIS_DB: int = Field(default=0, ge=0, le=15, description="Database do Redis")
    REDIS_EXPIRE_TIME: int = Field(
        default=3600,  # 1 hora
        description="Tempo de expiração do cache em segundos"
    )
    
    @property
    def REDIS_URL(self) -> str:
        """URL completa do Redis"""
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # ==========================================
    # CONFIGURAÇÕES DE MACHINE LEARNING
    # ==========================================
    
    # Dispositivo para execução dos modelos
    DEVICE: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Dispositivo para ML (auto/cpu/cuda/mps)"
    )
    
    # Modelos de embedding
    EMBEDDING_MODEL: str = Field(
        default="neuralmind/bert-base-portuguese-cased",
        description="Modelo para embeddings"
    )
    
    EMBEDDING_MODEL_CACHE_DIR: str = Field(
        default=str(BASE_DIR / "data" / "models" / "embeddings"),
        description="Diretório de cache para modelos de embedding"
    )
    
    # Modelo LLM principal
    LLM_MODEL: str = Field(
        default="microsoft/DialoGPT-medium",
        description="Modelo LLM principal"
    )
    
    # Modelo LLM para HyDE
    HYDE_LLM_MODEL: str = Field(
        default="microsoft/DialoGPT-medium",
        description="Modelo LLM para HyDE"
    )
    
    LLM_MODEL_CACHE_DIR: str = Field(
        default=str(BASE_DIR / "data" / "models" / "llm"),
        description="Diretório de cache para modelos LLM"
    )
    
    # Configurações de geração
    LLM_MAX_LENGTH: int = Field(
        default=512,
        ge=50,
        le=2048,
        description="Tamanho máximo de geração"
    )
    
    LLM_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.1,
        le=2.0,
        description="Temperatura de geração"
    )
    
    LLM_TOP_P: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Top-p para geração"
    )
    
    # ==========================================
    # CONFIGURAÇÕES DO PIPELINE RAG
    # ==========================================
    
    # Vector Store
    VECTOR_STORE_PATH: str = Field(
        default=str(BASE_DIR / "data" / "vector_store"),
        description="Caminho para o vector store"
    )
    
    VECTOR_STORE_TYPE: str = Field(
        default="faiss",
        regex="^(faiss|chroma|pinecone)$",
        description="Tipo de vector store"
    )
    
    # Configurações de chunking
    CHUNK_SIZE: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Tamanho dos chunks de texto"
    )
    
    CHUNK_OVERLAP: int = Field(
        default=200,
        ge=0,
        le=500,
        description="Sobreposição entre chunks"
    )
    
    # Configurações de retrieval
    DEFAULT_K: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número padrão de documentos recuperados"
    )
    
    DEFAULT_SCORE_THRESHOLD: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Score mínimo padrão para retrieval"
    )
    
    # ==========================================
    # CONFIGURAÇÕES DE PROCESSAMENTO
    # ==========================================
    
    # Processamento de documentos
    DOCUMENT_CACHE_DIR: str = Field(
        default=str(BASE_DIR / "data" / "document_cache"),
        description="Diretório de cache de documentos"
    )
    
    MAX_PROCESSING_WORKERS: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Número máximo de workers para processamento"
    )
    
    ENABLE_OCR: bool = Field(
        default=False,
        description="Habilitar OCR para documentos digitalizados"
    )
    
    OCR_LANGUAGE: str = Field(
        default="por",
        description="Idioma para OCR (por=português)"
    )
    
    # Processamento NLP
    ENABLE_NLP_PROCESSING: bool = Field(
        default=True,
        description="Habilitar processamento NLP avançado"
    )
    
    SPACY_MODEL: str = Field(
        default="pt_core_news_sm",
        description="Modelo spaCy para português"
    )
    
    # ==========================================
    # CONFIGURAÇÕES DE LOGGING E MONITORAMENTO
    # ==========================================
    
    LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO,
        description="Nível de log"
    )
    
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Formato dos logs"
    )
    
    LOG_FILE_PATH: Optional[str] = Field(
        default=None,
        description="Caminho para arquivo de log (None = apenas console)"
    )
    
    LOG_ROTATION: str = Field(
        default="1 day",
        description="Rotação de logs"
    )
    
    LOG_RETENTION: str = Field(
        default="30 days",
        description="Retenção de logs"
    )
    
    # Métricas e monitoramento
    ENABLE_METRICS: bool = Field(
        default=True,
        description="Habilitar coleta de métricas"
    )
    
    METRICS_PORT: int = Field(
        default=9090,
        ge=1,
        le=65535,
        description="Porta para métricas Prometheus"
    )
    
    # Sentry para error tracking
    SENTRY_DSN: Optional[str] = Field(
        default=None,
        description="DSN do Sentry para tracking de erros"
    )
    
    SENTRY_ENVIRONMENT: str = Field(
        default="development",
        description="Ambiente do Sentry"
    )
    
    # ==========================================
    # CONFIGURAÇÕES DE CACHE E PERFORMANCE
    # ==========================================
    
    # Cache de embeddings
    EMBEDDING_CACHE_SIZE: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Tamanho do cache de embeddings"
    )
    
    # Cache de modelos
    MODEL_CACHE_SIZE: str = Field(
        default="2GB",
        description="Tamanho máximo do cache de modelos"
    )
    
    # Configurações de batch processing
    BATCH_SIZE: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Tamanho do batch para processamento"
    )
    
    # Timeout para operações
    REQUEST_TIMEOUT: int = Field(
        default=300,  # 5 minutos
        ge=30,
        le=3600,
        description="Timeout para requisições em segundos"
    )
    
    # ==========================================
    # CONFIGURAÇÕES ESPECÍFICAS POR AMBIENTE
    # ==========================================
    
    @validator("DEBUG")
    def validate_debug_mode(cls, v, values):
        """Debug deve ser False em produção"""
        env = values.get("ENVIRONMENT")
        if env == Environment.PRODUCTION and v:
            raise ValueError("DEBUG não pode ser True em produção")
        return v
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v, values):
        """Valida se a secret key é segura em produção"""
        env = values.get("ENVIRONMENT")
        if env == Environment.PRODUCTION:
            if len(v) < 32:
                raise ValueError("SECRET_KEY deve ter pelo menos 32 caracteres em produção")
            if v == "your-secret-key" or "secret" in v.lower():
                raise ValueError("SECRET_KEY insegura detectada em produção")
        return v
    
    @validator("ALLOWED_ORIGINS")
    def validate_cors_origins(cls, v, values):
        """Valida URLs de CORS em produção"""
        env = values.get("ENVIRONMENT")
        if env == Environment.PRODUCTION:
            dangerous_origins = ["http://localhost", "http://127.0.0.1"]
            for origin in v:
                origin_str = str(origin)
                if any(dangerous in origin_str for dangerous in dangerous_origins):
                    raise ValueError(f"Origin insegura detectada em produção: {origin_str}")
        return v
    
    @root_validator
    def validate_database_config(cls, values):
        """Valida configuração completa do banco"""
        env = values.get("ENVIRONMENT")
        if env == Environment.PRODUCTION:
            required_fields = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
            for field in required_fields:
                if not values.get(field):
                    raise ValueError(f"{field} é obrigatório em produção")
        return values
    
    # ==========================================
    # CONFIGURAÇÕES COMPUTED PROPERTIES
    # ==========================================
    
    @property
    def is_production(self) -> bool:
        """Verifica se está em produção"""
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Verifica se está em desenvolvimento"""
        return self.ENVIRONMENT == Environment.DEVELOPMENT
    
    @property
    def is_testing(self) -> bool:
        """Verifica se está em modo de teste"""
        return self.ENVIRONMENT == Environment.TESTING
    
    @property
    def data_dir(self) -> Path:
        """Diretório de dados"""
        return BASE_DIR / "data"
    
    @property
    def logs_dir(self) -> Path:
        """Diretório de logs"""
        return BASE_DIR / "logs"
    
    @property
    def temp_dir(self) -> Path:
        """Diretório temporário"""
        return BASE_DIR / "temp"
    
    def ensure_directories(self):
        """Cria diretórios necessários se não existirem"""
        directories = [
            self.data_dir,
            self.logs_dir,
            self.temp_dir,
            Path(self.VECTOR_STORE_PATH),
            Path(self.DOCUMENT_CACHE_DIR),
            Path(self.EMBEDDING_MODEL_CACHE_DIR),
            Path(self.LLM_MODEL_CACHE_DIR)
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # CONFIGURAÇÕES DE AMBIENTE ESPECÍFICAS
    # ==========================================
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Retorna configurações específicas do ambiente atual"""
        base_config = {
            "app_name": self.APP_NAME,
            "version": self.APP_VERSION,
            "environment": self.ENVIRONMENT,
            "debug": self.DEBUG
        }
        
        if self.is_development:
            return {
                **base_config,
                "cors_origins": ["*"],  # Permite todas as origens em dev
                "log_level": "DEBUG",
                "enable_reload": True,
                "enable_profiling": True
            }
        
        elif self.is_testing:
            return {
                **base_config,
                "database_url": "sqlite:///./test.db",  # SQLite para testes
                "log_level": "WARNING",
                "enable_cache": False,
                "batch_size": 2  # Smaller batch for faster tests
            }
        
        elif self.is_production:
            return {
                **base_config,
                "log_level": "INFO",
                "enable_metrics": True,
                "enable_sentry": bool(self.SENTRY_DSN),
                "cors_origins": self.ALLOWED_ORIGINS,
                "ssl_required": True
            }
        
        return base_config
    
    # ==========================================
    # CONFIGURAÇÃO DE CLASSE
    # ==========================================
    
    class Config:
        """Configuração do Pydantic"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
        # Permite campos extras para flexibilidade
        extra = "allow"
        
        # Validação de campos
        validate_assignment = True
        
        # Schema extra para documentação
        schema_extra = {
            "example": {
                "APP_NAME": "JurisOracle",
                "ENVIRONMENT": "development",
                "HOST": "0.0.0.0",
                "PORT": 8000,
                "DB_HOST": "localhost",
                "DB_NAME": "jurisoracle_db",
                "EMBEDDING_MODEL": "neuralmind/bert-base-portuguese-cased",
                "DEVICE": "auto"
            }
        }

# ==========================================
# CONFIGURAÇÕES PARA DIFERENTES AMBIENTES
# ==========================================

class DevelopmentSettings(Settings):
    """Configurações específicas para desenvolvimento"""
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    DEBUG: bool = True
    LOG_LEVEL: LogLevel = LogLevel.DEBUG
    
    # Modelos menores para desenvolvimento
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    LLM_MODEL: str = "microsoft/DialoGPT-small"
    
    # Configurações relaxadas para dev
    RATE_LIMIT_ENABLED: bool = False
    MAX_PROCESSING_WORKERS: int = 2

class TestingSettings(Settings):
    """Configurações específicas para testes"""
    ENVIRONMENT: Environment = Environment.TESTING
    DEBUG: bool = False
    LOG_LEVEL: LogLevel = LogLevel.WARNING
    
    # Banco de dados em memória para testes
    DB_NAME: str = "test_jurisoracle_db"
    
    # Modelos mínimos para testes rápidos
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    LLM_MODEL: str = "microsoft/DialoGPT-small"
    
    # Configurações otimizadas para testes
    CHUNK_SIZE: int = 100
    BATCH_SIZE: int = 2
    MAX_PROCESSING_WORKERS: int = 1
    ENABLE_NLP_PROCESSING: bool = False

class ProductionSettings(Settings):
    """Configurações específicas para produção"""
    ENVIRONMENT: Environment = Environment.PRODUCTION
    DEBUG: bool = False
    LOG_LEVEL: LogLevel = LogLevel.INFO
    
    # Modelos de produção
    EMBEDDING_MODEL: str = "neuralmind/bert-large-portuguese-cased"
    LLM_MODEL: str = "microsoft/DialoGPT-large"
    
    # Configurações de segurança para produção
    RATE_LIMIT_ENABLED: bool = True
    SSL_REQUIRED: bool = True
    
    # Performance otimizada
    MAX_PROCESSING_WORKERS: int = 8
    BATCH_SIZE: int = 64

# ==========================================
# FACTORY FUNCTION PARA SETTINGS
# ==========================================

@lru_cache()
def get_settings() -> Settings:
    """
    Factory function para obter configurações baseado no ambiente
    
    Usa cache para evitar recarregamento desnecessário
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "development":
        settings = DevelopmentSettings()
    elif environment == "testing":
        settings = TestingSettings()
    elif environment == "production":
        settings = ProductionSettings()
    elif environment == "staging":
        # Staging usa config de produção com algumas modificações
        settings = ProductionSettings()
        settings.DEBUG = True
        settings.LOG_LEVEL = LogLevel.DEBUG
    else:
        # Default para desenvolvimento
        settings = DevelopmentSettings()
    
    # Cria diretórios necessários
    settings.ensure_directories()
    
    return settings

# ==========================================
# UTILITÁRIOS DE CONFIGURAÇÃO
# ==========================================

def validate_environment() -> bool:
    """
    Valida se o ambiente está configurado corretamente
    
    Returns:
        bool: True se válido, False caso contrário
    """
    try:
        settings = get_settings()
        
        # Validações básicas
        required_dirs = [
            settings.data_dir,
            Path(settings.VECTOR_STORE_PATH).parent,
            Path(settings.DOCUMENT_CACHE_DIR).parent
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                print(f"❌ Diretório não encontrado: {directory}")
                return False
        
        # Validações específicas por ambiente
        if settings.is_production:
            if not settings.SECRET_KEY or len(settings.SECRET_KEY) < 32:
                print("❌ SECRET_KEY inválida para produção")
                return False
            
            if not settings.DB_PASSWORD:
                print("❌ DB_PASSWORD não definida para produção")
                return False
        
        print("✅ Ambiente validado com sucesso")
        return True
        
    except Exception as e:
        print(f"❌ Erro na validação do ambiente: {e}")
        return False

def print_current_config():
    """Imprime configuração atual (sem dados sensíveis)"""
    settings = get_settings()
    
    print(f"""
🔧 Configuração Atual do JurisOracle
{'=' * 50}
Aplicação: {settings.APP_NAME} v{settings.APP_VERSION}
Ambiente: {settings.ENVIRONMENT}
Debug: {settings.DEBUG}
Host: {settings.HOST}:{settings.PORT}

🤖 Machine Learning
Dispositivo: {settings.DEVICE}
Modelo Embedding: {settings.EMBEDDING_MODEL}
Modelo LLM: {settings.LLM_MODEL}

💾 Armazenamento
Vector Store: {settings.VECTOR_STORE_PATH}
Cache Documentos: {settings.DOCUMENT_CACHE_DIR}
Banco de Dados: {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}

⚡ Performance
Workers: {settings.MAX_PROCESSING_WORKERS}
Chunk Size: {settings.CHUNK_SIZE}
Batch Size: {settings.BATCH_SIZE}

🔒 Segurança
Rate Limiting: {settings.RATE_LIMIT_ENABLED}
JWT Algorithm: {settings.JWT_ALGORITHM}
CORS Origins: {len(settings.ALLOWED_ORIGINS)} configuradas
""")

# ==========================================
# EXEMPLO DE ARQUIVO .env
# ==========================================

def create_example_env_file():
    """Cria arquivo .env.example com todas as variáveis"""
    env_content = """
# ==========================================
# CONFIGURAÇÕES GERAIS
# ==========================================
ENVIRONMENT=development
APP_NAME=JurisOracle
DEBUG=true

# ==========================================
# REDE E API
# ==========================================
HOST=0.0.0.0
PORT=8000
ALLOWED_ORIGINS=["http://localhost:3000","http://localhost:8000"]

# ==========================================
# SEGURANÇA
# ==========================================
SECRET_KEY=your-super-secret-key-change-this-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# APIs externas (opcional)
OPENAI_API_KEY=sk-your-openai-key
HUGGINGFACE_API_KEY=hf_your-huggingface-key

# ==========================================
# BANCO DE DADOS
# ==========================================
DB_HOST=localhost
DB_PORT=5432
DB_NAME=jurisoracle_db
DB_USER=jurisoracle_user
DB_PASSWORD=secure_password_here
DB_SCHEMA=public

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# ==========================================
# MACHINE LEARNING
# ==========================================
DEVICE=auto
EMBEDDING_MODEL=neuralmind/bert-base-portuguese-cased
LLM_MODEL=microsoft/DialoGPT-medium
HYDE_LLM_MODEL=microsoft/DialoGPT-medium

# ==========================================
# PROCESSAMENTO
# ==========================================
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_PROCESSING_WORKERS=4
ENABLE_NLP_PROCESSING=true
SPACY_MODEL=pt_core_news_sm

# ==========================================
# LOGS E MONITORAMENTO
# ==========================================
LOG_LEVEL=INFO
ENABLE_METRICS=true
METRICS_PORT=9090
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id

# ==========================================
# CAMINHOS E CACHE
# ==========================================
VECTOR_STORE_PATH=./data/vector_store
DOCUMENT_CACHE_DIR=./data/document_cache
EMBEDDING_MODEL_CACHE_DIR=./data/models/embeddings
LLM_MODEL_CACHE_DIR=./data/models/llm
"""
    
    env_file = BASE_DIR / ".env.example"
    with open(env_file, "w", encoding="utf-8") as f:
        f.write(env_content.strip())
    
    print(f"✅ Arquivo .env.example criado em: {env_file}")

if __name__ == "__main__":
    # Testa configurações
    print("🧪 Testando configurações...")
    
    if validate_environment():
        print_current_config()
    else:
        print("❌ Falha na validação do ambiente")
        create_example_env_file()