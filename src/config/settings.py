DATABASE_URL = "sqlite:///./database.db"
API_KEY = "your_api_key_here"
DEBUG = True
LOG_LEVEL = "INFO"

# Add other configuration settings as needed
"""
Configurações centralizadas do sistema JurisOracle.
"""
from typing import Optional, List
from pydantic import BaseSettings, validator
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Configurações principais do sistema."""
    
    # API Settings
    app_name: str = "JurisOracle"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Database
    database_url: str
    redis_url: str = "redis://localhost:6379"
    
    # AI Models
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    
    # Embedding Model
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384
    
    # Vector Store
    vector_store_type: str = "chroma"  # chroma, faiss, pinecone
    chroma_persist_directory: str = "./data/chroma"
    pinecone_environment: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    
    # Document Processing
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: List[str] = [".pdf", ".docx", ".txt"]
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # HyDE Configuration
    hyde_enabled: bool = True
    hyde_prompt_template: str = """
    Baseado na pergunta abaixo, escreva um documento hipotético que responderia perfeitamente a esta pergunta.
    O documento deve ser detalhado, técnico e usar terminologia jurídica apropriada.
    
    Pergunta: {question}
    
    Documento hipotético:
    """
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    
    # Monitoring
    sentry_dsn: Optional[str] = None
    prometheus_enabled: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    @validator("database_url")
    def validate_database_url(cls, v):
        if not v:
            raise ValueError("DATABASE_URL é obrigatório")
        return v
    
    @validator("secret_key")
    def validate_secret_key(cls, v):
        if not v or len(v) < 32:
            raise ValueError("SECRET_KEY deve ter pelo menos 32 caracteres")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Retorna instância singleton das configurações."""
    return Settings()


# Configurações específicas para diferentes ambientes
class DevelopmentSettings(Settings):
    debug: bool = True
    log_level: str = "DEBUG"


class ProductionSettings(Settings):
    debug: bool = False
    log_level: str = "WARNING"


class TestingSettings(Settings):
    database_url: str = "sqlite:///./test.db"
    redis_url: str = "redis://localhost:6379/1"
    secret_key: str = "test-secret-key-32-characters-long"


def get_settings_for_environment(env: str = None) -> Settings:
    """Retorna configurações baseadas no ambiente."""
    env = env or os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()