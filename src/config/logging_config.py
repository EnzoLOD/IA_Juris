"""
Configuração de logging estruturado para o sistema.
(See <attachments> above for file contents. You may not need to search or read the file again.)
"""
import logging
import sys
from typing import Dict, Any
import structlog
from structlog.stdlib import LoggerFactory
from .settings import get_settings

settings = get_settings()


def configure_logging() -> None:
    """Configura o sistema de logging estruturado."""
    
    # Configuração do structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.log_format == "json"
            else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configuração do logging padrão
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )
    
    # Configuração de loggers específicos
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


def get_logger(name: str = None) -> structlog.stdlib.BoundLogger:
    """Retorna um logger estruturado."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin para adicionar logging a classes."""
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Retorna logger para a classe."""
        return get_logger(self.__class__.__name__)

