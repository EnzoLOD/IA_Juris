# This file is intentionally left blank.
"""
Configuration module for JurisOracle system.
Centralized configuration management with automatic validation and logging setup.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Import configuration modules
from .settings import Settings, get_settings, EnvironmentType
from .logging_config import LoggingConfig, setup_logging

# Package metadata
__version__ = "1.0.0"
__author__ = "JurisOracle Team"
__description__ = "Configuration management for JurisOracle legal AI system"

# Global configuration instances
_settings: Optional[Settings] = None
_logging_config: Optional[LoggingConfig] = None
_is_initialized: bool = False

# Logger for this module
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class ConfigManager:
    """
    Central configuration manager for JurisOracle system.
    Handles initialization, validation, and provides unified access to all configurations.
    """
    
    def __init__(self):
        self._settings: Optional[Settings] = None
        self._logging_config: Optional[LoggingConfig] = None
        self._initialized: bool = False
        
    @property
    def settings(self) -> Settings:
        """Get settings instance, initializing if necessary."""
        if self._settings is None:
            self._settings = get_settings()
        return self._settings
    
    @property
    def logging_config(self) -> LoggingConfig:
        """Get logging configuration instance."""
        if self._logging_config is None:
            self._logging_config = LoggingConfig()
        return self._logging_config
    
    @property
    def is_initialized(self) -> bool:
        """Check if configuration manager is initialized."""
        return self._initialized
    
    def initialize(self, force_reload: bool = False) -> None:
        """
        Initialize configuration manager with validation and logging setup.
        
        Args:
            force_reload: Force reloading of configurations even if already initialized
        """
        if self._initialized and not force_reload:
            logger.debug("Configuration already initialized, skipping...")
            return
            
        try:
            logger.info("🚀 Initializing JurisOracle configuration...")
            
            # Load and validate settings
            self._load_settings()
            
            # Setup logging
            self._setup_logging()
            
            # Validate configuration integrity
            self._validate_configuration()
            
            # Create required directories
            self._create_directories()
            
            # Log configuration summary
            self._log_configuration_summary()
            
            self._initialized = True
            logger.info("✅ Configuration initialization completed successfully")
            
        except Exception as e:
            error_msg = f"❌ Failed to initialize configuration: {str(e)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
    
    def _load_settings(self) -> None:
        """Load and validate application settings."""
        try:
            logger.debug("Loading application settings...")
            self._settings = get_settings()
            logger.debug(f"Settings loaded for environment: {self._settings.ENVIRONMENT}")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load settings: {str(e)}") from e
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        try:
            logger.debug("Setting up logging configuration...")
            self._logging_config = LoggingConfig()
            
            # Setup logging with current settings
            setup_logging(
                level=self.settings.LOG_LEVEL,
                log_file=self.settings.LOG_FILE,
                environment=self.settings.ENVIRONMENT
            )
            
            logger.debug("Logging configuration completed")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to setup logging: {str(e)}") from e
    
    def _validate_configuration(self) -> None:
        """Validate configuration integrity and dependencies."""
        logger.debug("Validating configuration integrity...")
        
        # Validate required API keys based on enabled features
        if self.settings.OPENAI_ENABLED and not self.settings.OPENAI_API_KEY:
            raise ConfigurationError("OpenAI API key is required when OpenAI is enabled")
        
        # Validate database configuration
        if self.settings.DATABASE_URL == "sqlite:///./juris_oracle.db":
            logger.warning("⚠️  Using default SQLite database. Consider using PostgreSQL for production.")
        
        # Validate vector store configuration
        if self.settings.VECTOR_STORE_TYPE not in ["chroma", "faiss", "pinecone"]:
            raise ConfigurationError(f"Unsupported vector store type: {self.settings.VECTOR_STORE_TYPE}")
        
        if self.settings.VECTOR_STORE_TYPE == "pinecone" and not self.settings.PINECONE_API_KEY:
            raise ConfigurationError("Pinecone API key is required when using Pinecone vector store")
        
        # Validate environment-specific settings
        if self.settings.ENVIRONMENT == EnvironmentType.PRODUCTION:
            self._validate_production_config()
        
        logger.debug("Configuration validation completed")
    
    def _validate_production_config(self) -> None:
        """Validate production-specific configuration requirements."""
        logger.debug("Validating production configuration...")
        
        # Security validations
        if self.settings.SECRET_KEY == "dev-secret-key-change-in-production":
            raise ConfigurationError("SECRET_KEY must be changed from default value in production")
        
        if self.settings.DEBUG:
            logger.warning("⚠️  DEBUG is enabled in production environment")
        
        # Performance validations
        if self.settings.WORKERS < 2:
            logger.warning("⚠️  Consider using more workers in production for better performance")
        
        # Database validations
        if "sqlite" in self.settings.DATABASE_URL.lower():
            logger.warning("⚠️  SQLite is not recommended for production. Consider PostgreSQL.")
    
    def _create_directories(self) -> None:
        """Create required directories if they don't exist."""
        logger.debug("Creating required directories...")
        
        directories_to_create = [
            self.settings.DATA_DIR,
            self.settings.MODELS_DIR,
            self.settings.LOGS_DIR,
            Path(self.settings.DATA_DIR) / "raw",
            Path(self.settings.DATA_DIR) / "processed",
            Path(self.settings.DATA_DIR) / "embeddings",
            Path(self.settings.DATA_DIR) / "vector_stores"
        ]
        
        for directory in directories_to_create:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {directory}")
    
    def _log_configuration_summary(self) -> None:
        """Log a summary of the current configuration."""
        logger.info("🔧 Configuration Summary:")
        logger.info(f"   Environment: {self.settings.ENVIRONMENT}")
        logger.info(f"   Debug Mode: {self.settings.DEBUG}")
        logger.info(f"   Log Level: {self.settings.LOG_LEVEL}")
        logger.info(f"   Vector Store: {self.settings.VECTOR_STORE_TYPE}")
        logger.info(f"   Database: {self.settings.DATABASE_URL.split('://')[0]}")
        logger.info(f"   Data Directory: {self.settings.DATA_DIR}")
        logger.info(f"   Models Directory: {self.settings.MODELS_DIR}")
        logger.info(f"   Workers: {self.settings.WORKERS}")
        
        # Log enabled features
        enabled_features = []
        if self.settings.OPENAI_ENABLED:
            enabled_features.append("OpenAI")
        if self.settings.ENABLE_CORS:
            enabled_features.append("CORS")
        if hasattr(self.settings, 'ENABLE_RATE_LIMITING') and self.settings.ENABLE_RATE_LIMITING:
            enabled_features.append("Rate Limiting")
            
        if enabled_features:
            logger.info(f"   Enabled Features: {', '.join(enabled_features)}")
    
    def reload_settings(self) -> None:
        """Reload settings from environment."""
        logger.info("🔄 Reloading configuration...")
        self._settings = None
        self.initialize(force_reload=True)
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary (excluding sensitive information).
        
        Returns:
            Dict containing non-sensitive configuration values
        """
        sensitive_keys = {
            'SECRET_KEY', 'OPENAI_API_KEY', 'PINECONE_API_KEY', 
            'DATABASE_URL', 'REDIS_URL'
        }
        
        config_dict = {}
        for key, value in self.settings.model_dump().items():
            if key not in sensitive_keys:
                config_dict[key] = value
            else:
                config_dict[key] = "***HIDDEN***" if value else None
                
        return config_dict
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform configuration health check.
        
        Returns:
            Dict containing health check results
        """
        health_status = {
            "status": "healthy",
            "initialized": self._initialized,
            "environment": self.settings.ENVIRONMENT if self._settings else "unknown",
            "checks": {
                "settings_loaded": self._settings is not None,
                "logging_configured": self._logging_config is not None,
                "directories_exist": True,
                "api_keys_configured": True
            },
            "warnings": []
        }
        
        try:
            # Check directories
            for directory in [self.settings.DATA_DIR, self.settings.MODELS_DIR, self.settings.LOGS_DIR]:
                if not Path(directory).exists():
                    health_status["checks"]["directories_exist"] = False
                    health_status["warnings"].append(f"Directory missing: {directory}")
            
            # Check API keys
            if self.settings.OPENAI_ENABLED and not self.settings.OPENAI_API_KEY:
                health_status["checks"]["api_keys_configured"] = False
                health_status["warnings"].append("OpenAI API key not configured")
            
            # Overall status
            if not all(health_status["checks"].values()):
                health_status["status"] = "degraded"
                
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = str(e)
        
        return health_status


# Global configuration manager instance
config_manager = ConfigManager()


def initialize_config(force_reload: bool = False) -> None:
    """
    Initialize global configuration.
    
    Args:
        force_reload: Force reloading of configurations
    """
    global _is_initialized
    
    config_manager.initialize(force_reload=force_reload)
    _is_initialized = True


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    return config_manager


def get_current_settings() -> Settings:
    """Get current settings instance."""
    return config_manager.settings


def get_current_logging_config() -> LoggingConfig:
    """Get current logging configuration instance."""
    return config_manager.logging_config


def is_config_initialized() -> bool:
    """Check if configuration is initialized."""
    return config_manager.is_initialized


# Auto-initialize on import (can be disabled by setting environment variable)
if not os.getenv("JURIS_ORACLE_DISABLE_AUTO_INIT", "").lower() in ("true", "1", "yes"):
    try:
        initialize_config()
    except Exception as e:
        print(f"⚠️  Warning: Failed to auto-initialize configuration: {e}", file=sys.stderr)
        print("You can manually initialize using: from src.config import initialize_config; initialize_config()", file=sys.stderr)


# Export main components
__all__ = [
    # Main classes
    "Settings",
    "LoggingConfig",
    "ConfigManager",
    "ConfigurationError",
    
    # Functions
    "get_settings",
    "setup_logging",
    "initialize_config",
    "get_config_manager",
    "get_current_settings", 
    "get_current_logging_config",
    "is_config_initialized",
    
    # Enums
    "EnvironmentType",
    
    # Global instance
    "config_manager",
    
    # Package info
    "__version__",
    "__author__",
    "__description__"
]