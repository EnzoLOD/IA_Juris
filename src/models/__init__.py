# This file marks the models directory as a Python package.
"""
Models module for JurisOracle system.
Contains all data models, schemas, and validation logic.
"""

import logging
from typing import Dict, List, Type, Any, Optional, Union
from datetime import datetime
from enum import Enum

# Core imports
from .base import (
    BaseModel,
    TimestampMixin,
    ValidationError,
    ModelRegistry,
    ModelConfig,
    FieldValidator,
    CustomValidator
)

from .document import (
    Document,
    DocumentType,
    DocumentStatus,
    DocumentMetadata,
    DocumentChunk,
    DocumentVersion,
    ProcessingResult,
    DocumentQuery,
    DocumentFilter,
    DocumentStats,
    BulkDocumentOperation,
    DocumentExport,
    DocumentImport
)

from .query import (
    Query,
    QueryType,
    QueryStatus,
    QueryContext,
    QueryResult,
    QueryMetrics,
    QueryHistory,
    QueryFilter,
    QueryTemplate,
    QuerySuggestion,
    QueryAnalytics,
    HyDEQuery,
    SemanticQuery,
    BooleanQuery
)

# Configure logging
logger = logging.getLogger(__name__)

# Version info
__version__ = "1.0.0"
__author__ = "JurisOracle Team"

# Model registry for dynamic operations
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}

# Schema registry for API documentation
SCHEMA_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Validation registry for custom validators
VALIDATION_REGISTRY: Dict[str, List[FieldValidator]] = {}


class ModelType(Enum):
    """Enumeration of available model types."""
    DOCUMENT = "document"
    QUERY = "query" 
    BASE = "base"
    METADATA = "metadata"
    RESULT = "result"
    ANALYTICS = "analytics"


class ModelManager:
    """Central manager for all model operations."""
    
    def __init__(self):
        self._initialized = False
        self._models: Dict[str, Type[BaseModel]] = {}
        self._schemas: Dict[str, Dict] = {}
        self._validators: Dict[str, List] = {}
        
    def initialize(self) -> None:
        """Initialize the model manager."""
        if self._initialized:
            logger.warning("ModelManager already initialized")
            return
            
        try:
            # Register all models
            self._register_models()
            
            # Generate schemas
            self._generate_schemas()
            
            # Setup validators
            self._setup_validators()
            
            # Validate model integrity
            self._validate_models()
            
            self._initialized = True
            logger.info(f"ModelManager initialized with {len(self._models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize ModelManager: {e}")
            raise
    
    def _register_models(self) -> None:
        """Register all available models."""
        models_to_register = [
            # Base models
            ('BaseModel', BaseModel),
            ('TimestampMixin', TimestampMixin),
            
            # Document models
            ('Document', Document),
            ('DocumentMetadata', DocumentMetadata),
            ('DocumentChunk', DocumentChunk),
            ('DocumentVersion', DocumentVersion),
            ('ProcessingResult', ProcessingResult),
            ('DocumentQuery', DocumentQuery),
            ('DocumentFilter', DocumentFilter),
            ('DocumentStats', DocumentStats),
            ('BulkDocumentOperation', BulkDocumentOperation),
            ('DocumentExport', DocumentExport),
            ('DocumentImport', DocumentImport),
            
            # Query models
            ('Query', Query),
            ('QueryContext', QueryContext),
            ('QueryResult', QueryResult),
            ('QueryMetrics', QueryMetrics),
            ('QueryHistory', QueryHistory),
            ('QueryFilter', QueryFilter),
            ('QueryTemplate', QueryTemplate),
            ('QuerySuggestion', QuerySuggestion),
            ('QueryAnalytics', QueryAnalytics),
            ('HyDEQuery', HyDEQuery),
            ('SemanticQuery', SemanticQuery),
            ('BooleanQuery', BooleanQuery),
        ]
        
        for name, model_class in models_to_register:
            self._models[name] = model_class
            MODEL_REGISTRY[name] = model_class
            logger.debug(f"Registered model: {name}")
    
    def _generate_schemas(self) -> None:
        """Generate JSON schemas for all models."""
        for name, model_class in self._models.items():
            try:
                if hasattr(model_class, 'model_json_schema'):
                    schema = model_class.model_json_schema()
                    self._schemas[name] = schema
                    SCHEMA_REGISTRY[name] = schema
                    logger.debug(f"Generated schema for: {name}")
            except Exception as e:
                logger.warning(f"Failed to generate schema for {name}: {e}")
    
    def _setup_validators(self) -> None:
        """Setup custom validators for models."""
        # Document validators
        document_validators = [
            FieldValidator(
                field="content",
                validator=lambda v: len(v.strip()) > 0,
                message="Document content cannot be empty"
            ),
            FieldValidator(
                field="title",
                validator=lambda v: 3 <= len(v) <= 500,
                message="Title must be between 3 and 500 characters"
            )
        ]
        
        # Query validators  
        query_validators = [
            FieldValidator(
                field="text",
                validator=lambda v: len(v.strip()) > 0,
                message="Query text cannot be empty"
            ),
            FieldValidator(
                field="max_results",
                validator=lambda v: 1 <= v <= 100,
                message="Max results must be between 1 and 100"
            )
        ]
        
        self._validators['Document'] = document_validators
        self._validators['Query'] = query_validators
        VALIDATION_REGISTRY.update(self._validators)
        
        logger.info(f"Setup validators for {len(self._validators)} model types")
    
    def _validate_models(self) -> None:
        """Validate model integrity and relationships."""
        validation_errors = []
        
        for name, model_class in self._models.items():
            try:
                # Check if model has required attributes
                if not hasattr(model_class, '__annotations__'):
                    validation_errors.append(f"{name}: Missing type annotations")
                
                # Check if model inherits from BaseModel
                if not issubclass(model_class, BaseModel):
                    validation_errors.append(f"{name}: Must inherit from BaseModel")
                    
            except Exception as e:
                validation_errors.append(f"{name}: Validation error - {e}")
        
        if validation_errors:
            error_msg = "Model validation failed:\n" + "\n".join(validation_errors)
            logger.error(error_msg)
            raise ValidationError(error_msg)
        
        logger.info("All models validated successfully")
    
    def get_model(self, name: str) -> Optional[Type[BaseModel]]:
        """Get model class by name."""
        return self._models.get(name)
    
    def get_schema(self, name: str) -> Optional[Dict]:
        """Get schema by model name."""
        return self._schemas.get(name)
    
    def get_validators(self, name: str) -> List[FieldValidator]:
        """Get validators by model name."""
        return self._validators.get(name, [])
    
    def list_models(self, model_type: Optional[ModelType] = None) -> List[str]:
        """List all available models, optionally filtered by type."""
        if model_type is None:
            return list(self._models.keys())
        
        # Filter by type (basic implementation)
        filtered = []
        type_mapping = {
            ModelType.DOCUMENT: ['Document', 'DocumentMetadata', 'DocumentChunk', 'DocumentVersion', 'ProcessingResult'],
            ModelType.QUERY: ['Query', 'QueryContext', 'QueryResult', 'QueryMetrics', 'QueryHistory'],
            ModelType.BASE: ['BaseModel', 'TimestampMixin'],
        }
        
        target_models = type_mapping.get(model_type, [])
        for model_name in self._models.keys():
            if any(target in model_name for target in target_models):
                filtered.append(model_name)
        
        return filtered
    
    def create_instance(self, model_name: str, **kwargs) -> Optional[BaseModel]:
        """Create model instance with given parameters."""
        model_class = self.get_model(model_name)
        if not model_class:
            logger.error(f"Model not found: {model_name}")
            return None
        
        try:
            return model_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create {model_name} instance: {e}")
            return None
    
    def validate_data(self, model_name: str, data: Dict[str, Any]) -> bool:
        """Validate data against model schema."""
        model_class = self.get_model(model_name)
        if not model_class:
            return False
        
        try:
            model_class(**data)
            return True
        except Exception as e:
            logger.debug(f"Validation failed for {model_name}: {e}")
            return False
    
    @property
    def is_initialized(self) -> bool:
        """Check if manager is initialized."""
        return self._initialized
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "total_models": len(self._models),
            "total_schemas": len(self._schemas),
            "total_validators": len(self._validators),
            "initialized": self._initialized,
            "timestamp": datetime.utcnow().isoformat()
        }


# Create global model manager instance
model_manager = ModelManager()


def initialize_models() -> None:
    """Initialize all models and schemas."""
    try:
        model_manager.initialize()
        logger.info("Models module initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise


def get_model_info() -> Dict[str, Any]:
    """Get comprehensive model information."""
    return {
        "version": __version__,
        "author": __author__,
        "models": model_manager.list_models(),
        "stats": model_manager.stats,
        "enums": {
            "DocumentType": [e.value for e in DocumentType],
            "DocumentStatus": [e.value for e in DocumentStatus],
            "QueryType": [e.value for e in QueryType],
            "QueryStatus": [e.value for e in QueryStatus],
            "ModelType": [e.value for e in ModelType],
        }
    }


def create_model_instance(model_name: str, **kwargs) -> Optional[BaseModel]:
    """Create model instance (convenience function)."""
    return model_manager.create_instance(model_name, **kwargs)


def validate_model_data(model_name: str, data: Dict[str, Any]) -> bool:
    """Validate data against model (convenience function)."""
    return model_manager.validate_data(model_name, data)


def get_model_schema(model_name: str) -> Optional[Dict]:
    """Get model schema (convenience function)."""
    return model_manager.get_schema(model_name)


# Export all models and utilities
__all__ = [
    # Base classes
    "BaseModel",
    "TimestampMixin",
    "ValidationError",
    "ModelRegistry",
    "ModelConfig",
    "FieldValidator",
    "CustomValidator",
    
    # Document models
    "Document",
    "DocumentType", 
    "DocumentStatus",
    "DocumentMetadata",
    "DocumentChunk",
    "DocumentVersion",
    "ProcessingResult",
    "DocumentQuery",
    "DocumentFilter",
    "DocumentStats",
    "BulkDocumentOperation",
    "DocumentExport",
    "DocumentImport",
    
    # Query models
    "Query",
    "QueryType",
    "QueryStatus", 
    "QueryContext",
    "QueryResult",
    "QueryMetrics",
    "QueryHistory",
    "QueryFilter",
    "QueryTemplate",
    "QuerySuggestion",
    "QueryAnalytics",
    "HyDEQuery",
    "SemanticQuery",
    "BooleanQuery",
    
    # Enums and types
    "ModelType",
    
    # Manager and utilities
    "ModelManager",
    "model_manager",
    "MODEL_REGISTRY",
    "SCHEMA_REGISTRY", 
    "VALIDATION_REGISTRY",
    
    # Functions
    "initialize_models",
    "get_model_info",
    "create_model_instance", 
    "validate_model_data",
    "get_model_schema",
    
    # Module info
    "__version__",
    "__author__"
]


# Auto-initialize when imported
try:
    initialize_models()
except Exception as e:
    logger.warning(f"Failed to auto-initialize models: {e}")
    logger.info("Call initialize_models() manually if needed")


# Log successful import
logger.info(f"Models module loaded successfully (v{__version__})")