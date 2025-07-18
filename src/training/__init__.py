"""
JurisOracle Training Package
===========================

Este pacote fornece ferramentas completas para treinamento, avaliação e fine-tuning
de modelos de IA especializados em análise de documentos jurídicos.

O pacote inclui:
- Preparação e processamento de dados jurídicos
- Avaliação abrangente de modelos
- Fine-tuning especializado com técnicas avançadas (LoRA, QLoRA)
- Métricas específicas para domínio legal
- Pipelines de treinamento otimizados

Exemplo de uso:
    >>> from training import DataPreparator, ModelEvaluator, FineTuner
    >>> 
    >>> # Preparar dados
    >>> preparator = DataPreparator()
    >>> train_data, val_data = preparator.prepare_training_data(documents)
    >>> 
    >>> # Avaliar modelo
    >>> evaluator = ModelEvaluator()
    >>> metrics = evaluator.evaluate_model(model, test_data)
    >>> 
    >>> # Fine-tuning
    >>> trainer = FineTuner()
    >>> model = trainer.fine_tune(base_model, train_data)

Autores: Equipe JurisOracle
Licença: MIT
"""

import logging
import warnings
from typing import Optional, Dict, Any, List

# Configuração de logging para o pacote
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Variáveis de pacote
__version__ = "1.0.0"
__author__ = "Equipe JurisOracle"
__email__ = "dev@jurisoracle.com"
__license__ = "MIT"
__status__ = "Production"

# Metadados do pacote
__package_name__ = "juris_oracle.training"
__description__ = "Pacote de treinamento para modelos de IA jurídica"
__url__ = "https://github.com/jurisoracle/training"

# Configurações padrão do pacote
DEFAULT_CONFIG = {
    "log_level": "INFO",
    "cache_enabled": True,
    "max_workers": 4,
    "batch_size": 16,
    "model_cache_dir": "./models_cache",
    "data_cache_dir": "./data_cache",
}

# Constantes para o domínio jurídico
LEGAL_DOCUMENT_TYPES = [
    "petição", "sentença", "acórdão", "despacho", "decisão",
    "parecer", "contrato", "lei", "decreto", "portaria"
]

LEGAL_AREAS = [
    "civil", "penal", "trabalhista", "tributário", "administrativo",
    "constitucional", "comercial", "ambiental", "consumidor"
]

# Supressão de avisos desnecessários durante importação
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

try:
    # Importações principais do módulo de preparação de dados
    from .data_preparation import (
        DataPreparator,
        LegalDocumentProcessor,
        DatasetBuilder,
        DocumentTokenizer,
        prepare_legal_dataset,
        validate_training_data,
        split_dataset,
    )
    
    # Importações principais do módulo de avaliação
    from .evaluation import (
        ModelEvaluator,
        LegalMetrics,
        BenchmarkSuite,
        PerformanceAnalyzer,
        evaluate_model,
        compute_legal_metrics,
        generate_evaluation_report,
        compare_models,
    )
    
    # Importações principais do módulo de fine-tuning
    from .fine_tuning import (
        FineTuner,
        LoRATrainer,
        QLoRATrainer,
        TrainingConfig,
        ModelOptimizer,
        fine_tune_legal_model,
        setup_training_config,
        monitor_training_progress,
        save_trained_model,
    )
    
    # Flag indicando que todas as importações foram bem-sucedidas
    _IMPORTS_SUCCESSFUL = True
    
except ImportError as e:
    # Log do erro de importação sem quebrar o pacote
    logging.warning(f"Algumas funcionalidades podem não estar disponíveis: {e}")
    _IMPORTS_SUCCESSFUL = False
    
    # Importações básicas como fallback
    DataPreparator = None
    ModelEvaluator = None
    FineTuner = None

# API Pública - Classes principais
__all__ = [
    # Classes principais
    "DataPreparator",
    "ModelEvaluator", 
    "FineTuner",
    
    # Classes especializadas de preparação de dados
    "LegalDocumentProcessor",
    "DatasetBuilder",
    "DocumentTokenizer",
    
    # Classes especializadas de avaliação
    "LegalMetrics",
    "BenchmarkSuite", 
    "PerformanceAnalyzer",
    
    # Classes especializadas de fine-tuning
    "LoRATrainer",
    "QLoRATrainer",
    "TrainingConfig",
    "ModelOptimizer",
    
    # Funções utilitárias
    "prepare_legal_dataset",
    "validate_training_data",
    "split_dataset",
    "evaluate_model",
    "compute_legal_metrics",
    "generate_evaluation_report",
    "compare_models",
    "fine_tune_legal_model",
    "setup_training_config",
    "monitor_training_progress",
    "save_trained_model",
    
    # Constantes e configurações
    "DEFAULT_CONFIG",
    "LEGAL_DOCUMENT_TYPES",
    "LEGAL_AREAS",
    
    # Metadados
    "__version__",
    "__author__",
    "__license__",
]

# Funções de conveniência para o pacote
def get_version() -> str:
    """Retorna a versão do pacote training."""
    return __version__

def get_config() -> Dict[str, Any]:
    """Retorna a configuração padrão do pacote."""
    return DEFAULT_CONFIG.copy()

def set_log_level(level: str) -> None:
    """
    Define o nível de logging para o pacote training.
    
    Args:
        level: Nível de log ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logger = logging.getLogger(__name__)
    logger.setLevel(numeric_level)
    
    # Atualiza configuração
    DEFAULT_CONFIG["log_level"] = level.upper()

def check_dependencies() -> Dict[str, bool]:
    """
    Verifica se todas as dependências necessárias estão instaladas.
    
    Returns:
        Dict com status de cada dependência
    """
    dependencies = {}
    
    try:
        import torch
        dependencies["torch"] = True
    except ImportError:
        dependencies["torch"] = False
    
    try:
        import transformers
        dependencies["transformers"] = True
    except ImportError:
        dependencies["transformers"] = False
    
    try:
        import datasets
        dependencies["datasets"] = True
    except ImportError:
        dependencies["datasets"] = False
    
    try:
        import peft
        dependencies["peft"] = True
    except ImportError:
        dependencies["peft"] = False
    
    try:
        import accelerate
        dependencies["accelerate"] = True
    except ImportError:
        dependencies["accelerate"] = False
    
    return dependencies

def validate_environment() -> bool:
    """
    Valida se o ambiente está configurado corretamente para treinamento.
    
    Returns:
        True se o ambiente está válido, False caso contrário
    """
    if not _IMPORTS_SUCCESSFUL:
        logging.error("Falha nas importações principais do pacote")
        return False
    
    deps = check_dependencies()
    missing = [dep for dep, available in deps.items() if not available]
    
    if missing:
        logging.error(f"Dependências faltando: {', '.join(missing)}")
        return False
    
    return True

# Configuração inicial do pacote
def _initialize_package():
    """Inicialização interna do pacote."""
    logger = logging.getLogger(__name__)
    
    if _IMPORTS_SUCCESSFUL:
        logger.info(f"JurisOracle Training Package v{__version__} carregado com sucesso")
    else:
        logger.warning("Pacote carregado com funcionalidades limitadas")
    
    # Validação opcional do ambiente
    if not validate_environment():
        logger.warning("Ambiente pode não estar completamente configurado")

# Executa inicialização
_initialize_package()

# Funções de factory para facilitar criação de objetos
def create_data_preparator(**kwargs) -> Optional["DataPreparator"]:
    """
    Factory function para criar um DataPreparator com configurações padrão.
    
    Args:
        **kwargs: Argumentos específicos para o DataPreparator
        
    Returns:
        Instância de DataPreparator ou None se não disponível
    """
    if DataPreparator is None:
        logging.error("DataPreparator não disponível")
        return None
    
    config = get_config()
    config.update(kwargs)
    return DataPreparator(**config)

def create_model_evaluator(**kwargs) -> Optional["ModelEvaluator"]:
    """
    Factory function para criar um ModelEvaluator com configurações padrão.
    
    Args:
        **kwargs: Argumentos específicos para o ModelEvaluator
        
    Returns:
        Instância de ModelEvaluator ou None se não disponível
    """
    if ModelEvaluator is None:
        logging.error("ModelEvaluator não disponível")
        return None
    
    config = get_config()
    config.update(kwargs)
    return ModelEvaluator(**config)

def create_fine_tuner(**kwargs) -> Optional["FineTuner"]:
    """
    Factory function para criar um FineTuner com configurações padrão.
    
    Args:
        **kwargs: Argumentos específicos para o FineTuner
        
    Returns:
        Instância de FineTuner ou None se não disponível
    """
    if FineTuner is None:
        logging.error("FineTuner não disponível")
        return None
    
    config = get_config()
    config.update(kwargs)
    return FineTuner(**config)

# Adicionando factory functions ao __all__
__all__.extend([
    "get_version",
    "get_config", 
    "set_log_level",
    "check_dependencies",
    "validate_environment",
    "create_data_preparator",
    "create_model_evaluator",
    "create_fine_tuner",
])

# Limpeza de variáveis internas
del logging, warnings, Optional, Dict, Any, List