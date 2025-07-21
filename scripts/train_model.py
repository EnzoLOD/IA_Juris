#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
SISTEMA DE TREINAMENTO DE MODELOS IA - IA_JURIS (JurisOracle)
=============================================================================

Arquivo: train_model.py
Autor: Sistema IA_Juris
Versão: 2.0.0
Data: 2025

Descrição:
    Script robusto e à prova de falhas para treinamento de modelos de IA
    voltados para análise de processos jurídicos. Suporta diferentes tipos
    de modelos (classificação, NER, resumo) com tratamento completo de exceções.

Funcionalidades:
    - Verificação automática de dependências
    - Tratamento robusto de exceções
    - Validação de dados de entrada
    - Suporte a múltiplos tipos de modelo
    - Monitoramento de performance
    - Salvamento automático de checkpoints
    - Logging detalhado
    - Configuração flexível

Uso:
    python train_model.py --config config.json
    python train_model.py --model-type classification --data-path ./data/
"""

import os
import sys
import json
import logging
import argparse
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import time

# Configuração de warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# VERIFICAÇÃO E IMPORTAÇÃO DE DEPENDÊNCIAS
# =============================================================================

class DependencyChecker:
    """Verificador robusto de dependências do sistema."""
    
    REQUIRED_PACKAGES = {
        'core': [
            'numpy', 'pandas', 'scikit-learn', 'joblib'
        ],
        'ml': [
            'torch', 'transformers', 'datasets'
        ],
        'nlp': [
            'spacy', 'nltk', 'textblob'
        ],
        'data': [
            'sqlalchemy', 'psycopg2-binary'
        ],
        'utils': [
            'tqdm', 'matplotlib', 'seaborn'
        ]
    }
    
    @staticmethod
    def check_python_version() -> bool:
        """Verifica se a versão do Python é compatível."""
        if sys.version_info < (3, 8):
            logging.error(f"Python 3.8+ é necessário. Versão atual: {sys.version}")
            return False
        return True
    
    @staticmethod
    def check_and_import_package(package_name: str, import_name: str = None) -> bool:
        """Verifica e importa um pacote específico."""
        import_name = import_name or package_name
        try:
            __import__(import_name)
            return True
        except ImportError as e:
            logging.warning(f"Pacote '{package_name}' não encontrado: {e}")
            return False
    
    @classmethod
    def check_dependencies(cls, required_groups: List[str] = None) -> Dict[str, bool]:
        """Verifica todas as dependências necessárias."""
        if required_groups is None:
            required_groups = ['core', 'ml']
        
        results = {}
        
        # Verificar versão do Python
        results['python_version'] = cls.check_python_version()
        
        # Verificar pacotes por grupo
        for group in required_groups:
            if group in cls.REQUIRED_PACKAGES:
                for package in cls.REQUIRED_PACKAGES[group]:
                    # Mapeamento especial para alguns pacotes
                    import_map = {
                        'scikit-learn': 'sklearn',
                        'psycopg2-binary': 'psycopg2'
                    }
                    import_name = import_map.get(package, package)
                    results[package] = cls.check_and_import_package(package, import_name)
        
        return results
    
    @classmethod
    def install_missing_packages(cls, missing_packages: List[str]) -> bool:
        """Tenta instalar pacotes em falta automaticamente."""
        try:
            import subprocess
            for package in missing_packages:
                logging.info(f"Tentando instalar: {package}")
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logging.error(f"Falha ao instalar {package}: {result.stderr}")
                    return False
                    
            return True
        except Exception as e:
            logging.error(f"Erro durante instalação automática: {e}")
            return False

# Verificação inicial de dependências
dependency_results = DependencyChecker.check_dependencies()
missing_packages = [pkg for pkg, available in dependency_results.items() 
                   if not available and pkg != 'python_version']

if missing_packages:
    logging.warning(f"Pacotes em falta: {missing_packages}")
    
# Importações seguras
try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    import joblib
except ImportError as e:
    logging.error(f"Erro crítico de importação (core): {e}")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, pipeline
    )
    TORCH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PyTorch/Transformers não disponível: {e}")
    TORCH_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# =============================================================================
# CONFIGURAÇÃO DE LOGGING
# =============================================================================

def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Configura sistema de logging robusto."""
    
    # Criar diretório de logs se não existir
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configurar formato
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Configurar logger principal
    logger = logging.getLogger('IA_Juris_Training')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    # Handler para arquivo (se especificado)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    
    return logger

# =============================================================================
# CLASSES DE CONFIGURAÇÃO
# =============================================================================

@dataclass
class ModelConfig:
    """Configuração para modelos de ML/AI."""
    model_type: str = "classification"  # classification, ner, summarization
    model_name: str = "neuralmind/bert-base-portuguese-cased"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    save_steps: int = 1000
    eval_steps: int = 500
    output_dir: str = "./models"
    
@dataclass
class DataConfig:
    """Configuração para dados de treinamento."""
    data_path: str = "./data"
    train_file: str = "train.json"
    val_file: str = "validation.json"
    test_file: str = "test.json"
    text_column: str = "text"
    label_column: str = "label"
    max_samples: Optional[int] = None
    validation_split: float = 0.2
    test_split: float = 0.1
    
@dataclass
class TrainingConfig:
    """Configuração geral do treinamento."""
    project_name: str = "IA_Juris"
    experiment_name: str = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda
    num_workers: int = 4
    pin_memory: bool = True
    save_best_model: bool = True
    early_stopping_patience: int = 3
    model_config: ModelConfig = ModelConfig()
    data_config: DataConfig = DataConfig()

# =============================================================================
# CLASSES DE DATASET
# =============================================================================

class JuridicalDataset(Dataset):
    """Dataset customizado para textos jurídicos."""
    
    def __init__(self, texts: List[str], labels: List[str] = None, 
                 tokenizer=None, max_length: int = 512):
        """
        Inicializa o dataset.
        
        Args:
            texts: Lista de textos jurídicos
            labels: Lista de labels (opcional para inferência)
            tokenizer: Tokenizer para processamento
            max_length: Tamanho máximo de sequência
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validações
        if not texts:
            raise ValueError("Lista de textos não pode estar vazia")
        
        if labels and len(texts) != len(labels):
            raise ValueError("Número de textos e labels deve ser igual")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retorna um item do dataset."""
        try:
            text = str(self.texts[idx])
            
            # Tokenização
            if self.tokenizer:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                item = {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten()
                }
            else:
                # Fallback sem tokenizer
                item = {'text': text}
            
            # Adicionar label se disponível
            if self.labels:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
            return item
            
        except Exception as e:
            logging.error(f"Erro ao processar item {idx}: {e}")
            raise

# =============================================================================
# CLASSE PRINCIPAL DE TREINAMENTO
# =============================================================================

class ModelTrainer:
    """Classe principal para treinamento de modelos."""
    
    def __init__(self, config: TrainingConfig):
        """Inicializa o trainer com configuração."""
        self.config = config
        self.logger = logging.getLogger('IA_Juris_Training')
        self.device = self._setup_device()
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        
        # Criar diretórios necessários
        self._create_directories()
        
        # Definir seed para reprodutibilidade
        self._set_seed()
    
    def _setup_device(self) -> torch.device:
        """Configura o dispositivo de computação."""
        if not TORCH_AVAILABLE:
            return None
            
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"Usando GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                self.logger.info("Usando CPU")
        else:
            device = torch.device(self.config.device)
            
        return device
    
    def _create_directories(self):
        """Cria diretórios necessários."""
        directories = [
            self.config.model_config.output_dir,
            f"{self.config.model_config.output_dir}/checkpoints",
            f"{self.config.model_config.output_dir}/logs",
            f"{self.config.model_config.output_dir}/metrics"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _set_seed(self):
        """Define seed para reprodutibilidade."""
        import random
        
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        if TORCH_AVAILABLE:
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Carrega e valida dados de treinamento."""
        try:
            data_path = Path(self.config.data_config.data_path)
            
            if not data_path.exists():
                raise FileNotFoundError(f"Diretório de dados não encontrado: {data_path}")
            
            # Tentar carregar arquivos específicos primeiro
            train_file = data_path / self.config.data_config.train_file
            val_file = data_path / self.config.data_config.val_file
            test_file = data_path / self.config.data_config.test_file
            
            if train_file.exists():
                self.logger.info("Carregando dados de arquivos separados...")
                train_df = self._load_file(train_file)
                val_df = self._load_file(val_file) if val_file.exists() else None
                test_df = self._load_file(test_file) if test_file.exists() else None
                
                # Se não tem validação, criar split
                if val_df is None:
                    train_df, val_df = train_test_split(
                        train_df, 
                        test_size=self.config.data_config.validation_split,
                        random_state=self.config.seed,
                        stratify=train_df[self.config.data_config.label_column] if self.config.data_config.label_column in train_df.columns else None
                    )
                
                # Se não tem teste, criar split
                if test_df is None and self.config.data_config.test_split > 0:
                    train_df, test_df = train_test_split(
                        train_df,
                        test_size=self.config.data_config.test_split,
                        random_state=self.config.seed,
                        stratify=train_df[self.config.data_config.label_column] if self.config.data_config.label_column in train_df.columns else None
                    )
            
            else:
                # Procurar por qualquer arquivo de dados
                data_files = list(data_path.glob("*.json")) + list(data_path.glob("*.csv"))
                
                if not data_files:
                    raise FileNotFoundError("Nenhum arquivo de dados encontrado")
                
                self.logger.info(f"Carregando dados de: {data_files[0]}")
                full_df = self._load_file(data_files[0])
                
                # Fazer splits
                train_df, temp_df = train_test_split(
                    full_df,
                    test_size=self.config.data_config.validation_split + self.config.data_config.test_split,
                    random_state=self.config.seed,
                    stratify=full_df[self.config.data_config.label_column] if self.config.data_config.label_column in full_df.columns else None
                )
                
                if self.config.data_config.test_split > 0:
                    val_size = self.config.data_config.validation_split / (self.config.data_config.validation_split + self.config.data_config.test_split)
                    val_df, test_df = train_test_split(
                        temp_df,
                        test_size=1-val_size,
                        random_state=self.config.seed,
                        stratify=temp_df[self.config.data_config.label_column] if self.config.data_config.label_column in temp_df.columns else None
                    )
                else:
                    val_df = temp_df
                    test_df = pd.DataFrame()
            
            # Validar dados
            self._validate_data(train_df, "treino")
            self._validate_data(val_df, "validação")
            if not test_df.empty:
                self._validate_data(test_df, "teste")
            
            # Aplicar limite de amostras se especificado
            if self.config.data_config.max_samples:
                train_df = train_df.head(self.config.data_config.max_samples)
            
            self.logger.info(f"Dados carregados - Treino: {len(train_df)}, Validação: {len(val_df)}, Teste: {len(test_df)}")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {e}")
            raise
    
    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """Carrega arquivo de dados suportando múltiplos formatos."""
        try:
            if file_path.suffix.lower() == '.json':
                df = pd.read_json(file_path, lines=True)
            elif file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Formato de arquivo não suportado: {file_path.suffix}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar arquivo {file_path}: {e}")
            raise
    
    def _validate_data(self, df: pd.DataFrame, dataset_name: str):
        """Valida estrutura dos dados."""
        if df.empty:
            raise ValueError(f"Dataset de {dataset_name} está vazio")
        
        required_columns = [self.config.data_config.text_column]
        if self.config.data_config.label_column:
            required_columns.append(self.config.data_config.label_column)
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colunas ausentes no dataset de {dataset_name}: {missing_columns}")
        
        # Verificar dados nulos
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Dados nulos encontrados em {dataset_name}: {null_counts.to_dict()}")
    
    def prepare_model(self, num_labels: int = None):
        """Prepara modelo e tokenizer."""
        try:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch não está disponível")
            
            model_name = self.config.model_config.model_name
            self.logger.info(f"Carregando modelo: {model_name}")
            
            # Carregar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Carregar modelo baseado no tipo
            if self.config.model_config.model_type == "classification":
                if num_labels is None:
                    raise ValueError("num_labels é obrigatório para classificação")
                
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=num_labels
                )
            
            elif self.config.model_config.model_type == "ner":
                # Para NER, usar modelo base e adicionar cabeça customizada
                self.model = AutoModel.from_pretrained(model_name)
                # TODO: Adicionar cabeça NER customizada
                
            elif self.config.model_config.model_type == "summarization":
                # Para sumarização, usar pipeline específico
                self.model = pipeline("summarization", model=model_name)
                
            else:
                raise ValueError(f"Tipo de modelo não suportado: {self.config.model_config.model_type}")
            
            # Mover modelo para dispositivo
            if hasattr(self.model, 'to') and self.device:
                self.model = self.model.to(self.device)
            
            self.logger.info("Modelo carregado com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao preparar modelo: {e}")
            raise
    
    def prepare_data_loaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Prepara data loaders para treinamento."""
        try:
            # Preparar labels para classificação
            if self.config.model_config.model_type == "classification":
                if self.label_encoder is None:
                    self.label_encoder = LabelEncoder()
                    train_labels = self.label_encoder.fit_transform(train_df[self.config.data_config.label_column])
                else:
                    train_labels = self.label_encoder.transform(train_df[self.config.data_config.label_column])
                
                val_labels = self.label_encoder.transform(val_df[self.config.data_config.label_column])
            else:
                train_labels = None
                val_labels = None
            
            # Criar datasets
            train_dataset = JuridicalDataset(
                texts=train_df[self.config.data_config.text_column].tolist(),
                labels=train_labels,
                tokenizer=self.tokenizer,
                max_length=self.config.model_config.max_length
            )
            
            val_dataset = JuridicalDataset(
                texts=val_df[self.config.data_config.text_column].tolist(),
                labels=val_labels,
                tokenizer=self.tokenizer,
                max_length=self.config.model_config.max_length
            )
            
            # Criar data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.model_config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.model_config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
            
            return train_loader, val_loader
            
        except Exception as e:
            self.logger.error(f"Erro ao preparar data loaders: {e}")
            raise
    
    def train(self) -> Dict[str, Any]:
        """Executa o treinamento completo."""
        try:
            start_time = time.time()
            self.logger.info("Iniciando treinamento...")
            
            # 1. Carregar dados
            train_df, val_df, test_df = self.load_data()
            
            # 2. Preparar modelo
            if self.config.model_config.model_type == "classification":
                num_labels = len(train_df[self.config.data_config.label_column].unique())
                self.prepare_model(num_labels=num_labels)
            else:
                self.prepare_model()
            
            # 3. Preparar data loaders
            train_loader, val_loader = self.prepare_data_loaders(train_df, val_df)
            
            # 4. Configurar argumentos de treinamento
            training_args = TrainingArguments(
                output_dir=self.config.model_config.output_dir,
                num_train_epochs=self.config.model_config.num_epochs,
                per_device_train_batch_size=self.config.model_config.batch_size,
                per_device_eval_batch_size=self.config.model_config.batch_size,
                learning_rate=self.config.model_config.learning_rate,
                weight_decay=self.config.model_config.weight_decay,
                warmup_steps=self.config.model_config.warmup_steps,
                logging_steps=100,
                eval_steps=self.config.model_config.eval_steps,
                save_steps=self.config.model_config.save_steps,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=self.config.save_best_model,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                report_to=None,  # Desabilitar wandb/tensorboard por padrão
                seed=self.config.seed
            )
            
            # 5. Criar trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_loader.dataset,
                eval_dataset=val_loader.dataset,
                tokenizer=self.tokenizer
            )
            
            # 6. Executar treinamento
            train_result = trainer.train()
            
            # 7. Salvar modelo final
            trainer.save_model()
            if self.tokenizer:
                self.tokenizer.save_pretrained(self.config.model_config.output_dir)
            
            # 8. Salvar label encoder se usado
            if self.label_encoder:
                joblib.dump(
                    self.label_encoder,
                    f"{self.config.model_config.output_dir}/label_encoder.pkl"
                )
            
            # 9. Avaliar no conjunto de teste se disponível
            test_results = {}
            if not test_df.empty:
                test_results = self.evaluate_on_test(test_df, trainer)
            
            # 10. Salvar configuração
            self._save_config()
            
            # 11. Calcular tempo total
            total_time = time.time() - start_time
            
            results = {
                'train_results': train_result,
                'test_results': test_results,
                'training_time': total_time,
                'model_path': self.config.model_config.output_dir,
                'config': asdict(self.config)
            }
            
            self.logger.info(f"Treinamento concluído em {total_time:.2f} segundos")
            return results
            
        except Exception as e:
            self.logger.error(f"Erro durante treinamento: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def evaluate_on_test(self, test_df: pd.DataFrame, trainer) -> Dict[str, Any]:
        """Avalia modelo no conjunto de teste."""
        try:
            self.logger.info("Avaliando no conjunto de teste...")
            
            # Preparar dados de teste
            if self.config.model_config.model_type == "classification":
                test_labels = self.label_encoder.transform(test_df[self.config.data_config.label_column])
            else:
                test_labels = None
            
            test_dataset = JuridicalDataset(
                texts=test_df[self.config.data_config.text_column].tolist(),
                labels=test_labels,
                tokenizer=self.tokenizer,
                max_length=self.config.model_config.max_length
            )
            
            # Executar avaliação
            eval_results = trainer.evaluate(eval_dataset=test_dataset)
            
            # Para classificação, gerar relatório detalhado
            if self.config.model_config.model_type == "classification":
                predictions = trainer.predict(test_dataset)
                y_pred = np.argmax(predictions.predictions, axis=1)
                y_true = test_labels
                
                # Relatório de classificação
                class_report = classification_report(
                    y_true, y_pred,
                    target_names=self.label_encoder.classes_,
                    output_dict=True
                )
                
                eval_results['classification_report'] = class_report
                
                # Matriz de confusão
                conf_matrix = confusion_matrix(y_true, y_pred)
                eval_results['confusion_matrix'] = conf_matrix.tolist()
            
            self.logger.info("Avaliação no teste concluída")
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Erro durante avaliação: {e}")
            return {}
    
    def _save_config(self):
        """Salva configuração do treinamento."""
        try:
            config_path = f"{self.config.model_config.output_dir}/training_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuração salva em: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar configuração: {e}")

# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================

def load_config_from_file(config_path: str) -> TrainingConfig:
    """Carrega configuração de arquivo JSON."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Converter dicts aninhados para dataclasses
        if 'model_config' in config_dict:
            config_dict['model_config'] = ModelConfig(**config_dict['model_config'])
        
        if 'data_config' in config_dict:
            config_dict['data_config'] = DataConfig(**config_dict['data_config'])
        
        return TrainingConfig(**config_dict)
        
    except Exception as e:
        logging.error(f"Erro ao carregar configuração: {e}")
        raise

def create_sample_config(output_path: str = "sample_config.json"):
    """Cria arquivo de configuração de exemplo."""
    sample_config = TrainingConfig(
        project_name="IA_Juris_Sample",
        model_config=ModelConfig(
            model_type="classification",
            model_name="neuralmind/bert-base-portuguese-cased",
            max_length=256,
            batch_size=8,
            num_epochs=2
        ),
        data_config=DataConfig(
            data_path="./sample_data",
            text_column="texto_processo",
            label_column="categoria"
        )
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(sample_config), f, indent=2, ensure_ascii=False)
    
    print(f"Configuração de exemplo criada em: {output_path}")

def validate_environment() -> bool:
    """Valida se o ambiente está pronto para treinamento."""
    issues = []
    
    # Verificar dependências
    dep_results = DependencyChecker.check_dependencies(['core', 'ml'])
    missing_deps = [pkg for pkg, available in dep_results.items() if not available]
    
    if missing_deps:
        issues.append(f"Dependências em falta: {missing_deps}")
    
    # Verificar GPU se disponível
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            print(f"✅ GPU disponível: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  GPU não disponível, usando CPU")
    
    # Verificar espaço em disco
    import shutil
    free_space = shutil.disk_usage('.').free / (1024**3)  # GB
    if free_space < 5:
        issues.append(f"Pouco espaço em disco: {free_space:.1f}GB")
    
    if issues:
        print("❌ Problemas encontrados:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Ambiente pronto para treinamento!")
        return True

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """Função principal do script."""
    
    # Configurar argumentos de linha de comando
    parser = argparse.ArgumentParser(
        description="Sistema de Treinamento IA_Juris",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python train_model.py --validate-env
  python train_model.py --create-sample-config
  python train_model.py --config config.json
  python train_model.py --model-type classification --data-path ./data/
        """
    )
    
    parser.add_argument('--config', type=str, help='Caminho para arquivo de configuração JSON')
    parser.add_argument('--model-type', type=str, choices=['classification', 'ner', 'summarization'],
                       default='classification', help='Tipo de modelo a treinar')
    parser.add_argument('--data-path', type=str, default='./data', help='Caminho para dados de treinamento')
    parser.add_argument('--output-dir', type=str, default='./models', help='Diretório para salvar modelo')
    parser.add_argument('--validate-env', action='store_true', help='Validar ambiente antes do treinamento')
    parser.add_argument('--create-sample-config', action='store_true', help='Criar arquivo de configuração de exemplo')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--log-file', type=str, help='Arquivo para salvar logs')
    
    args = parser.parse_args()
    
    # Configurar logging
    logger = setup_logging(args.log_level, args.log_file)
    
    try:
        # Validar ambiente se solicitado
        if args.validate_env:
            validate_environment()
            return
        
        # Criar configuração de exemplo se solicitado
        if args.create_sample_config:
            create_sample_config()
            return
        
        # Carregar ou criar configuração
        if args.config:
            logger.info(f"Carregando configuração de: {args.config}")
            config = load_config_from_file(args.config)
        else:
            logger.info("Criando configuração padrão")
            config = TrainingConfig(
                model_config=ModelConfig(
                    model_type=args.model_type,
                    output_dir=args.output_dir
                ),
                data_config=DataConfig(
                    data_path=args.data_path
                )
            )
        
        # Validar ambiente automaticamente
        if not validate_environment():
            logger.warning("Ambiente pode não estar totalmente preparado")
            response = input("Continuar mesmo assim? (y/N): ")
            if response.lower() != 'y':
                return
        
        # Inicializar e executar treinamento
        logger.info("Inicializando trainer...")
        trainer = ModelTrainer(config)
        
        logger.info("Executando treinamento...")
        results = trainer.train()
        
        # Exibir resultados
        logger.info("=== RESULTADOS DO TREINAMENTO ===")
        logger.info(f"Tempo total: {results['training_time']:.2f} segundos")
        logger.info(f"Modelo salvo em: {results['model_path']}")
        
        if 'test_results' in results and results['test_results']:
            logger.info("Resultados no teste:")
            for key, value in results['test_results'].items():
                if key not in ['classification_report', 'confusion_matrix']:
                    logger.info(f"  {key}: {value}")
        
        logger.info("✅ Treinamento concluído com sucesso!")
        
    except KeyboardInterrupt:
        logger.info("Treinamento interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        sys.exit(1)

# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    main()