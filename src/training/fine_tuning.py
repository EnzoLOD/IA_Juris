"""
Fine-Tuning Module para JurisOracle
====================================

Módulo completo para fine-tuning de modelos no domínio jurídico.
Suporta múltiplas arquiteturas, técnicas avançadas e otimizações.

Author: JurisOracle Team
Version: 1.0.0
"""

import os
import json
import torch
import logging
import warnings
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import gc
import psutil
from tqdm import tqdm
import wandb
from functools import partial
from collections import defaultdict
import shutil
import pickle

# Transformers
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    DataCollatorWithPadding, get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup, AdamW,
    BitsAndBytesConfig, TrainerCallback
)

# PEFT para LoRA
from peft import (
    LoraConfig, get_peft_model, TaskType,
    prepare_model_for_kbit_training, PeftModel
)

# Datasets
from datasets import Dataset, DatasetDict, load_metric
import evaluate

# Scientific
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.distributed import init_process_group, destroy_process_group

# Local imports
from ..core.logger import get_logger
from ..core.config import Config
from ..models.embeddings import JuridicalEmbeddings
from ..models.hyde import HyDEModel
from .data_preparation import DataPreprocessor
from .evaluation import ModelEvaluator

# Warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger(__name__)

@dataclass
class FineTuningConfig:
    """Configuração para fine-tuning"""
    
    # Model settings
    model_name: str = "neuralmind/bert-base-portuguese-cased"
    model_type: str = "bert"  # bert, roberta, gpt, t5, etc.
    task_type: str = "classification"  # classification, qa, summarization, embedding
    
    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # LoRA settings
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    
    # Quantization
    use_quantization: bool = False
    quantization_bits: int = 4
    
    # Advanced settings
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    fp16_opt_level: str = "O1"
    dataloader_num_workers: int = 4
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    logging_steps: int = 10
    save_total_limit: int = 3
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Data settings
    max_length: int = 512
    test_size: float = 0.2
    validation_size: float = 0.1
    cross_validation_folds: int = 5
    
    # Legal-specific
    preserve_legal_entities: bool = True
    legal_vocab_extension: bool = True
    domain_adaptation: bool = True
    
    # Output settings
    output_dir: str = "fine_tuned_models"
    run_name: str = "juridical_fine_tune"
    logging_dir: str = "logs"
    
    # Experimental
    use_gradient_checkpointing: bool = True
    use_cpu_offload: bool = False
    use_deepspeed: bool = False
    
class CustomLegalLoss(nn.Module):
    """Loss function customizada para domínio jurídico"""
    
    def __init__(self, task_type: str, alpha: float = 0.7):
        super().__init__()
        self.task_type = task_type
        self.alpha = alpha
        
    def forward(self, predictions, labels, legal_weights=None):
        """Forward pass com weights específicos para termos jurídicos"""
        
        if self.task_type == "classification":
            base_loss = F.cross_entropy(predictions, labels, reduction='none')
        elif self.task_type == "regression":
            base_loss = F.mse_loss(predictions, labels, reduction='none')
        else:
            base_loss = F.cross_entropy(predictions.view(-1, predictions.size(-1)), 
                                      labels.view(-1), reduction='none')
        
        if legal_weights is not None:
            # Aplicar weights para termos jurídicos importantes
            weighted_loss = base_loss * legal_weights
            return weighted_loss.mean()
        
        return base_loss.mean()

class LegalMetricsCallback(TrainerCallback):
    """Callback para métricas específicas do domínio jurídico"""
    
    def __init__(self, evaluator: ModelEvaluator):
        self.evaluator = evaluator
        self.best_score = -float('inf')
        
    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        """Executado após cada avaliação"""
        
        try:
            # Calcular métricas jurídicas customizadas
            legal_metrics = self.evaluator.calculate_legal_metrics(
                model, eval_dataloader
            )
            
            # Log das métricas
            for metric_name, value in legal_metrics.items():
                logger.info(f"Legal {metric_name}: {value:.4f}")
                
            # Tracking do melhor modelo
            current_score = legal_metrics.get('legal_f1', 0.0)
            if current_score > self.best_score:
                self.best_score = current_score
                logger.info(f"Novo melhor modelo! Legal F1: {current_score:.4f}")
                
        except Exception as e:
            logger.error(f"Erro no cálculo de métricas jurídicas: {e}")

class MemoryOptimizer:
    """Otimizador de memória para treinamento eficiente"""
    
    @staticmethod
    def optimize_memory():
        """Otimiza uso de memória"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    @staticmethod
    def get_memory_usage():
        """Retorna uso atual de memória"""
        memory_info = {
            'cpu_percent': psutil.virtual_memory().percent,
            'cpu_available_gb': psutil.virtual_memory().available / (1024**3)
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                'gpu_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'gpu_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'gpu_free_gb': (torch.cuda.get_device_properties(0).total_memory - 
                              torch.cuda.memory_reserved()) / (1024**3)
            })
            
        return memory_info

class FineTuner:
    """Classe principal para fine-tuning de modelos jurídicos"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.evaluator = ModelEvaluator()
        self.memory_optimizer = MemoryOptimizer()
        
        # Setup logging
        self.setup_logging()
        
        # Inicializar tracking
        self.training_history = defaultdict(list)
        self.best_metrics = {}
        
        logger.info(f"FineTuner inicializado com device: {self.device}")
        
    def setup_logging(self):
        """Configura logging avançado"""
        
        os.makedirs(self.config.logging_dir, exist_ok=True)
        
        # Configurar WandB se disponível
        try:
            wandb.init(
                project="jurisoracle-finetuning",
                name=self.config.run_name,
                config=self.config.__dict__
            )
            self.use_wandb = True
            logger.info("WandB configurado com sucesso")
        except Exception as e:
            logger.warning(f"WandB não disponível: {e}")
            self.use_wandb = False
            
    def load_model_and_tokenizer(self):
        """Carrega modelo e tokenizer com configurações otimizadas"""
        
        try:
            logger.info(f"Carregando modelo: {self.config.model_name}")
            
            # Configurar quantização se habilitada
            if self.config.use_quantization:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            else:
                quantization_config = None
                
            # Carregar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Adicionar tokens especiais se necessário
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Carregar modelo
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.config.fp16 else torch.float32
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            # Configurar LoRA se habilitado
            if self.config.use_lora:
                self.setup_lora()
                
            # Gradient checkpointing
            if self.config.use_gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                
            # Preparar para quantização se necessário
            if self.config.use_quantization:
                self.model = prepare_model_for_kbit_training(self.model)
                
            logger.info("Modelo e tokenizer carregados com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
            
    def setup_lora(self):
        """Configura LoRA para fine-tuning eficiente"""
        
        try:
            # Mapear task type para PEFT task type
            task_type_mapping = {
                "classification": TaskType.SEQUENCE_CLS,
                "qa": TaskType.QUESTION_ANS,
                "summarization": TaskType.SUMMARIZATION,
                "embedding": TaskType.FEATURE_EXTRACTION
            }
            
            task_type = task_type_mapping.get(
                self.config.task_type, 
                TaskType.SEQUENCE_CLS
            )
            
            # Configurar LoRA
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=task_type
            )
            
            # Aplicar LoRA ao modelo
            self.model = get_peft_model(self.model, lora_config)
            
            # Log dos parâmetros treináveis
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            
            logger.info(f"LoRA configurado - Parâmetros treináveis: {trainable_params:,}")
            logger.info(f"Total de parâmetros: {total_params:,}")
            logger.info(f"Percentual treinável: {100 * trainable_params / total_params:.2f}%")
            
        except Exception as e:
            logger.error(f"Erro ao configurar LoRA: {e}")
            raise
            
    def prepare_dataset(self, data: Union[Dict, List, str]) -> DatasetDict:
        """Prepara dataset para fine-tuning"""
        
        try:
            logger.info("Preparando dataset para fine-tuning")
            
            # Carregar dados se for caminho
            if isinstance(data, str):
                if data.endswith('.json'):
                    with open(data, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                elif data.endswith('.pkl'):
                    with open(data, 'rb') as f:
                        data = pickle.load(f)
                        
            # Preprocessar dados
            preprocessor = DataPreprocessor()
            
            if isinstance(data, dict) and 'train' in data:
                # Dataset já dividido
                train_data = data['train']
                test_data = data.get('test', [])
                val_data = data.get('validation', [])
            else:
                # Dividir dataset
                if isinstance(data, list):
                    train_data, test_data = train_test_split(
                        data, 
                        test_size=self.config.test_size,
                        random_state=42,
                        stratify=None  # TODO: Implementar estratificação se necessário
                    )
                    
                    if self.config.validation_size > 0:
                        train_data, val_data = train_test_split(
                            train_data,
                            test_size=self.config.validation_size,
                            random_state=42
                        )
                    else:
                        val_data = []
                else:
                    raise ValueError("Formato de dados não suportado")
                    
            # Converter para formato adequado
            def process_samples(samples):
                processed = []
                for sample in samples:
                    if isinstance(sample, dict):
                        # Extrair texto e labels
                        text = sample.get('text', '') or sample.get('content', '')
                        label = sample.get('label', 0) or sample.get('target', 0)
                        
                        # Preprocessar texto jurídico
                        text = preprocessor.preprocess_legal_text(text)
                        
                        processed.append({
                            'text': text,
                            'label': label,
                            'metadata': sample.get('metadata', {})
                        })
                    else:
                        processed.append({'text': str(sample), 'label': 0})
                        
                return processed
                
            # Processar cada split
            train_processed = process_samples(train_data)
            test_processed = process_samples(test_data) if test_data else []
            val_processed = process_samples(val_data) if val_data else []
            
            # Tokenizar
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                
            # Criar datasets
            train_dataset = Dataset.from_list(train_processed)
            train_dataset = train_dataset.map(
                tokenize_function, 
                batched=True,
                remove_columns=['text']
            )
            
            dataset_dict = {'train': train_dataset}
            
            if test_processed:
                test_dataset = Dataset.from_list(test_processed)
                test_dataset = test_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=['text']
                )
                dataset_dict['test'] = test_dataset
                
            if val_processed:
                val_dataset = Dataset.from_list(val_processed)
                val_dataset = val_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=['text']
                )
                dataset_dict['validation'] = val_dataset
                
            dataset = DatasetDict(dataset_dict)
            
            logger.info(f"Dataset preparado: {len(dataset['train'])} samples de treino")
            if 'validation' in dataset:
                logger.info(f"Validation: {len(dataset['validation'])} samples")
            if 'test' in dataset:
                logger.info(f"Test: {len(dataset['test'])} samples")
                
            return dataset
            
        except Exception as e:
            logger.error(f"Erro ao preparar dataset: {e}")
            raise
            
    def setup_trainer(self, dataset: DatasetDict) -> Trainer:
        """Configura o Trainer com todas as otimizações"""
        
        try:
            logger.info("Configurando Trainer")
            
            # Criar diretório de output
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                warmup_ratio=self.config.warmup_ratio,
                max_grad_norm=self.config.max_grad_norm,
                
                # Precision
                fp16=self.config.fp16,
                fp16_opt_level=self.config.fp16_opt_level,
                
                # Evaluation
                evaluation_strategy=self.config.evaluation_strategy,
                eval_steps=1 if self.config.evaluation_strategy == "steps" else None,
                
                # Saving
                save_strategy=self.config.save_strategy,
                save_steps=1 if self.config.save_strategy == "steps" else None,
                save_total_limit=self.config.save_total_limit,
                
                # Logging
                logging_dir=self.config.logging_dir,
                logging_steps=self.config.logging_steps,
                report_to="wandb" if self.use_wandb else None,
                run_name=self.config.run_name,
                
                # Data loading
                dataloader_num_workers=self.config.dataloader_num_workers,
                dataloader_pin_memory=True,
                
                # Optimization
                remove_unused_columns=False,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                
                # Advanced
                gradient_checkpointing=self.config.use_gradient_checkpointing,
                ddp_find_unused_parameters=False,
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True,
                return_tensors="pt"
            )
            
            # Callbacks
            callbacks = []
            
            # Early stopping
            if self.config.early_stopping_patience > 0:
                early_stopping = EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
                callbacks.append(early_stopping)
                
            # Métricas jurídicas
            legal_metrics_callback = LegalMetricsCallback(self.evaluator)
            callbacks.append(legal_metrics_callback)
            
            # Compute metrics function
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                
                if self.config.task_type == "classification":
                    predictions = np.argmax(predictions, axis=1)
                    
                    # Métricas básicas
                    accuracy = np.mean(predictions == labels)
                    
                    # Usar sklearn para métricas detalhadas
                    try:
                        from sklearn.metrics import f1_score, precision_score, recall_score
                        
                        f1 = f1_score(labels, predictions, average='weighted')
                        precision = precision_score(labels, predictions, average='weighted')
                        recall = recall_score(labels, predictions, average='weighted')
                        
                        return {
                            'accuracy': accuracy,
                            'f1': f1,
                            'precision': precision,
                            'recall': recall
                        }
                    except:
                        return {'accuracy': accuracy}
                        
                elif self.config.task_type == "regression":
                    mse = np.mean((predictions - labels) ** 2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(predictions - labels))
                    
                    return {
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae
                    }
                    
                return {}
                
            # Criar Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset.get('validation') or dataset.get('test'),
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                callbacks=callbacks
            )
            
            logger.info("Trainer configurado com sucesso")
            return trainer
            
        except Exception as e:
            logger.error(f"Erro ao configurar Trainer: {e}")
            raise
            
    def train_model(self, data: Union[Dict, List, str]) -> Dict[str, Any]:
        """Treina o modelo com monitoramento completo"""
        
        try:
            logger.info("=== Iniciando Fine-Tuning ===")
            start_time = datetime.now()
            
            # Carregar modelo se necessário
            if self.model is None:
                self.load_model_and_tokenizer()
                
            # Preparar dataset
            dataset = self.prepare_dataset(data)
            
            # Configurar trainer
            self.trainer = self.setup_trainer(dataset)
            
            # Log de informações iniciais
            memory_info = self.memory_optimizer.get_memory_usage()
            logger.info(f"Memória antes do treino: {memory_info}")
            
            # Treinar modelo
            logger.info("Iniciando treinamento...")
            train_result = self.trainer.train()
            
            # Salvar modelo final
            final_model_path = os.path.join(
                self.config.output_dir, 
                "final_model"
            )
            self.trainer.save_model(final_model_path)
            
            # Avaliação final
            if 'test' in dataset:
                logger.info("Executando avaliação final...")
                eval_result = self.trainer.evaluate(dataset['test'])
            else:
                eval_result = self.trainer.evaluate()
                
            # Calcular tempo total
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Compilar resultados
            results = {
                'training_time_seconds': training_time,
                'final_train_loss': train_result.training_loss,
                'final_eval_metrics': eval_result,
                'model_path': final_model_path,
                'config': self.config.__dict__,
                'memory_usage': self.memory_optimizer.get_memory_usage(),
                'training_history': dict(self.training_history)
            }
            
            # Log final
            logger.info("=== Fine-Tuning Concluído ===")
            logger.info(f"Tempo total: {training_time/60:.2f} minutos")
            logger.info(f"Loss final: {train_result.training_loss:.4f}")
            logger.info(f"Modelo salvo em: {final_model_path}")
            
            # Limpeza de memória
            self.memory_optimizer.optimize_memory()
            
            return results
            
        except Exception as e:
            logger.error(f"Erro durante o treinamento: {e}")
            raise
            
    def cross_validate(self, data: Union[Dict, List, str]) -> Dict[str, Any]:
        """Executa validação cruzada para robustez"""
        
        try:
            logger.info("=== Iniciando Validação Cruzada ===")
            
            # Preparar dados
            if isinstance(data, str):
                with open(data, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
            if isinstance(data, dict) and 'samples' in data:
                samples = data['samples']
            elif isinstance(data, list):
                samples = data
            else:
                raise ValueError("Formato de dados inválido")
                
            # Configurar K-Fold
            kfold = KFold(
                n_splits=self.config.cross_validation_folds,
                shuffle=True,
                random_state=42
            )
            
            cv_results = []
            fold_metrics = defaultdict(list)
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(samples)):
                logger.info(f"=== Fold {fold + 1}/{self.config.cross_validation_folds} ===")
                
                # Dividir dados
                train_samples = [samples[i] for i in train_idx]
                val_samples = [samples[i] for i in val_idx]
                
                fold_data = {
                    'train': train_samples,
                    'validation': val_samples
                }
                
                # Recarregar modelo para cada fold
                self.model = None
                self.load_model_and_tokenizer()
                
                # Treinar no fold
                fold_results = self.train_model(fold_data)
                
                # Armazenar resultados
                cv_results.append(fold_results)
                
                # Agregar métricas
                for metric, value in fold_results['final_eval_metrics'].items():
                    fold_metrics[metric].append(value)
                    
                logger.info(f"Fold {fold + 1} concluído")
                
            # Calcular estatísticas finais
            final_metrics = {}
            for metric, values in fold_metrics.items():
                final_metrics[f"{metric}_mean"] = np.mean(values)
                final_metrics[f"{metric}_std"] = np.std(values)
                final_metrics[f"{metric}_min"] = np.min(values)
                final_metrics[f"{metric}_max"] = np.max(values)
                
            cv_summary = {
                'cv_metrics': final_metrics,
                'fold_results': cv_results,
                'num_folds': self.config.cross_validation_folds,
                'config': self.config.__dict__
            }
            
            logger.info("=== Validação Cruzada Concluída ===")
            for metric in ['accuracy', 'f1', 'precision', 'recall']:
                if f"{metric}_mean" in final_metrics:
                    mean_val = final_metrics[f"{metric}_mean"]
                    std_val = final_metrics[f"{metric}_std"]
                    logger.info(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
                    
            return cv_summary
            
        except Exception as e:
            logger.error(f"Erro na validação cruzada: {e}")
            raise
            
    def hyperparameter_search(self, 
                            data: Union[Dict, List, str],
                            param_grid: Dict[str, List],
                            n_trials: int = 20) -> Dict[str, Any]:
        """Busca de hiperparâmetros com Optuna"""
        
        try:
            import optuna
            
            logger.info("=== Iniciando Busca de Hiperparâmetros ===")
            
            def objective(trial):
                # Sugerir parâmetros
                config = FineTuningConfig()
                
                for param, values in param_grid.items():
                    if hasattr(config, param):
                        if isinstance(values[0], float):
                            setattr(config, param, trial.suggest_float(
                                param, min(values), max(values)
                            ))
                        elif isinstance(values[0], int):
                            setattr(config, param, trial.suggest_int(
                                param, min(values), max(values)
                            ))
                        else:
                            setattr(config, param, trial.suggest_categorical(
                                param, values
                            ))
                            
                # Treinar com configuração atual
                tuner = FineTuner(config)
                results = tuner.train_model(data)
                
                # Retornar métrica objetivo
                return results['final_eval_metrics'].get('eval_f1', 0.0)
                
            # Executar otimização
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            # Resultados
            best_params = study.best_params
            best_score = study.best_value
            
            logger.info(f"Melhor score: {best_score:.4f}")
            logger.info(f"Melhores parâmetros: {best_params}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'study': study,
                'trials': study.trials
            }
            
        except ImportError:
            logger.error("Optuna não instalado. Install: pip install optuna")
            raise
        except Exception as e:
            logger.error(f"Erro na busca de hiperparâmetros: {e}")
            raise
            
    def load_fine_tuned_model(self, model_path: str):
        """Carrega modelo fine-tuned"""
        
        try:
            logger.info(f"Carregando modelo fine-tuned: {model_path}")
            
            # Carregar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Carregar modelo
            if self.config.use_lora:
                # Carregar modelo base primeiro
                base_model = AutoModel.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32
                )
                
                # Carregar adaptador LoRA
                self.model = PeftModel.from_pretrained(base_model, model_path)
            else:
                self.model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.config.fp16 else torch.float32
                )
                
            self.model.to(self.device)
            logger.info("Modelo carregado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
            
    def export_model(self, 
                    model_path: str, 
                    export_format: str = "onnx",
                    quantize: bool = False) -> str:
        """Exporta modelo para diferentes formatos"""
        
        try:
            logger.info(f"Exportando modelo para {export_format}")
            
            if self.model is None:
                self.load_fine_tuned_model(model_path)
                
            export_path = f"{model_path}_exported.{export_format}"
            
            if export_format == "onnx":
                # Exportar para ONNX
                dummy_input = torch.randint(
                    0, self.tokenizer.vocab_size, 
                    (1, self.config.max_length)
                ).to(self.device)
                
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    export_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input_ids'],
                    output_names=['output'],
                    dynamic_axes={
                        'input_ids': {0: 'batch_size', 1: 'sequence'},
                        'output': {0: 'batch_size'}
                    }
                )
                
            elif export_format == "torchscript":
                # Exportar para TorchScript
                traced_model = torch.jit.trace(
                    self.model, 
                    torch.randint(0, self.tokenizer.vocab_size, (1, self.config.max_length)).to(self.device)
                )
                traced_model.save(export_path)
                
            else:
                raise ValueError(f"Formato não suportado: {export_format}")
                
            # Quantização se solicitada
            if quantize and export_format == "onnx":
                try:
                    from onnxruntime.quantization import quantize_dynamic, QuantType
                    
                    quantized_path = f"{model_path}_quantized.onnx"
                    quantize_dynamic(
                        export_path,
                        quantized_path,
                        weight_type=QuantType.QUInt8
                    )
                    
                    logger.info(f"Modelo quantizado salvo em: {quantized_path}")
                    return quantized_path
                    
                except ImportError:
                    logger.warning("ONNXRuntime não disponível para quantização")
                    
            logger.info(f"Modelo exportado para: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Erro ao exportar modelo: {e}")
            raise
            
    def generate_training_report(self, results: Dict[str, Any]) -> str:
        """Gera relatório detalhado do treinamento"""
        
        try:
            report_path = os.path.join(
                self.config.output_dir,
                f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>JurisOracle Fine-Tuning Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ color: #2E86C1; border-bottom: 2px solid #2E86C1; }}
                    .metric {{ background: #F8F9FA; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                    .config {{ background: #E8F6F3; padding: 15px; border-radius: 5px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1 class="header">🏛️ JurisOracle Fine-Tuning Report</h1>
                
                <h2>📊 Métricas Finais</h2>
                <div class="metric">
                    <strong>Training Loss:</strong> {results.get('final_train_loss', 'N/A'):.4f}
                </div>
            """
            
            # Adicionar métricas de avaliação
            eval_metrics = results.get('final_eval_metrics', {})
            for metric, value in eval_metrics.items():
                html_content += f"""
                <div class="metric">
                    <strong>{metric.replace('eval_', '').title()}:</strong> {value:.4f}
                </div>
                """
                
            # Configuração
            html_content += f"""
                <h2>⚙️ Configuração</h2>
                <div class="config">
                    <table>
                        <tr><th>Parâmetro</th><th>Valor</th></tr>
            """
            
            for key, value in self.config.__dict__.items():
                html_content += f"<tr><td>{key}</td><td>{value}</td></tr>"
                
            html_content += """
                    </table>
                </div>
                
                <h2>💾 Informações do Modelo</h2>
                <div class="metric">
                    <strong>Modelo Base:</strong> {self.config.model_name}
                </div>
                <div class="metric">
                    <strong>Caminho Final:</strong> {results.get('model_path', 'N/A')}
                </div>
                <div class="metric">
                    <strong>Tempo de Treinamento:</strong> {results.get('training_time_seconds', 0)/60:.2f} minutos
                </div>
                
                <h2>📈 Uso de Memória</h2>
            """
            
            memory_info = results.get('memory_usage', {})
            for key, value in memory_info.items():
                html_content += f"""
                <div class="metric">
                    <strong>{key.replace('_', ' ').title()}:</strong> {value:.2f}
                </div>
                """
                
            html_content += """
                <footer style="margin-top: 50px; text-align: center; color: #666;">
                    <p>Relatório gerado pelo JurisOracle - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
                </footer>
            </body>
            </html>
            """
            
            # Salvar relatório
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            logger.info(f"Relatório gerado: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
            return ""
            
    def cleanup(self):
        """Limpeza de recursos"""
        
        try:
            # Limpeza de memória
            self.memory_optimizer.optimize_memory()
            
            # Fechar WandB
            if self.use_wandb:
                wandb.finish()
                
            logger.info("Limpeza de recursos concluída")
            
        except Exception as e:
            logger.error(f"Erro na limpeza: {e}")

# Factory Functions
def create_classification_config(**kwargs) -> FineTuningConfig:
    """Cria configuração para classificação jurídica"""
    
    config = FineTuningConfig(
        task_type="classification",
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=5,
        max_length=512,
        **kwargs
    )
    
    return config

def create_qa_config(**kwargs) -> FineTuningConfig:
    """Cria configuração para QA jurídico"""
    
    config = FineTuningConfig(
        task_type="qa",
        learning_rate=3e-5,
        batch_size=8,
        num_epochs=3,
        max_length=1024,
        target_modules=["query", "value", "dense"],
        **kwargs
    )
    
    return config

def create_summarization_config(**kwargs) -> FineTuningConfig:
    """Cria configuração para sumarização jurídica"""
    
    config = FineTuningConfig(
        task_type="summarization",
        model_name="pierreguillou/gpt2-small-portuguese",
        learning_rate=5e-5,
        batch_size=4,
        num_epochs=8,
        max_length=2048,
        **kwargs
    )
    
    return config

# Utilities
def estimate_training_time(
    dataset_size: int,
    batch_size: int,
    num_epochs: int,
    model_params: int
) -> Dict[str, float]:
    """Estima tempo de treinamento"""
    
    # Estimativas baseadas em benchmarks
    base_time_per_sample = 0.1  # segundos por sample (estimativa)
    
    if model_params > 1e9:  # Modelos grandes
        base_time_per_sample *= 3
    elif model_params > 100e6:  # Modelos médios
        base_time_per_sample *= 2
        
    batches_per_epoch = dataset_size // batch_size
    total_batches = batches_per_epoch * num_epochs
    
    estimated_seconds = total_batches * base_time_per_sample
    
    return {
        'estimated_seconds': estimated_seconds,
        'estimated_minutes': estimated_seconds / 60,
        'estimated_hours': estimated_seconds / 3600,
        'batches_per_epoch': batches_per_epoch,
        'total_batches': total_batches
    }

if __name__ == "__main__":
    # Exemplo de uso
    
    # Configuração para classificação
    config = create_classification_config(
        model_name="neuralmind/bert-base-portuguese-cased",
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=5,
        use_lora=True,
        run_name="legal_classification"
    )
    
    # Dados de exemplo
    sample_data = [
        {
            "text": "Este é um processo de direito civil sobre contratos.",
            "label": 0,  # civil
            "metadata": {"tribunal": "TJSP", "ano": 2023}
        },
        {
            "text": "Processo criminal por furto qualificado.",
            "label": 1,  # criminal
            "metadata": {"tribunal": "TJRJ", "ano": 2023}
        }
    ]
    
    # Inicializar fine-tuner
    fine_tuner = FineTuner(config)
    
    try:
        # Executar treinamento
        results = fine_tuner.train_model(sample_data)
        
        # Gerar relatório
        report_path = fine_tuner.generate_training_report(results)
        
        print(f"✅ Treinamento concluído!")
        print(f"📊 Relatório: {report_path}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        
    finally:
        # Limpeza
        fine_tuner.cleanup()