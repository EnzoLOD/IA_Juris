"""
Training Service for JurisOracle system.

This module provides comprehensive training capabilities for:
- Custom embeddings models
- Legal language models fine-tuning
- HyDE system training
- Model evaluation and validation
- Distributed training support
- Experiment tracking
"""

import asyncio
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import AdamW, lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import wandb
from tqdm import tqdm

from ..config import get_settings, get_logger
from ..models import Document, Query, TrainingData, ModelMetrics
from ..core import EmbeddingsService, VectorStore, HydeRetriever
from ..utils import TextProcessor, ValidationError, ProcessingError
from ..utils.exceptions import TrainingError, ModelNotFoundError

logger = get_logger(__name__)
settings = get_settings()


class TrainingStatus(Enum):
    """Training status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class ModelType(Enum):
    """Model type enumeration."""
    EMBEDDINGS = "embeddings"
    LANGUAGE_MODEL = "language_model"
    HYDE_GENERATOR = "hyde_generator"
    CLASSIFIER = "classifier"
    SUMMARIZER = "summarizer"


class TrainingStrategy(Enum):
    """Training strategy enumeration."""
    FULL_FINE_TUNING = "full_fine_tuning"
    LORA = "lora"
    QLORA = "qlora"
    ADAPTER = "adapter"
    PROMPT_TUNING = "prompt_tuning"


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Model Configuration
    model_name: str
    model_type: ModelType
    base_model_path: Optional[str] = None
    
    # Training Strategy
    strategy: TrainingStrategy = TrainingStrategy.FULL_FINE_TUNING
    
    # Hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    max_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    
    # Data Configuration
    train_data_path: str = ""
    validation_data_path: str = ""
    test_data_path: str = ""
    max_sequence_length: int = 512
    validation_split: float = 0.2
    
    # Training Options
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 4
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    save_total_limit: int = 3
    
    # Early Stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Distributed Training
    use_distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    # Experiment Tracking
    experiment_name: str = ""
    run_name: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Output Paths
    output_dir: str = ""
    checkpoint_dir: str = ""
    logs_dir: str = ""


@dataclass
class TrainingMetrics:
    """Training metrics container."""
    
    # Training Metrics
    train_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    train_f1: List[float] = field(default_factory=list)
    
    # Validation Metrics
    val_loss: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    val_f1: List[float] = field(default_factory=list)
    val_precision: List[float] = field(default_factory=list)
    val_recall: List[float] = field(default_factory=list)
    
    # Learning Metrics
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    
    # Best Metrics
    best_val_loss: float = float('inf')
    best_val_accuracy: float = 0.0
    best_val_f1: float = 0.0
    best_epoch: int = 0
    
    # Training Info
    total_epochs: int = 0
    total_steps: int = 0
    training_time: float = 0.0


@dataclass
class TrainingJob:
    """Training job container."""
    
    job_id: str = field(default_factory=lambda: str(uuid4()))
    config: TrainingConfig = None
    status: TrainingStatus = TrainingStatus.PENDING
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    model_path: Optional[str] = None
    checkpoints: List[str] = field(default_factory=list)


class LegalDataset(Dataset):
    """Custom dataset for legal documents."""
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize text
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
        
        # Add labels if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


class TrainingCallback:
    """Custom training callback."""
    
    def __init__(self, training_service: 'TrainingService', job: TrainingJob):
        self.training_service = training_service
        self.job = job
        self.best_metric = 0.0
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float]):
        """Called at the end of each epoch."""
        self.job.metrics.train_loss.append(logs.get('train_loss', 0.0))
        self.job.metrics.val_loss.append(logs.get('val_loss', 0.0))
        self.job.metrics.val_accuracy.append(logs.get('val_accuracy', 0.0))
        self.job.metrics.val_f1.append(logs.get('val_f1', 0.0))
        
        # Update best metrics
        val_f1 = logs.get('val_f1', 0.0)
        if val_f1 > self.job.metrics.best_val_f1:
            self.job.metrics.best_val_f1 = val_f1
            self.job.metrics.best_val_accuracy = logs.get('val_accuracy', 0.0)
            self.job.metrics.best_val_loss = logs.get('val_loss', 0.0)
            self.job.metrics.best_epoch = epoch
        
        # Log to tracking services
        self.training_service._log_metrics(self.job.job_id, logs, epoch)
    
    def on_train_end(self, logs: Dict[str, float]):
        """Called at the end of training."""
        self.job.status = TrainingStatus.COMPLETED
        self.job.completed_at = datetime.now()
        
        # Calculate total training time
        if self.job.started_at:
            self.job.metrics.training_time = (
                self.job.completed_at - self.job.started_at
            ).total_seconds()


class TrainingService:
    """Training service for JurisOracle models."""
    
    def __init__(self):
        """Initialize training service."""
        self.jobs: Dict[str, TrainingJob] = {}
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.text_processor = TextProcessor()
        self.executor = ThreadPoolExecutor(max_workers=settings.TRAINING_MAX_WORKERS)
        
        # Training directories
        self.models_dir = Path(settings.MODELS_DIR)
        self.checkpoints_dir = Path(settings.CHECKPOINTS_DIR)
        self.logs_dir = Path(settings.LOGS_DIR)
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking services
        self._init_tracking_services()
        
        logger.info("TrainingService initialized successfully")
    
    def _init_tracking_services(self):
        """Initialize experiment tracking services."""
        try:
            # Initialize MLflow
            if settings.MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
                mlflow.set_experiment("juris-oracle-training")
            
            # Initialize Weights & Biases
            if settings.WANDB_PROJECT:
                wandb.init(
                    project=settings.WANDB_PROJECT,
                    entity=settings.WANDB_ENTITY
                )
            
            logger.info("Experiment tracking services initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize tracking services: {e}")
    
    async def start_training(
        self,
        config: TrainingConfig,
        priority: int = 0
    ) -> str:
        """Start a new training job."""
        try:
            # Validate configuration
            self._validate_config(config)
            
            # Create training job
            job = TrainingJob(config=config)
            job.status = TrainingStatus.PENDING
            
            # Generate paths
            job.config.output_dir = str(self.models_dir / job.job_id)
            job.config.checkpoint_dir = str(self.checkpoints_dir / job.job_id)
            job.config.logs_dir = str(self.logs_dir / job.job_id)
            
            # Create directories
            Path(job.config.output_dir).mkdir(parents=True, exist_ok=True)
            Path(job.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            Path(job.config.logs_dir).mkdir(parents=True, exist_ok=True)
            
            # Store job
            self.jobs[job.job_id] = job
            
            # Start training task
            task = asyncio.create_task(self._run_training(job))
            self.active_jobs[job.job_id] = task
            
            logger.info(f"Training job {job.job_id} started")
            return job.job_id
            
        except Exception as e:
            logger.error(f"Error starting training: {e}")
            raise TrainingError(f"Failed to start training: {e}")
    
    def _validate_config(self, config: TrainingConfig):
        """Validate training configuration."""
        if not config.model_name:
            raise ValidationError("Model name is required")
        
        if not config.train_data_path and not config.validation_data_path:
            raise ValidationError("Training or validation data path is required")
        
        if config.batch_size <= 0:
            raise ValidationError("Batch size must be positive")
        
        if config.learning_rate <= 0:
            raise ValidationError("Learning rate must be positive")
        
        if config.max_epochs <= 0:
            raise ValidationError("Max epochs must be positive")
    
    async def _run_training(self, job: TrainingJob):
        """Run training job."""
        try:
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.now()
            logger.info(f"Starting training for job {job.job_id}")
            # Load and prepare data
            train_data, val_data, test_data = await self._prepare_data(job.config)
            # Initialize model and tokenizer
            model, tokenizer = self._initialize_model(job.config)
            # Create datasets
            train_dataset = self._create_dataset(train_data, tokenizer, job.config)
            val_dataset = self._create_dataset(val_data, tokenizer, job.config)
            # Train model
            await self._train_model(job, model, tokenizer, train_dataset, val_dataset)
            # Evaluate model
            if test_data:
                test_dataset = self._create_dataset(test_data, tokenizer, job.config)
                await self._evaluate_model(job, model, tokenizer, test_dataset)
            # Save final model
            self._save_model(job, model, tokenizer)
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.now()
            logger.info(f"Training job {job.job_id} completed successfully")
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            logger.error(f"Training job {job.job_id} failed: {e}")
            raise TrainingError(f"Training failed: {e}")
        finally:
            # Clean up
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    async def _prepare_data(
        self, 
        config: TrainingConfig
    ) -> Tuple[List[Dict], List[Dict], Optional[List[Dict]]]:
        """Prepare training data."""
        logger.info("Preparing training data...")
        
        # Load data from files
        train_data = []
        val_data = []
        test_data = None
        
        if config.train_data_path:
            train_data = self._load_data(config.train_data_path)
        
        if config.validation_data_path:
            val_data = self._load_data(config.validation_data_path)
        elif train_data:
            # Split training data
            train_data, val_data = train_test_split(
                train_data, 
                test_size=config.validation_split,
                random_state=42
            )
        
        if config.test_data_path:
            test_data = self._load_data(config.test_data_path)
        
        # Data preprocessing
        train_data = [self._preprocess_sample(sample) for sample in train_data]
        val_data = [self._preprocess_sample(sample) for sample in val_data]
        
        if test_data:
            test_data = [self._preprocess_sample(sample) for sample in test_data]
        
        logger.info(f"Data prepared - Train: {len(train_data)}, Val: {len(val_data)}")
        
        return train_data, val_data, test_data
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load data from file."""
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif data_path.suffix == '.jsonl':
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        elif data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
            return df.to_dict('records')
        else:
            raise ValueError(f"Unsupported data format: {data_path.suffix}")
    
    def _preprocess_sample(self, sample: Dict) -> Dict:
        """Preprocess a single data sample."""
        # Clean and normalize text
        if 'text' in sample:
            sample['text'] = self.text_processor.clean_text(sample['text'])
            sample['text'] = self.text_processor.normalize_legal_text(sample['text'])
        
        # Process query-answer pairs
        if 'query' in sample and 'answer' in sample:
            sample['query'] = self.text_processor.clean_text(sample['query'])
            sample['answer'] = self.text_processor.clean_text(sample['answer'])
        
        return sample
    
    def _initialize_model(self, config: TrainingConfig) -> Tuple[nn.Module, Any]:
        """Initialize model and tokenizer."""
        logger.info(f"Initializing model: {config.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Add special tokens if needed
        special_tokens = ['[LEGAL]', '[JURISPRUDENCE]', '[ARTICLE]']
        tokenizer.add_tokens(special_tokens)
        
        # Load model based on type
        if config.model_type == ModelType.EMBEDDINGS:
            model = AutoModel.from_pretrained(config.model_name)
        elif config.model_type == ModelType.CLASSIFIER:
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=config.num_labels if hasattr(config, 'num_labels') else 2
            )
        else:
            model = AutoModel.from_pretrained(config.model_name)
        
        # Resize token embeddings
        model.resize_token_embeddings(len(tokenizer))
        
        # Apply training strategy
        if config.strategy == TrainingStrategy.LORA:
            model = self._apply_lora(model, config)
        elif config.strategy == TrainingStrategy.QLORA:
            model = self._apply_qlora(model, config)
        
        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        logger.info(f"Model initialized on device: {device}")
        
        return model, tokenizer
    
    def _apply_lora(self, model: nn.Module, config: TrainingConfig) -> nn.Module:
        """Apply LoRA (Low-Rank Adaptation) to model."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            # Select TaskType based on model_type
            if config.model_type == ModelType.CLASSIFIER:
                task_type = TaskType.SEQ_CLS
            elif config.model_type == ModelType.LANGUAGE_MODEL:
                task_type = TaskType.CAUSAL_LM
            else:
                task_type = TaskType.FEATURE_EXTRACTION

            lora_config = LoraConfig(
                task_type=task_type,
                inference_mode=False,
                r=getattr(config, 'lora_r', 16),
                lora_alpha=getattr(config, 'lora_alpha', 32),
                lora_dropout=getattr(config, 'lora_dropout', 0.1)
            )
            model = get_peft_model(model, lora_config)
            logger.info(f"LoRA applied to model with task_type: {task_type}")
        except ImportError as e:
            logger.warning(f"PEFT library not available ({e}), using full fine-tuning")
        except Exception as e:
            logger.error(f"Error applying LoRA: {e}")
        return model

    def _apply_qlora(self, model: nn.Module, config: TrainingConfig) -> nn.Module:
        """Apply QLoRA (Quantized LoRA) to model."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            from transformers import BitsAndBytesConfig
            # Select TaskType based on model_type
            if config.model_type == ModelType.CLASSIFIER:
                task_type = TaskType.SEQ_CLS
            elif config.model_type == ModelType.LANGUAGE_MODEL:
                task_type = TaskType.CAUSAL_LM
            else:
                task_type = TaskType.FEATURE_EXTRACTION
            # Quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            # LoRA config
            lora_config = LoraConfig(
                task_type=task_type,
                inference_mode=False,
                r=getattr(config, 'lora_r', 16),
                lora_alpha=getattr(config, 'lora_alpha', 32),
                lora_dropout=getattr(config, 'lora_dropout', 0.1)
            )
            # Re-load model with quantization config if possible
            try:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(config.model_name, quantization_config=quantization_config)
                logger.info("Model loaded with BitsAndBytes quantization config for QLoRA")
            except Exception as e:
                logger.warning(f"Could not reload model with quantization config: {e}")
            model = get_peft_model(model, lora_config)
            logger.info(f"QLoRA applied to model with task_type: {task_type}")
        except ImportError as e:
            logger.warning(f"PEFT or BitsAndBytes library not available ({e})")
        except Exception as e:
            logger.error(f"Error applying QLoRA: {e}")
        return model
    
    def _create_dataset(
        self, 
        data: List[Dict], 
        tokenizer: Any, 
        config: TrainingConfig
    ) -> Dataset:
        """Create dataset from data."""
        texts = []
        labels = []
        
        for sample in data:
            if 'text' in sample:
                texts.append(sample['text'])
            elif 'query' in sample and 'answer' in sample:
                texts.append(f"{sample['query']} {tokenizer.sep_token} {sample['answer']}")
            
            if 'label' in sample:
                labels.append(sample['label'])
        
        return LegalDataset(
            texts=texts,
            labels=labels if labels else None,
            tokenizer=tokenizer,
            max_length=config.max_sequence_length
        )
    
    async def _train_model(
        self,
        job: TrainingJob,
        model: nn.Module,
        tokenizer: Any,
        train_dataset: Dataset,
        val_dataset: Dataset
    ):
        """Train the model."""
        logger.info(f"Starting model training for job {job.job_id}")
        
        config = job.config
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.max_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            warmup_steps=config.warmup_steps,
            weight_decay=config.weight_decay,
            logging_dir=config.logs_dir,
            logging_steps=10,
            evaluation_strategy=config.evaluation_strategy,
            save_strategy=config.save_strategy,
            save_total_limit=config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            fp16=config.use_mixed_precision,
            dataloader_num_workers=config.dataloader_num_workers,
            remove_unused_columns=False,
            report_to="none"  # We handle logging manually
        )
        
        # Custom trainer with callbacks
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=config.early_stopping_patience,
                    early_stopping_threshold=config.early_stopping_threshold
                )
            ]
        )
        
        # Add custom callback
        callback = TrainingCallback(self, job)
        trainer.add_callback(callback)
        
        # Start training
        start_time = time.time()
        trainer.train()
        
        # Update metrics
        job.metrics.total_epochs = config.max_epochs
        job.metrics.training_time = time.time() - start_time
        
        logger.info(f"Model training completed for job {job.job_id}")
    
    async def _evaluate_model(
        self,
        job: TrainingJob,
        model: nn.Module,
        tokenizer: Any,
        test_dataset: Dataset
    ):
        """Evaluate the trained model."""
        logger.info(f"Evaluating model for job {job.job_id}")
        
        model.eval()
        device = next(model.parameters()).device
        
        # Create test dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=job.config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                if hasattr(outputs, 'logits'):
                    preds = torch.argmax(outputs.logits, dim=-1)
                    predictions.extend(preds.cpu().numpy())
                    
                    if 'labels' in batch:
                        true_labels.extend(batch['labels'].numpy())
        
        # Calculate metrics
        if true_labels:
            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='weighted')
            precision = precision_score(true_labels, predictions, average='weighted')
            recall = recall_score(true_labels, predictions, average='weighted')
            
            # Update job metrics
            job.metrics.val_accuracy.append(accuracy)
            job.metrics.val_f1.append(f1)
            job.metrics.val_precision.append(precision)
            job.metrics.val_recall.append(recall)
            
            logger.info(f"Test Results - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    def _save_model(self, job: TrainingJob, model: nn.Module, tokenizer: Any):
        """Save the trained model."""
        logger.info(f"Saving model for job {job.job_id}")
        
        output_path = Path(job.config.output_dir)
        
        # Save model and tokenizer
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # Save training configuration
        config_path = output_path / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            # Convert config to dict (excluding non-serializable fields)
            config_dict = {
                'model_name': job.config.model_name,
                'model_type': job.config.model_type.value,
                'strategy': job.config.strategy.value,
                'learning_rate': job.config.learning_rate,
                'batch_size': job.config.batch_size,
                'max_epochs': job.config.max_epochs,
                'max_sequence_length': job.config.max_sequence_length
            }
            json.dump(config_dict, f, indent=2)
        
        # Save metrics
        metrics_path = output_path / "training_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            metrics_dict = {
                'best_val_accuracy': job.metrics.best_val_accuracy,
                'best_val_f1': job.metrics.best_val_f1,
                'best_val_loss': job.metrics.best_val_loss,
                'best_epoch': job.metrics.best_epoch,
                'total_epochs': job.metrics.total_epochs,
                'training_time': job.metrics.training_time
            }
            json.dump(metrics_dict, f, indent=2)
        
        job.model_path = str(output_path)
        
        logger.info(f"Model saved to {output_path}")
    
    def _log_metrics(self, job_id: str, metrics: Dict[str, float], epoch: int):
        """Log metrics to tracking services."""
        try:
            # Log to MLflow
            if settings.MLFLOW_TRACKING_URI:
                with mlflow.start_run(run_name=f"job_{job_id}"):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(metric_name, value, step=epoch)
            
            # Log to Weights & Biases
            if settings.WANDB_PROJECT:
                wandb.log(metrics, step=epoch)
            
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")
    
    async def stop_training(self, job_id: str) -> bool:
        """Stop a running training job."""
        try:
            if job_id not in self.jobs:
                raise ValueError(f"Training job {job_id} not found")
            
            job = self.jobs[job_id]
            
            if job.status != TrainingStatus.RUNNING:
                raise ValueError(f"Training job {job_id} is not running")
            
            # Cancel the task
            if job_id in self.active_jobs:
                task = self.active_jobs[job_id]
                task.cancel()
                del self.active_jobs[job_id]
            
            job.status = TrainingStatus.CANCELLED
            job.completed_at = datetime.now()
            
            logger.info(f"Training job {job_id} cancelled")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping training job {job_id}: {e}")
            return False
    
    async def pause_training(self, job_id: str) -> bool:
        """Pause a running training job."""
        try:
            if job_id not in self.jobs:
                raise ValueError(f"Training job {job_id} not found")
            
            job = self.jobs[job_id]
            
            if job.status != TrainingStatus.RUNNING:
                raise ValueError(f"Training job {job_id} is not running")
            
            job.status = TrainingStatus.PAUSED
            
            logger.info(f"Training job {job_id} paused")
            return True
            
        except Exception as e:
            logger.error(f"Error pausing training job {job_id}: {e}")
            return False
    
    async def resume_training(self, job_id: str) -> bool:
        """Resume a paused training job."""
        try:
            if job_id not in self.jobs:
                raise ValueError(f"Training job {job_id} not found")
            
            job = self.jobs[job_id]
            
            if job.status != TrainingStatus.PAUSED:
                raise ValueError(f"Training job {job_id} is not paused")
            
            job.status = TrainingStatus.RUNNING
            
            logger.info(f"Training job {job_id} resumed")
            return True
            
        except Exception as e:
            logger.error(f"Error resuming training job {job_id}: {e}")
            return False
    
    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get training job status."""
        return self.jobs.get(job_id)
    
    def list_jobs(
        self,
        status: Optional[TrainingStatus] = None,
        limit: int = 50
    ) -> List[TrainingJob]:
        """List training jobs."""
        jobs = list(self.jobs.values())
        
        if status:
            jobs = [job for job in jobs if job.status == status]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        
        return jobs[:limit]
    
    async def cleanup_old_jobs(self, days: int = 30) -> int:
        """Clean up old completed training jobs."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            cleaned_count = 0
            
            jobs_to_remove = []
            
            for job_id, job in self.jobs.items():
                if (job.status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED] and
                    job.completed_at and job.completed_at < cutoff_date):
                    
                    # Remove job files
                    if job.config and job.config.output_dir:
                        output_path = Path(job.config.output_dir)
                        if output_path.exists():
                            shutil.rmtree(output_path)
                    
                    jobs_to_remove.append(job_id)
                    cleaned_count += 1
            
            # Remove jobs from memory
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
            
            logger.info(f"Cleaned up {cleaned_count} old training jobs")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old jobs: {e}")
            return 0
    
    async def get_training_statistics(self) -> Dict[str, Any]:
        """Get training service statistics."""
        try:
            total_jobs = len(self.jobs)
            running_jobs = len([j for j in self.jobs.values() if j.status == TrainingStatus.RUNNING])
            completed_jobs = len([j for j in self.jobs.values() if j.status == TrainingStatus.COMPLETED])
            failed_jobs = len([j for j in self.jobs.values() if j.status == TrainingStatus.FAILED])
            
            # Calculate average training time
            completed_job_list = [j for j in self.jobs.values() if j.status == TrainingStatus.COMPLETED]
            avg_training_time = 0.0
            if completed_job_list:
                total_time = sum(j.metrics.training_time for j in completed_job_list if j.metrics.training_time)
                avg_training_time = total_time / len(completed_job_list)
            
            # Get resource usage
            device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            memory_usage = 0.0
            if torch.cuda.is_available():
                memory_usage = torch.cuda.memory_allocated() / 1024**3  # GB
            
            return {
                "total_jobs": total_jobs,
                "running_jobs": running_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "average_training_time": avg_training_time,
                "device_count": device_count,
                "memory_usage_gb": memory_usage,
                "active_jobs": list(self.active_jobs.keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting training statistics: {e}")
            return {}
    
    async def export_model(
        self, 
        job_id: str, 
        export_format: str = "pytorch",
        quantize: bool = False
    ) -> str:
        """Export trained model in different formats."""
        try:
            if job_id not in self.jobs:
                raise ValueError(f"Training job {job_id} not found")
            
            job = self.jobs[job_id]
            
            if job.status != TrainingStatus.COMPLETED:
                raise ValueError(f"Training job {job_id} is not completed")
            
            if not job.model_path:
                raise ValueError(f"Model path not found for job {job_id}")
            
            model_path = Path(job.model_path)
            export_path = model_path.parent / f"exported_{export_format}"
            export_path.mkdir(exist_ok=True)
            
            # Load model
            model = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if export_format == "onnx":
                # Export to ONNX
                import torch.onnx
                
                dummy_input = torch.randint(0, 1000, (1, 512))
                
                torch.onnx.export(
                    model,
                    dummy_input,
                    export_path / "model.onnx",
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input_ids'],
                    output_names=['output'],
                    dynamic_axes={
                        'input_ids': {0: 'batch_size', 1: 'sequence'},
                        'output': {0: 'batch_size', 1: 'sequence'}
                    }
                )
                
            elif export_format == "tensorrt":
                # Export to TensorRT (placeholder)
                logger.warning("TensorRT export not implemented yet")
                
            elif export_format == "coreml":
                # Export to CoreML (placeholder)
                logger.warning("CoreML export not implemented yet")
                
            else:
                # Default PyTorch export
                model.save_pretrained(export_path)
                tokenizer.save_pretrained(export_path)
            
            # Quantization
            if quantize:
                quantized_path = export_path / "quantized"
                quantized_path.mkdir(exist_ok=True)
                
                # Apply dynamic quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                
                quantized_model.save_pretrained(quantized_path)
                tokenizer.save_pretrained(quantized_path)
            
            logger.info(f"Model exported to {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            raise ProcessingError(f"Failed to export model: {e}")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            # Cancel all active jobs
            for task in self.active_jobs.values():
                if not task.done():
                    task.cancel()
            
            # Shutdown executor
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
                
        except Exception as e:
            logger.error(f"Error during TrainingService cleanup: {e}")


# Global instance
training_service = TrainingService()


# Convenience functions
async def start_training(config: TrainingConfig) -> str:
    """Start a new training job."""
    return await training_service.start_training(config)


async def stop_training(job_id: str) -> bool:
    """Stop a training job."""
    return await training_service.stop_training(job_id)


def get_job_status(job_id: str) -> Optional[TrainingJob]:
    """Get training job status."""
    return training_service.get_job_status(job_id)


def list_training_jobs(status: Optional[TrainingStatus] = None) -> List[TrainingJob]:
    """List training jobs."""
    return training_service.list_jobs(status)


async def get_training_stats() -> Dict[str, Any]:
    """Get training statistics."""
    return await training_service.get_training_statistics()