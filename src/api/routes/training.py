"""
training.py - Pipeline Completa de Treinamento para Modelos de PLN Jurídico

Este módulo implementa uma pipeline robusta e escalável para treinamento de modelos
de Machine Learning/Processamento de Linguagem Natural especificamente otimizados
para dados jurídicos, integrado com a arquitetura do projeto juris_oracle.

Autor: AI Assistant
Data: 2025-01-11
Versão: 1.0.0
"""

import asyncio
import logging
import os
import pickle
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup, TrainingArguments, Trainer
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Imports específicos do projeto
from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi import HTTPException, BackgroundTasks
import uvloop

# Imports do projeto juris_oracle
from app.database.database import get_db, engine
from app.models.legal_document import LegalDocument
from app.models.case_analysis import CaseAnalysis
from app.core.config import settings
from app.core.logger import get_logger

# Configuração inicial do logging
logger = get_logger(__name__)

# Download de recursos NLTK necessários
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    logger.warning(f"Erro ao baixar recursos NLTK: {e}")

# Carregamento do modelo spaCy para português jurídico
try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    logger.warning("Modelo spaCy português não encontrado. Usando tokenização básica.")
    nlp = None


class ModelType(Enum):
    """Tipos de modelos disponíveis para treinamento."""
    DOCUMENT_CLASSIFIER = "document_classifier"
    ENTITY_EXTRACTION = "entity_extraction"
    CASE_ANALYZER = "case_analyzer"
    LEGAL_QA = "legal_qa"
    TEXT_SUMMARIZER = "text_summarizer"


class TrainingStatus(Enum):
    """Status do processo de treinamento."""
    INITIALIZED = "initialized"
    LOADING_DATA = "loading_data"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    SAVING_MODEL = "saving_model"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingConfig:
    """Configuração para o processo de treinamento."""
    model_type: ModelType
    model_name: str = "neuralmind/bert-base-portuguese-cased"
    num_labels: int = 5
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42
    output_dir: str = "models/trained"
    use_gpu: bool = True
    save_best_model: bool = True
    early_stopping_patience: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte a configuração para dicionário."""
        return asdict(self)


@dataclass
class TrainingMetrics:
    """Métricas de avaliação do modelo."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    matthews_corr: float
    roc_auc: Optional[float] = None
    loss: float = 0.0
    val_loss: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Converte as métricas para dicionário."""
        return asdict(self)


class LegalTextDataset(Dataset):
    """Dataset customizado para textos jurídicos."""
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenizer, 
        max_length: int = 512
    ):
        """
        Inicializa o dataset.
        
        Args:
            texts: Lista de textos jurídicos
            labels: Lista de labels correspondentes
            tokenizer: Tokenizador do modelo
            max_length: Comprimento máximo dos tokens
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retorna um item do dataset.
        
        Args:
            idx: Índice do item
            
        Returns:
            Dicionário com tokens e labels
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenização
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class LegalTextPreprocessor:
    """Preprocessador especializado para textos jurídicos."""
    
    def __init__(self, language: str = 'portuguese'):
        """
        Inicializa o preprocessador.
        
        Args:
            language: Idioma para processamento
        """
        self.language = language
        self.stop_words = set(stopwords.words('portuguese'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Termos jurídicos específicos que não devem ser removidos
        self.legal_terms = {
            'artigo', 'lei', 'decreto', 'jurisprudência', 'acórdão',
            'sentença', 'decisão', 'tribunal', 'juiz', 'desembargador',
            'ministro', 'réu', 'autor', 'apelante', 'apelado',
            'recurso', 'agravo', 'embargos', 'habeas', 'corpus',
            'mandado', 'segurança', 'injunção', 'ação', 'processo'
        }
        
        # Remove termos jurídicos das stop words
        self.stop_words = self.stop_words - self.legal_terms
    
    def clean_legal_text(self, text: str) -> str:
        """
        Limpa texto jurídico removendo elementos desnecessários.
        
        Args:
            text: Texto a ser limpo
            
        Returns:
            Texto limpo
        """
        if not text:
            return ""
        
        # Remove numeração de páginas e referências
        import re
        text = re.sub(r'\b\d+\b(?:\s*[-–]\s*\d+)?', '', text)  # Remove números isolados
        text = re.sub(r'(?:pág|página|p\.)\s*\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'art\.?\s*\d+', lambda m: m.group(0), text, flags=re.IGNORECASE)  # Preserva artigos
        
        # Remove URLs e emails
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excesso de espaços em branco
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extrai entidades jurídicas do texto.
        
        Args:
            text: Texto para extração
            
        Returns:
            Dicionário com entidades extraídas
        """
        entities = {
            'laws': [],
            'articles': [],
            'courts': [],
            'judges': [],
            'case_numbers': []
        }
        
        if not nlp:
            return entities
        
        doc = nlp(text)
        
        # Extração usando regex para entidades jurídicas específicas
        import re
        
        # Leis e decretos
        law_pattern = r'(?:lei|decreto)(?:\s+n[ºo°]?)?\s*(\d+(?:[./]\d+)*)'
        entities['laws'] = re.findall(law_pattern, text, re.IGNORECASE)
        
        # Artigos
        article_pattern = r'art(?:igo)?\.?\s*(\d+(?:[ºo°])?)'
        entities['articles'] = re.findall(article_pattern, text, re.IGNORECASE)
        
        # Tribunais
        court_pattern = r'((?:supremo\s+)?tribunal\s+(?:de\s+)?(?:justiça|federal|regional|superior)[^.]*)'
        entities['courts'] = re.findall(court_pattern, text, re.IGNORECASE)
        
        # Números de processo
        case_pattern = r'(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4})'
        entities['case_numbers'] = re.findall(case_pattern, text)
        
        return entities
    
    def preprocess_text(self, text: str, for_embedding: bool = True) -> str:
        """
        Preprocessa texto jurídico para treinamento.
        
        Args:
            text: Texto a ser preprocessado
            for_embedding: Se True, mantém mais informações para embeddings
            
        Returns:
            Texto preprocessado
        """
        if not text:
            return ""
        
        # Limpeza inicial
        text = self.clean_legal_text(text)
        
        if for_embedding:
            # Para embeddings, mantemos mais informações
            return text.lower()
        
        # Tokenização
        tokens = word_tokenize(text.lower(), language='portuguese')
        
        # Remove stop words (exceto termos jurídicos)
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Remove tokens muito curtos ou números puros
        tokens = [token for token in tokens if len(token) > 2 and not token.isdigit()]
        
        # Lematização
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)


class LegalModelTrainer:
    """Classe principal para treinamento de modelos jurídicos."""
    
    def __init__(self, config: TrainingConfig):
        """
        Inicializa o trainer.
        
        Args:
            config: Configuração de treinamento
        """
        self.config = config
        self.status = TrainingStatus.INITIALIZED
        self.preprocessor = LegalTextPreprocessor()
        self.label_encoder = LabelEncoder()
        self.tokenizer = None
        self.model = None
        self.metrics = None
        
        # Configuração de device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu'
        )
        logger.info(f"Usando device: {self.device}")
        
        # Criação do diretório de saída
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Configuração de random seed
        self._set_random_seed(config.random_seed)
    
    def _set_random_seed(self, seed: int) -> None:
        """Define seed para reprodutibilidade."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    async def load_legal_data(self) -> Tuple[List[str], List[str]]:
        """
        Carrega dados jurídicos do banco de dados.
        
        Returns:
            Tupla com textos e labels
        """
        self.status = TrainingStatus.LOADING_DATA
        logger.info("Iniciando carregamento de dados jurídicos...")
        
        try:
            texts = []
            labels = []
            
            # Conexão com banco de dados
            db = next(get_db())
            
            if self.config.model_type == ModelType.DOCUMENT_CLASSIFIER:
                # Carrega documentos legais para classificação
                query = """
                SELECT content, document_type, legal_area 
                FROM legal_documents 
                WHERE content IS NOT NULL 
                AND char_length(content) > 100
                ORDER BY created_at DESC
                LIMIT 10000
                """
                
                result = db.execute(text(query))
                for row in result:
                    if row.content:
                        texts.append(row.content)
                        # Combina tipo e área para criar label
                        label = f"{row.document_type}_{row.legal_area}"
                        labels.append(label)
            
            elif self.config.model_type == ModelType.CASE_ANALYZER:
                # Carrega análises de casos
                query = """
                SELECT ca.summary, ca.legal_foundation, ca.decision_type,
                       ld.content, ld.legal_area
                FROM case_analyses ca
                JOIN legal_documents ld ON ca.document_id = ld.id
                WHERE ca.summary IS NOT NULL
                AND char_length(ca.summary) > 50
                ORDER BY ca.created_at DESC
                LIMIT 5000
                """
                
                result = db.execute(text(query))
                for row in result:
                    if row.summary:
                        # Combina summary e conteúdo
                        text = f"{row.summary} {row.legal_foundation or ''}"
                        texts.append(text)
                        labels.append(row.decision_type or 'unknown')
            
            else:
                # Dados sintéticos para outros tipos de modelo
                logger.warning(f"Usando dados sintéticos para {self.config.model_type}")
                texts, labels = self._generate_synthetic_data()
            
            db.close()
            
            logger.info(f"Carregados {len(texts)} documentos com {len(set(labels))} classes únicas")
            return texts, labels
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            # Fallback para dados sintéticos
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> Tuple[List[str], List[str]]:
        """
        Gera dados sintéticos para teste.
        
        Returns:
            Tupla com textos e labels sintéticos
        """
        logger.info("Gerando dados sintéticos para treinamento...")
        
        synthetic_texts = [
            "O tribunal decidiu pela procedência da ação de indenização por danos morais.",
            "A empresa foi condenada ao pagamento de multa por descumprimento contratual.",
            "O recurso foi provido para reformar a sentença de primeiro grau.",
            "A liminar foi deferida para suspender os efeitos da decisão administrativa.",
            "O processo foi extinto sem resolução do mérito por ilegitimidade ativa.",
        ] * 200  # Replica para ter mais dados
        
        synthetic_labels = [
            "civil_indenizacao",
            "empresarial_contrato",
            "processual_recurso",
            "administrativo_liminar",
            "processual_extincao"
        ] * 200
        
        return synthetic_texts, synthetic_labels
    
    def preprocess_data(
        self, 
        texts: List[str], 
        labels: List[str]
    ) -> Tuple[List[str], np.ndarray]:
        """
        Preprocessa os dados para treinamento.
        
        Args:
            texts: Lista de textos
            labels: Lista de labels
            
        Returns:
            Tupla com textos processados e labels codificados
        """
        self.status = TrainingStatus.PREPROCESSING
        logger.info("Iniciando pré-processamento dos dados...")
        
        # Preprocessamento dos textos
        processed_texts = []
        for i, text in enumerate(texts):
            try:
                processed_text = self.preprocessor.preprocess_text(text, for_embedding=True)
                if processed_text:  # Só adiciona se não estiver vazio
                    processed_texts.append(processed_text)
                else:
                    processed_texts.append(text)  # Fallback para texto original
            except Exception as e:
                logger.warning(f"Erro no preprocessamento do texto {i}: {e}")
                processed_texts.append(text)  # Fallback
        
        # Codificação dos labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Atualiza configuração com número de classes
        self.config.num_labels = len(self.label_encoder.classes_)
        
        logger.info(f"Preprocessamento concluído. {len(processed_texts)} textos, {self.config.num_labels} classes")
        
        return processed_texts, encoded_labels
    
    def create_data_loaders(
        self, 
        texts: List[str], 
        labels: np.ndarray
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Cria data loaders para treinamento, validação e teste.
        
        Args:
            texts: Textos preprocessados
            labels: Labels codificados
            
        Returns:
            Tupla com train_loader, val_loader, test_loader
        """
        # Divisão dos dados
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, 
            test_size=self.config.test_split, 
            random_state=self.config.random_seed,
            stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config.validation_split / (1 - self.config.test_split),
            random_state=self.config.random_seed,
            stratify=y_temp
        )
        
        # Criação dos datasets
        train_dataset = LegalTextDataset(X_train, y_train, self.tokenizer, self.config.max_length)
        val_dataset = LegalTextDataset(X_val, y_val, self.tokenizer, self.config.max_length)
        test_dataset = LegalTextDataset(X_test, y_test, self.tokenizer, self.config.max_length)
        
        # Criação dos data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        logger.info(f"Data loaders criados - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def initialize_model(self) -> None:
        """Inicializa o modelo e tokenizer."""
        logger.info(f"Inicializando modelo {self.config.model_name}...")
        
        try:
            # Inicializa tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Inicializa modelo
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels
            )
            
            self.model.to(self.device)
            
            logger.info("Modelo inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar modelo: {e}")
            raise
    
    def train_model(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader
    ) -> None:
        """
        Treina o modelo.
        
        Args:
            train_loader: DataLoader de treinamento
            val_loader: DataLoader de validação
        """
        self.status = TrainingStatus.TRAINING
        logger.info("Iniciando treinamento do modelo...")
        
        # Configuração do otimizador
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Variáveis para early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Loop de treinamento
        for epoch in range(self.config.num_epochs):
            logger.info(f"Época {epoch + 1}/{self.config.num_epochs}")
            
            # Treinamento
            self.model.train()
            total_train_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch para device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update weights
                optimizer.step()
                scheduler.step()
                
                if batch_idx % 100 == 0:
                    logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validação
            val_loss = self._validate_model(val_loader)
            
            logger.info(f"Época {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if self.config.save_best_model:
                    self._save_checkpoint(epoch, val_loss)
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping na época {epoch + 1}")
                break
        
        logger.info("Treinamento concluído")
    
    def _validate_model(self, val_loader: DataLoader) -> float:
        """
        Executa validação do modelo.
        
        Args:
            val_loader: DataLoader de validação
            
        Returns:
            Loss médio de validação
        """
        self.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_val_loss += outputs.loss.item()
        
        return total_val_loss / len(val_loader)
    
    def evaluate_model(self, test_loader: DataLoader) -> TrainingMetrics:
        """
        Avalia o modelo no conjunto de teste.
        
        Args:
            test_loader: DataLoader de teste
            
        Returns:
            Métricas de avaliação
        """
        self.status = TrainingStatus.EVALUATING
        logger.info("Iniciando avaliação do modelo...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                # Predições
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Cálculo das métricas
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        matthews_corr = matthews_corrcoef(all_labels, all_predictions)
        
        # ROC-AUC (apenas para classificação binária ou com probabilidades)
        roc_auc = None
        if self.config.num_labels == 2:
            try:
                # Para classificação binária, calcular probabilidades
                self.model.eval()
                all_probs = []
                with torch.no_grad():
                    for batch in test_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        probs = torch.softmax(outputs.logits, dim=-1)
                        all_probs.extend(probs[:, 1].cpu().numpy())  # Probabilidade da classe positiva
                
                roc_auc = roc_auc_score(all_labels, all_probs)
            except Exception as e:
                logger.warning(f"Erro ao calcular ROC-AUC: {e}")
        
        self.metrics = TrainingMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            matthews_corr=matthews_corr,
            roc_auc=roc_auc,
            loss=total_loss / len(test_loader)
        )
        
        # Log das métricas
        logger.info("=== MÉTRICAS DE AVALIAÇÃO ===")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"Matthews Correlation: {matthews_corr:.4f}")
        if roc_auc:
            logger.info(f"ROC-AUC: {roc_auc:.4f}")
        logger.info(f"Loss: {total_loss / len(test_loader):.4f}")
        
        # Classification report detalhado
        class_names = self.label_encoder.classes_
        report = classification_report(
            all_labels, all_predictions, 
            target_names=class_names, 
            output_dict=True
        )
        logger.info(f"Classification Report:\n{classification_report(all_labels, all_predictions, target_names=class_names)}")
        
        return self.metrics
    
    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """
        Salva checkpoint do modelo.
        
        Args:
            epoch: Época atual
            val_loss: Loss de validação
        """
        checkpoint_dir = self.output_path / f"checkpoint_epoch_{epoch}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Salva modelo e tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Salva metadados
        metadata = {
            'epoch': epoch,
            'val_loss': val_loss,
            'config': self.config.to_dict(),
            'label_encoder_classes': self.label_encoder.classes_.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Checkpoint salvo em {checkpoint_dir}")
    
    def save_model(self) -> str:
        """
        Salva o modelo final treinado.
        
        Returns:
            Caminho onde o modelo foi salvo
        """
        self.status = TrainingStatus.SAVING_MODEL
        logger.info("Salvando modelo final...")
        
        # Diretório de saída com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.output_path / f"{self.config.model_type.value}_{timestamp}"
        model_dir.mkdir(exist_ok=True)
        
        try:
            # Salva modelo e tokenizer
            self.model.save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)
            
            # Salva label encoder
            with open(model_dir / 'label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            # Salva preprocessador
            with open(model_dir / 'preprocessor.pkl', 'wb') as f:
                pickle.dump(self.preprocessor, f)
            
            # Salva configuração e métricas
            final_metadata = {
                'model_type': self.config.model_type.value,
                'config': self.config.to_dict(),
                'metrics': self.metrics.to_dict() if self.metrics else None,
                'label_encoder_classes': self.label_encoder.classes_.tolist(),
                'training_completed_at': datetime.now().isoformat(),
                'device_used': str(self.device)
            }
            
            with open(model_dir / 'model_info.json', 'w', encoding='utf-8') as f:
                json.dump(final_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Modelo salvo com sucesso em: {model_dir}")
            return str(model_dir)
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
            raise
    
    async def run_training_pipeline(self) -> Dict[str, Any]:
        """
        Executa a pipeline completa de treinamento.
        
        Returns:
            Dicionário com resultados do treinamento
        """
        start_time = datetime.now()
        logger.info("=== INICIANDO PIPELINE DE TREINAMENTO ===")
        
        try:
            # 1. Carregamento de dados
            texts, labels = await self.load_legal_data()
            
            if len(texts) == 0:
                raise ValueError("Nenhum dado encontrado para treinamento")
            
            # 2. Pré-processamento
            processed_texts, encoded_labels = self.preprocess_data(texts, labels)
            
            # 3. Inicialização do modelo
            self.initialize_model()
            
            # 4. Criação dos data loaders
            train_loader, val_loader, test_loader = self.create_data_loaders(
                processed_texts, encoded_labels
            )
            
            # 5. Treinamento
            self.train_model(train_loader, val_loader)
            
            # 6. Avaliação
            metrics = self.evaluate_model(test_loader)
            
            # 7. Salvamento do modelo
            model_path = self.save_model()
            
            # 8. Finalização
            self.status = TrainingStatus.COMPLETED
            end_time = datetime.now()
            training_duration = end_time - start_time
            
            results = {
                'status': self.status.value,
                'model_path': model_path,
                'metrics': metrics.to_dict(),
                'training_duration_seconds': training_duration.total_seconds(),
                'num_samples': len(texts),
                'num_classes': self.config.num_labels,
                'config': self.config.to_dict(),
                'completed_at': end_time.isoformat()
            }
            
            logger.info("=== TREINAMENTO CONCLUÍDO COM SUCESSO ===")
            logger.info(f"Duração: {training_duration}")
            logger.info(f"F1-Score: {metrics.f1_score:.4f}")
            logger.info(f"Modelo salvo em: {model_path}")
            
            return results
            
        except Exception as e:
            self.status = TrainingStatus.FAILED
            logger.error(f"Erro na pipeline de treinamento: {e}")
            logger.error(traceback.format_exc())
            
            return {
                'status': self.status.value,
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            }


# === FUNÇÕES DE API PARA INTEGRAÇÃO COM FASTAPI ===

async def train_legal_model_async(
    model_type: ModelType,
    config_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Função assíncrona para treinamento de modelo via API.
    
    Args:
        model_type: Tipo de modelo a ser treinado
        config_override: Configurações customizadas
        
    Returns:
        Resultados do treinamento
    """
    # Configuração padrão
    config = TrainingConfig(model_type=model_type)
    
    # Aplica overrides se fornecidos
    if config_override:
        for key, value in config_override.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Cria trainer e executa pipeline
    trainer = LegalModelTrainer(config)
    results = await trainer.run_training_pipeline()
    
    return results


def train_legal_model_background(
    background_tasks: BackgroundTasks,
    model_type: ModelType,
    config_override: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Inicia treinamento em background via API.
    
    Args:
        background_tasks: Background tasks do FastAPI
        model_type: Tipo de modelo
        config_override: Configurações customizadas
        
    Returns:
        Status de início do treinamento
    """
    # Cria task em background
    background_tasks.add_task(
        train_legal_model_async,
        model_type,
        config_override
    )
    
    return {
        'status': 'training_started',
        'message': f'Treinamento do modelo {model_type.value} iniciado em background',
        'timestamp': datetime.now().isoformat()
    }


def get_available_models() -> Dict[str, List[str]]:
    """
    Retorna lista de modelos disponíveis e treinados.
    
    Returns:
        Dicionário com modelos disponíveis
    """
    models_dir = Path("models/trained")
    
    if not models_dir.exists():
        return {
            'trained_models': [],
            'available_model_types': [mt.value for mt in ModelType]
        }
    
    trained_models = []
    for model_path in models_dir.iterdir():
        if model_path.is_dir() and (model_path / 'model_info.json').exists():
            try:
                with open(model_path / 'model_info.json', 'r', encoding='utf-8') as f:
                    info = json.load(f)
                trained_models.append({
                    'path': str(model_path),
                    'type': info.get('model_type'),
                    'created_at': info.get('training_completed_at'),
                    'metrics': info.get('metrics')
                })
            except Exception as e:
                logger.warning(f"Erro ao ler info do modelo {model_path}: {e}")
    
    return {
        'trained_models': trained_models,
        'available_model_types': [mt.value for mt in ModelType]
    }


def load_trained_model(model_path: str) -> Dict[str, Any]:
    """
    Carrega um modelo treinado.
    
    Args:
        model_path: Caminho para o modelo
        
    Returns:
        Dicionário com modelo e metadados
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Modelo não encontrado")
    
    try:
        # Carrega metadados
        with open(model_path / 'model_info.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Carrega tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Carrega modelo
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Carrega label encoder
        with open(model_path / 'label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Carrega preprocessador
        with open(model_path / 'preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'label_encoder': label_encoder,
            'preprocessor': preprocessor,
            'metadata': metadata
        }
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao carregar modelo: {str(e)}")


# === EXEMPLO DE USO ===

if __name__ == "__main__":
    # Configuração de exemplo
    config = TrainingConfig(
        model_type=ModelType.DOCUMENT_CLASSIFIER,
        num_epochs=2,
        batch_size=8,
        learning_rate=2e-5
    )
    
    # Execução do treinamento
    async def main():
        trainer = LegalModelTrainer(config)
        results = await trainer.run_training_pipeline()
        print(json.dumps(results, indent=2, ensure_ascii=False))
    
    # Executa com uvloop para melhor performance
    uvloop.install()
    asyncio.run(main())