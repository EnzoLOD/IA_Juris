"""
JurisOracle - Data Preparation Module
===================================

Módulo especializado para preparação de dados jurídicos para treinamento.
Inclui processamento, limpeza, normalização e transformação de documentos jurídicos.

Author: JurisOracle Development Team
Version: 1.0.0
"""

import os
import re
import json
import logging
import hashlib
import mimetypes
import time
from typing import Dict, List, Optional, Tuple, Any, Generator, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache
import threading
from collections import defaultdict, Counter

# Data Processing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Document Processing
import PyPDF2
import pdfplumber
from docx import Document
import textract
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

# Text Processing
import spacy
from transformers import AutoTokenizer
import unicodedata
import ftfy

# Utilities
import tqdm
from multiprocessing import cpu_count
import psutil
import magic
from dateutil import parser as date_parser

# JurisOracle imports
from ..config import get_settings
from ..core.logging_setup import get_logger
from ..models import DocumentModel, DatasetModel

logger = get_logger(__name__)

# --- INÍCIO DO MÓDULO AVANÇADO DE PREPARAÇÃO DE DADOS JURÍDICOS ---

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

@dataclass
class ProcessingStats:
    """Estatísticas de processamento de dados."""
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    total_size_bytes: int = 0
    processing_time_seconds: float = 0.0
    avg_doc_size: float = 0.0
    document_types: Dict[str, int] = field(default_factory=dict)
    error_types: Dict[str, int] = field(default_factory=dict)
    text_stats: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetSplit:
    """Divisão de dataset para treinamento."""
    train: List[Dict[str, Any]] = field(default_factory=list)
    validation: List[Dict[str, Any]] = field(default_factory=list)
    test: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_stats(self) -> Dict[str, int]:
        return {
            'train_size': len(self.train),
            'validation_size': len(self.validation),
            'test_size': len(self.test),
            'total_size': len(self.train) + len(self.validation) + len(self.test)
        }

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by handling missing values and duplicates."""
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def prepare_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Prepare features for training by selecting relevant columns."""
    return df[feature_columns]

def prepare_labels(df: pd.DataFrame, label_column: str) -> pd.Series:
    """Prepare labels for training."""
    return df[label_column]

def split_data(df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, pd.DataFrame]:
    """Split the data into training and testing sets."""
    train_df = df.sample(frac=1 - test_size, random_state=42)
    test_df = df.drop(train_df.index)
    return {'train': train_df, 'test': test_df}

def save_prepared_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str) -> None:
    """Save the prepared training and testing data to CSV files."""
    train_df.to_csv(f"{output_dir}/train_data.csv", index=False)
    test_df.to_csv(f"{output_dir}/test_data.csv", index=False)

class DocumentProcessor:
    """Processador especializado para documentos jurídicos."""
    # Padrões regex para identificação de elementos jurídicos
    LEGAL_PATTERNS = {
        'lei': re.compile(r'Lei\s+n[°º]?\s*(\d+(?:\.\d+)*(?:/\d{4})?)', re.IGNORECASE),
        'artigo': re.compile(r'Art(?:igo)?\.?\s*(\d+(?:[°º]|[-\.]\w+)*)', re.IGNORECASE),
        'inciso': re.compile(r'Inciso\s+([IVX]+|[a-z]|\d+)', re.IGNORECASE),
        'paragrafo': re.compile(r'[§¶]\s*(\d+[°º]?)', re.IGNORECASE),
        'processo': re.compile(r'(\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}|\d{4}\.\d{2}\.\d{2}\.\d{6}-\d)', re.IGNORECASE),
        'cpf': re.compile(r'\d{3}\.\d{3}\.\d{3}-\d{2}'),
        'cnpj': re.compile(r'\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}'),
        'cep': re.compile(r'\d{5}-?\d{3}'),
        'data_br': re.compile(r'\d{1,2}/\d{1,2}/\d{4}|\d{1,2}\s+de\s+\w+\s+de\s+\d{4}', re.IGNORECASE),
        'valor_monetario': re.compile(r'R\$\s*[\d.,]+'),
        'tribunal': re.compile(r'(STF|STJ|TST|TSE|STM|TRF|TRT|TRE|TJSP|TJRJ|TJ\w+)', re.IGNORECASE),
    }
    LEGAL_STOPWORDS = {
        'português': {
            'autos', 'processo', 'recurso', 'apelação', 'embargos', 'agravo',
            'mandado', 'habeas', 'corpus', 'requerente', 'requerido', 'autor',
            'réu', 'apelante', 'apelado', 'recorrente', 'recorrido', 'exequente',
            'executado', 'impetrante', 'impetrado', 'agravante', 'agravado',
            'embargante', 'embargado', 'interessado', 'terceiro', 'litisconsorte'
        }
    }
    def __init__(self):
        self.stemmer = SnowballStemmer('portuguese')
        self.stopwords = set(stopwords.words('portuguese')).union(self.LEGAL_STOPWORDS['português'])
        try:
            self.nlp = spacy.load('pt_core_news_sm')
        except OSError:
            logger.warning("Modelo spaCy pt_core_news_sm não encontrado. Algumas funcionalidades serão limitadas.")
            self.nlp = None

    def extract_text_from_file(self, file_path: str) -> str:
        """Extrai o texto de um arquivo jurídico."""
        file_extension = Path(file_path).suffix.lower()
        if file_extension in ['.pdf', '.docx']:
            # Usar processamento em paralelo para PDFs e DOCXs
            with ThreadPoolExecutor() as executor:
                future = executor.submit(self._extract_from_pdf if file_extension == '.pdf' else self._extract_from_docx, file_path)
                text = future.result()
        else:
            logger.warning(f"Formato de arquivo não suportado para extração de texto: {file_extension}")
            text = ""
        return text

    def _extract_from_pdf(self, file_path: str) -> str:
        """Extrai texto de arquivos PDF."""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Erro ao extrair texto de {file_path}: {e}")
        return text

    def _extract_from_docx(self, file_path: str) -> str:
        """Extrai texto de arquivos DOCX."""
        text = ""
        try:
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            logger.error(f"Erro ao extrair texto de {file_path}: {e}")
        return text

    def clean_text(self, text: str) -> str:
        """Limpa o texto extraído, removendo cabeçalhos, rodapés e caracteres indesejados."""
        # Remover cabeçalhos e rodapés comuns
        text = self._remove_headers_footers(text)
        # Remover múltiplos espaços em branco
        text = re.sub(r'\s+', ' ', text)
        # Remover espaços em branco no início e no fim
        text = text.strip()
        return text

    def _remove_headers_footers(self, text: str) -> str:
        """Remove cabeçalhos e rodapés com base em padrões comuns."""
        lines = text.split('\n')
        # Supõe-se que o cabeçalho/rodapé ocupe até 10% do texto
        threshold = len(lines) // 10
        # Remove as primeiras e últimas 'threshold' linhas
        return '\n'.join(lines[threshold:-threshold])

    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """Extrai entidades legais do texto usando padrões regex."""
        entities = {key: [] for key in self.LEGAL_PATTERNS.keys()}
        for key, pattern in self.LEGAL_PATTERNS.items():
            matches = pattern.findall(text)
            entities[key] = matches
        return entities

    def anonymize_text(self, text: str, entities: Optional[Dict[str, List[str]]] = None) -> str:
        """Anonimiza dados sensíveis no texto, como CPF e CNPJ."""
        if entities is None:
            entities = self.extract_legal_entities(text)
        text = re.sub(self.LEGAL_PATTERNS['cpf'], 'XXX.XXX.XXX-XX', text)
        text = re.sub(self.LEGAL_PATTERNS['cnpj'], 'XX.XXX.XXX/XXXX-XX', text)
        # Anonimiza pessoas identificadas pelo spaCy
        if hasattr(self, 'nlp') and self.nlp and 'spacy_person' in entities:
            for person in entities['spacy_person']:
                if len(person) > 3:
                    text = text.replace(person, '[PESSOA_ANONIMIZADA]')
        return text

    def segment_document(self, text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Segmenta o documento em partes menores para processamento."""
        # Usa a pontuação como delimitador para segmentação
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ''
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                if overlap > 0:
                    current_chunk = ' '.join(current_chunk.split()[-overlap:]) + ' '
                else:
                    current_chunk = ''
            current_chunk += sentence + ' '
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return [seg for seg in chunks if seg.strip()]

class DatasetBuilder:
    """Construtor de datasets especializados para treinamento."""
    def __init__(self, tokenizer_name: str = "neuralmind/bert-base-portuguese-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.processor = DocumentProcessor()
        self.label_encoders = {}

    def create_qa_dataset(self, documents: List[Dict[str, Any]], output_dir: str) -> None:
        """Cria um dataset para tarefas de Pergunta e Resposta (Q&A)."""
        # Exemplo de implementação: salva perguntas/respostas extraídas dos documentos
        qa_samples = []
        for doc in documents:
            chunks = self.processor.segment_document(doc['text'])
            for chunk in chunks:
                qa_samples.append({
                    'context': chunk,
                    'question': f'Qual o assunto principal deste trecho?',
                    'answer': chunk[:100]  # Exemplo: resposta fictícia
                })
        with open(os.path.join(output_dir, 'qa_dataset.json'), 'w', encoding='utf-8') as f:
            json.dump(qa_samples, f, ensure_ascii=False, indent=2)

    def create_summarization_dataset(self, documents: List[Dict[str, Any]], output_dir: str) -> None:
        """Cria um dataset para tarefas de sumarização."""
        summarization_samples = []
        for doc in documents:
            chunks = self.processor.segment_document(doc['text'])
            for chunk in chunks:
                summarization_samples.append({
                    'text': chunk,
                    'summary': chunk[:150]  # Exemplo: resumo fictício
                })
        with open(os.path.join(output_dir, 'summarization_dataset.json'), 'w', encoding='utf-8') as f:
            json.dump(summarization_samples, f, ensure_ascii=False, indent=2)

    def create_classification_dataset(self, documents: List[Dict[str, Any]], output_dir: str) -> None:
        """Cria um dataset para tarefas de classificação."""
        classification_samples = []
        for doc in documents:
            label = doc.get('legal_area', 'unknown')
            chunks = self.processor.segment_document(doc['text'])
            for chunk in chunks:
                classification_samples.append({
                    'text': chunk,
                    'label': label
                })
        with open(os.path.join(output_dir, 'classification_dataset.json'), 'w', encoding='utf-8') as f:
            json.dump(classification_samples, f, ensure_ascii=False, indent=2)

    # Métodos auxiliares podem ser implementados conforme necessário

class DataPreparationPipeline:
    """Pipeline completo de preparação de dados."""
    def __init__(self, input_dir: Union[str, Path], output_dir: Union[str, Path], cache_dir: Optional[Union[str, Path]] = None, max_workers: Optional[int] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.output_dir / "cache"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers or min(cpu_count(), 8)
        self.processor = DocumentProcessor()
        self.dataset_builder = DatasetBuilder()
        self.stats = ProcessingStats()
        self._stats_lock = threading.Lock()
        logger.info(f"Pipeline inicializado - Input: {self.input_dir}, Output: {self.output_dir}")
        logger.info(f"Workers: {self.max_workers}, Cache: {self.cache_dir}")

    def run_full_pipeline(self) -> None:
        """Executa o pipeline completo de preparação de dados."""
        logger.info("Iniciando o pipeline completo de preparação de dados.")
        # Coleta e processamento dos arquivos
        files = self._collect_files()
        documents = self._process_documents(files)
        # Classificação dos documentos
        self._classify_documents(documents)
        # Divisão e salvamento dos datasets
        self._split_dataset(documents)
        logger.info("Pipeline completo de preparação de dados finalizado.")

    def _collect_files(self) -> List[str]:
        """Coleta os arquivos de entrada do diretório especificado."""
        logger.info(f"Coletando arquivos do diretório: {self.input_dir}")
        files = list(self.input_dir.rglob("*.*"))
        logger.info(f"Total de arquivos coletados: {len(files)}")
        return files

    def _process_documents(self, files: List[str]) -> List[Dict[str, Any]]:
        """Processa os documentos coletados, extraindo texto e metadados."""
        logger.info("Iniciando o processamento dos documentos.")
        documents = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self.processor.extract_text_from_file, str(file)): file for file in files}
            for future in tqdm.tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processando documentos"):
                file = future_to_file[future]
                try:
                    text = future.result()
                    cleaned_text = self.processor.clean_text(text)
                    entities = self.processor.extract_legal_entities(cleaned_text)
                    anonymized_text = self.processor.anonymize_text(cleaned_text)
                    documents.append({
                        'file_name': file.name,
                        'file_path': str(file),
                        'text': cleaned_text,
                        'anonymized_text': anonymized_text,
                        'entities': entities,
                        'legal_area': None,
                        'document_type': None,
                    })
                except Exception as e:
                    logger.error(f"Erro ao processar o arquivo {file}: {e}")
                    with self._stats_lock:
                        self.stats.failed_documents += 1
        with self._stats_lock:
            self.stats.total_documents = len(documents)
            self.stats.processed_documents = len(documents) - self.stats.failed_documents
        logger.info(f"Processamento de documentos concluído. Total: {self.stats.total_documents}, Erros: {self.stats.failed_documents}")
        return documents

    def _classify_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Classifica os documentos em áreas do direito e tipos de documento."""
        logger.info("Iniciando a classificação dos documentos.")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doc = {executor.submit(self._classify_document_type, doc): doc for doc in documents}
            for future in tqdm.tqdm(as_completed(future_to_doc), total=len(future_to_doc), desc="Classificando documentos"):
                doc = future_to_doc[future]
                try:
                    result = future.result()
                    with self._stats_lock:
                        self.stats.document_types[result] = self.stats.document_types.get(result, 0) + 1
                except Exception as e:
                    logger.error(f"Erro ao classificar o documento {doc['file_name']}: {e}")
                    with self._stats_lock:
                        self.stats.failed_documents += 1
        logger.info("Classificação dos documentos concluída.")

    def _classify_document_type(self, document: Dict[str, Any]) -> str:
        """Classifica o tipo de documento com base no conteúdo e entidades."""
        # Implementar lógica de classificação de documento
        return "tipo_documento_exemplo"

    def _split_dataset(self, documents: List[Dict[str, Any]]) -> None:
        """Divide o dataset em conjuntos de treinamento, validação e teste."""
        logger.info("Dividindo o dataset em conjuntos de treinamento, validação e teste.")
        try:
            train_val_docs, test_docs = train_test_split(documents, test_size=0.2, random_state=42)
            train_docs, val_docs = train_test_split(train_val_docs, test_size=0.1, random_state=42)
            self._save_datasets(train_docs, val_docs, test_docs)
        except Exception as e:
            logger.error(f"Erro ao dividir o dataset: {e}")

    def _save_datasets(self, train_docs: List[Dict[str, Any]], val_docs: List[Dict[str, Any]], test_docs: List[Dict[str, Any]]) -> None:
        """Salva os conjuntos de dados em arquivos CSV."""
        logger.info(f"Salvando conjuntos de dados - Train: {len(train_docs)}, Validation: {len(val_docs)}, Test: {len(test_docs)}")
        try:
            train_df = pd.DataFrame(train_docs)
            val_df = pd.DataFrame(val_docs)
            test_df = pd.DataFrame(test_docs)
            save_prepared_data(train_df, test_df, self.output_dir)
        except Exception as e:
            logger.error(f"Erro ao salvar os conjuntos de dados: {e}")

    def _generate_report(self) -> None:
        """Gera um relatório sobre o processamento e estatísticas dos dados."""
        logger.info("Gerando relatório de processamento.")
        try:
            report = {
                'total_documents': self.stats.total_documents,
                'processed_documents': self.stats.processed_documents,
                'failed_documents': self.stats.failed_documents,
                'document_types': self.stats.document_types,
                'error_types': self.stats.error_types,
                'text_stats': self.stats.text_stats,
            }
            with open(self.output_dir / "processing_report.json", "w") as report_file:
                json.dump(report, report_file, ensure_ascii=False, indent=4)
            logger.info("Relatório gerado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")

    def clean_cache(self) -> None:
        """Limpa o cache do pipeline."""
        logger.info(f"Limpando o cache em {self.cache_dir}")
        try:
            for item in self.cache_dir.glob("*"):
                if item.is_dir():
                    psutil.Process(item.name).terminate()
                else:
                    item.unlink()
            logger.info("Cache limpo com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao limpar o cache: {e}")

    def get_processing_stats(self) -> ProcessingStats:
        """Retorna as estatísticas de processamento."""
        with self._stats_lock:
            return self.stats

class LegalDataPreparator:
    """
    Classe principal para preparação de dados jurídicos.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.chunk_size = config.get('chunk_size', 512)
        self.overlap = config.get('overlap', 50)
        self.min_chunk_length = config.get('min_chunk_length', 100)
        self.quality_threshold = config.get('quality_threshold', 0.7)
        self.min_text_length = config.get('min_text_length', 50)
        self.max_text_length = config.get('max_text_length', 10000)
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'samples_filtered': 0,
            'processing_time': 0
        }
        self.processor = DocumentProcessor()
        self._stats_lock = threading.Lock()

    def prepare_training_data(self, input_paths: List[str], output_dir: str, task_type: str = 'general') -> Dict[str, Any]:
        """
        Prepara dados para treinamento.
        Args:
            input_paths: Lista de caminhos de entrada
            output_dir: Diretório de saída
            task_type: Tipo de tarefa ('qa', 'summarization', 'classification', 'general')
        Returns:
            Relatório de preparação
        """
        start_time = time.time()
        self.logger.info(f"Iniciando preparação de dados para tarefa: {task_type}")
        os.makedirs(output_dir, exist_ok=True)
        all_samples = []
        for input_path in input_paths:
            if os.path.isfile(input_path):
                samples = self._process_single_file(input_path, output_dir)
                all_samples.extend(samples)
            elif os.path.isdir(input_path):
                for file in Path(input_path).rglob('*'):
                    if file.suffix.lower() in ['.pdf', '.docx']:
                        samples = self._process_single_file(str(file), output_dir)
                        all_samples.extend(samples)
            else:
                self.logger.warning(f"Caminho não encontrado: {input_path}")
        filtered_samples = [s for s in all_samples if len(s.get('text', '')) >= self.min_text_length]
        self.stats['processing_time'] = time.time() - start_time
        report = {
            'total_samples': len(filtered_samples),
            'task_type': task_type,
            'processing_time': self.stats['processing_time'],
        }
        self.logger.info(f"Preparação concluída: {len(filtered_samples)} amostras")
        return report

    def _process_single_file(self, file_path: str, output_dir: str) -> List[Dict[str, Any]]:
        self.logger.info(f"Processando arquivo: {file_path}")
        try:
            text = self.processor.extract_text_from_file(file_path)
            cleaned_text = self.processor.clean_text(text)
            entities = self.processor.extract_legal_entities(cleaned_text)
            anonymized_text = self.processor.anonymize_text(cleaned_text)
            chunks = self.processor.segment_document(anonymized_text, self.chunk_size, self.overlap)
            samples = []
            for chunk in chunks:
                sample = {'text': chunk, 'entities': entities}
                file_name = hashlib.md5(chunk.encode('utf-8')).hexdigest()
                file_path_out = os.path.join(output_dir, f"{file_name}.json")
                with open(file_path_out, "w", encoding="utf-8") as json_file:
                    json.dump(sample, json_file, ensure_ascii=False, indent=4)
                samples.append(sample)
            with self._stats_lock:
                self.stats['documents_processed'] += 1
                self.stats['chunks_created'] += len(chunks)
            return samples
        except Exception as e:
            self.logger.error(f"Erro ao processar o arquivo {file_path}: {e}")
            return []

# Funções utilitárias do módulo avançado de preparação de dados jurídicos
# Exemplos: validate_dataset_quality, create_training_splits, augment_legal_dataset, validate_dataset_format, merge_datasets

def validate_dataset_quality(dataset: pd.DataFrame, min_quality: float = 0.7) -> bool:
    """
    Valida a qualidade do dataset com base em métricas como tamanho do texto e diversidade de entidades.

    :param dataset: O dataset a ser validado.
    :param min_quality: O limiar mínimo de qualidade (0 a 1).
    :return: True se o dataset atender aos critérios de qualidade, False caso contrário.
    """
    # Calcular métricas de qualidade
    quality_metrics = dataset['text'].apply(lambda x: len(x) / len(x.split()))
    # Verificar se a qualidade média atende ao mínimo requerido
    return quality_metrics.mean() >= min_quality

def create_training_splits(dataset: pd.DataFrame, train_size: float = 0.8, val_size: float = 0.1, test_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cria as divisões de treinamento, validação e teste a partir do dataset completo.

    :param dataset: O dataset completo.
    :param train_size: A proporção de dados para o conjunto de treinamento.
    :param val_size: A proporção de dados para o conjunto de validação.
    :param test_size: A proporção de dados para o conjunto de teste.
    :return: Tupla contendo os dataframes de treino, validação e teste.
    """
    assert train_size + val_size + test_size == 1, "As proporções devem somar 1."
    train_val_set, test_set = train_test_split(dataset, test_size=test_size, random_state=42)
    train_set, val_set = train_test_split(train_val_set, test_size=val_size/(train_size+val_size), random_state=42)
    return train_set, val_set, test_set

def augment_legal_dataset(dataset: pd.DataFrame, nlp_model, num_augmented_samples: int = 1000) -> pd.DataFrame:
    """
    Aumenta o dataset jurídico com amostras sintéticas usando um modelo de linguagem.

    :param dataset: O dataset original.
    :param nlp_model: O modelo de linguagem para geração de texto.
    :param num_augmented_samples: O número de amostras aumentadas a serem geradas.
    :return: O dataset aumentado.
    """
    augmented_samples = []
    for _ in range(num_augmented_samples):
        sample = dataset.sample(1).iloc[0]
        generated_text = nlp_model.make_sentence(sample['text'])
        augmented_samples.append({
            'text': generated_text,
            'entities': sample['entities'],  # Manter as mesmas entidades da amostra original
        })
    return pd.DataFrame(augmented_samples)

def validate_dataset_format(dataset: pd.DataFrame, expected_columns: List[str]) -> bool:
    """
    Valida o formato do dataset verificando se contém as colunas esperadas.

    :param dataset: O dataset a ser validado.
    :param expected_columns: As colunas esperadas no dataset.
    :return: True se o formato estiver correto, False caso contrário.
    """
    return all(col in dataset.columns for col in expected_columns)

def merge_datasets(datasets: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Mescla múltiplos dataframes de datasets em um único dataframe.

    :param datasets: Lista de dataframes a serem mesclados.
    :return: Um único dataframe contendo todos os dados.
    """
    return pd.concat(datasets, ignore_index=True)

# Exemplo de uso
if __name__ == "__main__":
    # Configuração de exemplo
    config = {
        'chunk_size': 512,
        'overlap': 50,
        'min_chunk_length': 100,
        'quality_threshold': 0.7,
        'min_text_length': 50,
        'max_text_length': 10000
    }
    preparator = LegalDataPreparator(config)
    input_paths = ['data/raw/documentos_juridicos/']
    output_dir = 'data/processed/training/'
    report = preparator.prepare_training_data(
        input_paths=input_paths,
        output_dir=output_dir,
        task_type='qa'
    )
    print("Relatório de preparação:")
    print(json.dumps(report, indent=2, ensure_ascii=False))