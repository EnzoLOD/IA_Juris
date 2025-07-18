"""
Embedding generation module for legal documents.
Handles text encoding using various embedding models with optimization for legal content.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
import hashlib
import pickle
import os
from pathlib import Path

from ..config.settings import Settings

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Advanced embedding generator optimized for legal documents.
    Supports multiple models with caching and batch processing.
    """
    
    def __init__(self, model_name: str = None, cache_dir: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the embedding model to use
            cache_dir: Directory for caching embeddings
        """
        self.settings = Settings()
        self.model_name = model_name or self.settings.EMBEDDING_MODEL
        self.cache_dir = Path(cache_dir or self.settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = 512
        self.batch_size = 32
        
        # Cache para embeddings
        self._embedding_cache = {}
        self._load_cache()
        
        logger.info(f"Inicializando EmbeddingGenerator com modelo: {self.model_name}")
        logger.info(f"Dispositivo: {self.device}")
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            if 'sentence-transformers' in self.model_name.lower() or 'all-' in self.model_name:
                self.model = SentenceTransformer(self.model_name, device=str(self.device))
                logger.info(f"Modelo SentenceTransformer carregado: {self.model_name}")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
                logger.info(f"Modelo Transformers carregado: {self.model_name}")
                
        except Exception as e:
            logger.error(f"Erro ao carregar modelo {self.model_name}: {str(e)}")
            # Fallback para modelo padrão
            self.model_name = "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name, device=str(self.device))
            logger.warning(f"Usando modelo fallback: {self.model_name}")
    
    def _load_cache(self):
        """Load embedding cache from disk."""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self._embedding_cache = pickle.load(f)
                logger.info(f"Cache carregado: {len(self._embedding_cache)} embeddings")
        except Exception as e:
            logger.warning(f"Erro ao carregar cache: {str(e)}")
            self._embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self._embedding_cache, f)
            logger.debug("Cache salvo com sucesso")
        except Exception as e:
            logger.warning(f"Erro ao salvar cache: {str(e)}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        text_hash = hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()
        return text_hash
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    async def generate_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            use_cache: Whether to use cache
            
        Returns:
            List of embedding values
        """
        if not text or not text.strip():
            logger.warning("Texto vazio fornecido para embedding")
            return [0.0] * 384  # Dimensão padrão
        
        # Verificar cache
        cache_key = self._get_cache_key(text)
        if use_cache and cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # Carregar modelo se necessário
        if self.model is None:
            self._load_model()
        
        try:
            # Gerar embedding
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, self._generate_single_embedding, text)
            
            # Salvar no cache
            if use_cache:
                self._embedding_cache[cache_key] = embedding
                if len(self._embedding_cache) % 100 == 0:  # Salvar cache periodicamente
                    self._save_cache()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Erro ao gerar embedding: {str(e)}")
            return [0.0] * 384
    
    def _generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text (synchronous)."""
        try:
            if isinstance(self.model, SentenceTransformer):
                embedding = self.model.encode(text, convert_to_tensor=False, show_progress_bar=False)
                return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            else:
                # Usar modelo Transformers
                encoded_input = self.tokenizer(
                    text, 
                    padding=True, 
                    truncation=True, 
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    sentence_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
                    sentence_embedding = torch.nn.functional.normalize(sentence_embedding, p=2, dim=1)
                
                return sentence_embedding.cpu().numpy().flatten().tolist()
                
        except Exception as e:
            logger.error(f"Erro na geração de embedding: {str(e)}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str], use_cache: bool = True, batch_size: int = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of input texts
            use_cache: Whether to use cache
            batch_size: Batch size for processing
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.batch_size
        
        # Carregar modelo se necessário
        if self.model is None:
            self._load_model()
        
        try:
            # Verificar cache primeiro
            embeddings = []
            texts_to_process = []
            cache_indices = []
            
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    embeddings.append([0.0] * 384)
                    continue
                    
                cache_key = self._get_cache_key(text)
                if use_cache and cache_key in self._embedding_cache:
                    embeddings.append(self._embedding_cache[cache_key])
                else:
                    texts_to_process.append(text)
                    cache_indices.append(i)
                    embeddings.append(None)  # Placeholder
            
            # Processar textos não encontrados no cache
            if texts_to_process:
                logger.info(f"Processando {len(texts_to_process)} textos em batches de {batch_size}")
                
                loop = asyncio.get_event_loop()
                new_embeddings = await loop.run_in_executor(
                    None, 
                    self._generate_batch_embeddings, 
                    texts_to_process, 
                    batch_size
                )
                
                # Inserir novos embeddings e atualizar cache
                for i, embedding in enumerate(new_embeddings):
                    original_index = cache_indices[i]
                    embeddings[original_index] = embedding
                    
                    if use_cache:
                        cache_key = self._get_cache_key(texts_to_process[i])
                        self._embedding_cache[cache_key] = embedding
                
                # Salvar cache
                if use_cache:
                    self._save_cache()
            
            logger.info(f"Gerados {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Erro ao gerar embeddings em batch: {str(e)}")
            raise
    
    def _generate_batch_embeddings(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Generate embeddings for batch of texts (synchronous)."""
        try:
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                if isinstance(self.model, SentenceTransformer):
                    batch_embeddings = self.model.encode(
                        batch, 
                        convert_to_tensor=False, 
                        show_progress_bar=False,
                        batch_size=len(batch)
                    )
                    
                    # Converter para lista se necessário
                    if hasattr(batch_embeddings, 'tolist'):
                        batch_embeddings = batch_embeddings.tolist()
                    elif isinstance(batch_embeddings, np.ndarray):
                        batch_embeddings = batch_embeddings.tolist()
                    
                    all_embeddings.extend(batch_embeddings)
                else:
                    # Processar com modelo Transformers
                    encoded_input = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    with torch.no_grad():
                        model_output = self.model(**encoded_input)
                        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                        
                        batch_embeddings = sentence_embeddings.cpu().numpy().tolist()
                        all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Processado batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Erro no processamento batch: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self.model is None:
            self._load_model()
        
        try:
            if isinstance(self.model, SentenceTransformer):
                return self.model.get_sentence_embedding_dimension()
            else:
                # Testar com texto dummy
                test_embedding = self._generate_single_embedding("test")
                return len(test_embedding)
        except Exception as e:
            logger.warning(f"Erro ao obter dimensão: {str(e)}")
            return 384  # Dimensão padrão
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Normalizar vetores
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            
            # Calcular similaridade cosseno
            similarity = np.dot(vec1_norm, vec2_norm)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Erro ao calcular similaridade: {str(e)}")
            return 0.0
    
    def cleanup_cache(self, max_size: int = 10000):
        """Clean up cache if it gets too large."""
        if len(self._embedding_cache) > max_size:
            # Manter apenas os mais recentes (aproximação simples)
            cache_items = list(self._embedding_cache.items())
            self._embedding_cache = dict(cache_items[-max_size//2:])
            self._save_cache()
            logger.info(f"Cache limpo, mantidos {len(self._embedding_cache)} embeddings")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self._save_cache()
        except:
            pass

