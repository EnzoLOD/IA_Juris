"""
Pipeline RAG (Retrieval-Augmented Generation) para JurisOracle
Implementa recuperação de documentos jurídicos com embeddings e geração de respostas
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import pickle

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Representa um chunk de documento com metadados"""
    id: str
    content: str
    source_document: str
    chunk_index: int
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

@dataclass
class RetrievalResult:
    """Resultado de uma recuperação de documento"""
    chunk: DocumentChunk
    score: float
    rank: int

@dataclass
class RAGResponse:
    """Resposta completa do sistema RAG"""
    query: str
    answer: str
    sources: List[RetrievalResult]
    confidence: float
    processing_time: float
    model_used: str
    timestamp: str

class JurisRAGPipeline:
    """
    Pipeline RAG principal para análise de documentos jurídicos
    
    Funcionalidades:
    - Embedding de documentos com modelos especializados em português
    - Armazenamento em vector store (FAISS)
    - Recuperação semântica de documentos relevantes
    - Geração de respostas usando LLM
    - Suporte a chunking inteligente
    - Persistência de embeddings
    """
    
    def __init__(
        self,
        embedding_model: str = "neuralmind/bert-base-portuguese-cased",
        llm_model: str = "microsoft/DialoGPT-medium",
        vector_store_path: str = "./data/vector_store",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        device: str = "auto"
    ):
        """
        Inicializa o pipeline RAG
        
        Args:
            embedding_model: Modelo para embeddings
            llm_model: Modelo para geração de texto
            vector_store_path: Caminho para salvar vector store
            chunk_size: Tamanho dos chunks de texto
            chunk_overlap: Sobreposição entre chunks
            device: Dispositivo (cpu/cuda/auto)
        """
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.vector_store_path = vector_store_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Detecta dispositivo
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Inicializando RAG Pipeline com dispositivo: {self.device}")
        
        # Inicializa componentes
        self._initialize_embedding_model()
        self._initialize_llm()
        self._initialize_vector_store()
        
        # Storage para chunks
        self.document_chunks: List[DocumentChunk] = []
        self.chunk_id_to_index: Dict[str, int] = {}
        
        # Métricas
        self.total_documents = 0
        self.total_chunks = 0
        
    def _initialize_embedding_model(self):
        """Inicializa modelo de embeddings"""
        try:
            logger.info(f"Carregando modelo de embedding: {self.embedding_model_name}")
            self.embedder = SentenceTransformer(
                self.embedding_model_name,
                device=self.device
            )
            self.embedding_dimension = self.embedder.get_sentence_embedding_dimension()
            logger.info(f"Modelo carregado. Dimensão: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de embedding: {e}")
            raise
    
    def _initialize_llm(self):
        """Inicializa modelo de linguagem para geração"""
        try:
            logger.info(f"Carregando LLM: {self.llm_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Configura pad token se necessário
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("LLM carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar LLM: {e}")
            # Fallback para pipeline simples
            logger.info("Usando pipeline simples como fallback")
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.llm_model_name,
                device=0 if self.device == "cuda" else -1
            )
    
    def _initialize_vector_store(self):
        """Inicializa ou carrega vector store FAISS"""
        self.index_file = os.path.join(self.vector_store_path, "faiss_index.bin")
        self.metadata_file = os.path.join(self.vector_store_path, "metadata.pkl")
        
        # Cria diretório se não existir
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            self._load_vector_store()
        else:
            self._create_new_vector_store()
    
    def _create_new_vector_store(self):
        """Cria novo vector store vazio"""
        logger.info("Criando novo vector store")
        self.vector_store = faiss.IndexFlatIP(self.embedding_dimension)
        self.document_chunks = []
        self.chunk_id_to_index = {}
    
    def _load_vector_store(self):
        """Carrega vector store existente"""
        try:
            logger.info("Carregando vector store existente")
            self.vector_store = faiss.read_index(self.index_file)
            
            with open(self.metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.document_chunks = data['chunks']
                self.chunk_id_to_index = data['chunk_mapping']
                
            logger.info(f"Vector store carregado: {len(self.document_chunks)} chunks")
        except Exception as e:
            logger.error(f"Erro ao carregar vector store: {e}")
            self._create_new_vector_store()
    
    def save_vector_store(self):
        """Salva vector store no disco"""
        try:
            logger.info("Salvando vector store")
            faiss.write_index(self.vector_store, self.index_file)
            
            metadata = {
                'chunks': self.document_chunks,
                'chunk_mapping': self.chunk_id_to_index
            }
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info("Vector store salvo com sucesso")
        except Exception as e:
            logger.error(f"Erro ao salvar vector store: {e}")
            raise
    
    def _create_chunks(self, text: str, document_id: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Cria chunks de texto com sobreposição
        
        Args:
            text: Texto para dividir
            document_id: ID do documento fonte
            metadata: Metadados do documento
            
        Returns:
            Lista de chunks criados
        """
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text.strip()) < 50:  # Skip chunks muito pequenos
                continue
                
            chunk_id = f"{document_id}_chunk_{len(chunks)}"
            
            chunk = DocumentChunk(
                id=chunk_id,
                content=chunk_text,
                source_document=document_id,
                chunk_index=len(chunks),
                metadata=metadata.copy()
            )
            
            chunks.append(chunk)
            
        return chunks
    
    def add_document(
        self,
        content: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Adiciona documento ao pipeline
        
        Args:
            content: Conteúdo do documento
            document_id: ID único do documento
            metadata: Metadados opcionais
            
        Returns:
            Número de chunks criados
        """
        if metadata is None:
            metadata = {}
            
        logger.info(f"Adicionando documento: {document_id}")
        
        # Cria chunks
        chunks = self._create_chunks(content, document_id, metadata)
        
        if not chunks:
            logger.warning(f"Nenhum chunk criado para documento {document_id}")
            return 0
        
        # Gera embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.encode(
            chunk_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Adiciona embeddings aos chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            
        # Adiciona ao vector store
        self.vector_store.add(embeddings.astype('float32'))
        
        # Atualiza índices
        start_index = len(self.document_chunks)
        for i, chunk in enumerate(chunks):
            self.document_chunks.append(chunk)
            self.chunk_id_to_index[chunk.id] = start_index + i
            
        self.total_documents += 1
        self.total_chunks += len(chunks)
        
        logger.info(f"Documento adicionado: {len(chunks)} chunks criados")
        return len(chunks)
    
    def add_documents_batch(
        self,
        documents: List[Tuple[str, str, Optional[Dict[str, Any]]]]
    ) -> int:
        """
        Adiciona múltiplos documentos em batch
        
        Args:
            documents: Lista de (content, document_id, metadata)
            
        Returns:
            Total de chunks criados
        """
        total_chunks = 0
        
        for content, doc_id, metadata in documents:
            chunks_created = self.add_document(content, doc_id, metadata)
            total_chunks += chunks_created
            
        # Salva após batch
        self.save_vector_store()
        
        return total_chunks
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Recupera documentos relevantes para a query
        
        Args:
            query: Pergunta/consulta
            k: Número de resultados
            score_threshold: Score mínimo para incluir resultado
            
        Returns:
            Lista de resultados ranqueados
        """
        if self.vector_store.ntotal == 0:
            logger.warning("Vector store vazio")
            return []
        
        # Gera embedding da query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        
        # Busca no vector store
        scores, indices = self.vector_store.search(
            query_embedding.astype('float32'), 
            min(k, self.vector_store.ntotal)
        )
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if score >= score_threshold and idx < len(self.document_chunks):
                chunk = self.document_chunks[idx]
                result = RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    rank=rank
                )
                results.append(result)
        
        return results
    
    def _generate_answer(
        self,
        query: str,
        context_chunks: List[DocumentChunk],
        max_length: int = 512
    ) -> str:
        """
        Gera resposta usando LLM com contexto recuperado
        
        Args:
            query: Pergunta original
            context_chunks: Chunks de contexto
            max_length: Tamanho máximo da resposta
            
        Returns:
            Resposta gerada
        """
        # Prepara contexto
        context_texts = [chunk.content for chunk in context_chunks]
        context = "\n\n".join(context_texts[:3])  # Usa apenas top 3
        
        # Cria prompt
        prompt = f"""
Contexto jurídico:
{context}

Pergunta: {query}

Resposta baseada no contexto acima:"""
        
        try:
            # Usa o modelo carregado
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                max_length=1024, 
                truncation=True
            )
            
            if self.device == "cuda":
                inputs = inputs.to("cuda")
            
            with torch.no_grad():
                outputs = self.llm.generate(
                    inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove o prompt da resposta
            if "Resposta baseada no contexto acima:" in response:
                response = response.split("Resposta baseada no contexto acima:")[-1].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Erro na geração: {e}")
            return "Desculpe, não foi possível gerar uma resposta neste momento."
    
    def _calculate_confidence(self, results: List[RetrievalResult]) -> float:
        """
        Calcula confiança da resposta baseada nos scores de recuperação
        
        Args:
            results: Resultados da recuperação
            
        Returns:
            Score de confiança (0-1)
        """
        if not results:
            return 0.0
            
        # Usa score do melhor resultado e diversidade
        top_score = results[0].score
        score_variance = np.var([r.score for r in results]) if len(results) > 1 else 0
        
        # Normaliza e combina métricas
        confidence = min(top_score / 1.0, 1.0)  # Assume max score ~1.0
        confidence *= (1 - min(score_variance, 0.3) / 0.3)  # Penaliza alta variância
        
        return max(0.0, min(1.0, confidence))
    
    def query(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.1,
        generate_answer: bool = True
    ) -> RAGResponse:
        """
        Executa query completa no pipeline RAG
        
        Args:
            query: Pergunta
            k: Número de documentos para recuperar
            score_threshold: Score mínimo
            generate_answer: Se deve gerar resposta ou só recuperar
            
        Returns:
            Resposta completa do RAG
        """
        start_time = datetime.now()
        
        # Recupera documentos
        results = self.retrieve(query, k, score_threshold)
        
        # Gera resposta se solicitado
        answer = ""
        if generate_answer and results:
            chunks = [r.chunk for r in results]
            answer = self._generate_answer(query, chunks)
        elif generate_answer:
            answer = "Não foram encontrados documentos relevantes para sua pergunta."
        
        # Calcula confiança
        confidence = self._calculate_confidence(results)
        
        # Tempo de processamento
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return RAGResponse(
            query=query,
            answer=answer,
            sources=results,
            confidence=confidence,
            processing_time=processing_time,
            model_used=self.embedding_model_name,
            timestamp=datetime.now().isoformat()
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do pipeline"""
        return {
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "embedding_dimension": self.embedding_dimension,
            "vector_store_size": self.vector_store.ntotal if hasattr(self, 'vector_store') else 0,
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model_name,
            "device": self.device
        }
    
    def clear_vector_store(self):
        """Limpa vector store (cuidado!)"""
        logger.warning("Limpando vector store")
        self._create_new_vector_store()
        self.total_documents = 0
        self.total_chunks = 0
    
    def __del__(self):
        """Salva vector store ao destruir objeto"""
        try:
            if hasattr(self, 'vector_store'):
                self.save_vector_store()
        except:
            pass