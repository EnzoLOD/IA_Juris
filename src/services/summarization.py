"""
Summarization service for JurisOracle system.

Advanced document summarization with specialized support for legal documents,
multiple summarization techniques, and comprehensive quality control.
"""
import asyncio
import hashlib
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import json
import re

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from rouge_score import rouge_scorer
import spacy

from ..config import get_settings, get_logger
from ..models import Document as DocumentModel, QueryModel
from ..utils.text_processing import TextProcessor
from ..utils.validators import validate_text_length, validate_json_structure
from ..utils.exceptions import (
    SummarizationError, 
    ValidationError, 
    ProcessingError,
    ModelError
)

# Initialize components
logger = get_logger(__name__)
settings = get_settings()


class SummaryType(Enum):
    """Types of summary generation."""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive"
    HYBRID = "hybrid"
    KEYWORD = "keyword"
    BULLET_POINTS = "bullet_points"


class SummaryLevel(Enum):
    """Levels of summary detail."""
    EXECUTIVE = "executive"  # High-level overview
    DETAILED = "detailed"   # Comprehensive summary
    TECHNICAL = "technical" # Technical/legal details
    QUICK = "quick"        # Brief overview


class DocumentType(Enum):
    """Types of legal documents."""
    SENTENCA = "sentenca"
    ACORDAO = "acordao"
    CONTRATO = "contrato"
    PETICAO = "peticao"
    JURISPRUDENCIA = "jurisprudencia"
    LEGISLACAO = "legislacao"
    PARECER = "parecer"
    DESPACHO = "despacho"
    DECISAO = "decisao"
    GENERAL = "general"


@dataclass
class SummaryRequest:
    """Request configuration for summarization."""
    text: str
    summary_type: SummaryType = SummaryType.ABSTRACTIVE
    summary_level: SummaryLevel = SummaryLevel.DETAILED
    document_type: DocumentType = DocumentType.GENERAL
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    focus_areas: Optional[List[str]] = None
    language: str = "pt-br"
    include_keywords: bool = True
    include_entities: bool = True
    custom_instructions: Optional[str] = None


@dataclass
class SummaryMetrics:
    """Metrics for summary quality assessment."""
    compression_ratio: float
    readability_score: float
    coherence_score: float
    completeness_score: float
    legal_accuracy_score: float
    processing_time: float
    model_confidence: float
    rouge_scores: Dict[str, float]


@dataclass
class SummaryResult:
    """Complete summarization result."""
    summary: str
    keywords: List[str]
    key_entities: List[Dict[str, Any]]
    main_topics: List[str]
    legal_references: List[str]
    confidence_score: float
    metrics: SummaryMetrics
    metadata: Dict[str, Any]
    chunks_processed: int
    total_tokens: int
    summary_tokens: int


class LegalPromptTemplates:
    """Specialized prompt templates for legal document summarization."""
    
    SENTENCA_TEMPLATE = """
    Você é um especialista em análise de sentenças judiciais brasileiras.
    
    Analise a seguinte sentença e produza um resumo {level} seguindo esta estrutura:
    
    **DADOS PROCESSUAIS:**
    - Número do processo, comarca, vara
    
    **PARTES:**
    - Autor(es) e Réu(s)
    
    **OBJETO:**
    - Natureza da ação e pedidos
    
    **FUNDAMENTOS:**
    - Principais argumentos jurídicos
    
    **DECISÃO:**
    - Dispositivo da sentença
    
    **OBSERVAÇÕES:**
    - Recursos cabíveis, prazos, custas
    
    Texto da sentença:
    {text}
    
    Instruções adicionais: {custom_instructions}
    
    Resumo {level}:
    """
    
    ACORDAO_TEMPLATE = """
    Você é um especialista em análise de acórdãos de tribunais brasileiros.
    
    Analise o seguinte acórdão e produza um resumo {level} seguindo esta estrutura:
    
    **IDENTIFICAÇÃO:**
    - Tribunal, órgão julgador, relator
    
    **RECURSO:**
    - Tipo de recurso e origem
    
    **PARTES:**
    - Recorrente(s) e Recorrido(s)
    
    **MATÉRIA:**
    - Área do direito e questões jurídicas
    
    **TESES:**
    - Principais argumentos das partes
    
    **FUNDAMENTAÇÃO:**
    - Ratio decidendi do tribunal
    
    **DECISÃO:**
    - Resultado do julgamento
    
    **PRECEDENTE:**
    - Importância jurisprudencial
    
    Texto do acórdão:
    {text}
    
    Instruções adicionais: {custom_instructions}
    
    Resumo {level}:
    """
    
    CONTRATO_TEMPLATE = """
    Você é um especialista em análise de contratos e instrumentos jurídicos.
    
    Analise o seguinte contrato e produza um resumo {level} seguindo esta estrutura:
    
    **IDENTIFICAÇÃO:**
    - Tipo de contrato e finalidade
    
    **PARTES:**
    - Contratantes e suas qualificações
    
    **OBJETO:**
    - Descrição detalhada do objeto contratual
    
    **OBRIGAÇÕES:**
    - Principais obrigações de cada parte
    
    **CONDIÇÕES:**
    - Prazo, valor, forma de pagamento
    
    **CLÁUSULAS ESPECIAIS:**
    - Penalidades, garantias, rescisão
    
    **ASPECTOS JURÍDICOS:**
    - Lei aplicável e foro competente
    
    Texto do contrato:
    {text}
    
    Instruções adicionais: {custom_instructions}
    
    Resumo {level}:
    """
    
    GENERAL_TEMPLATE = """
    Você é um especialista em análise de documentos jurídicos brasileiros.
    
    Analise o seguinte documento jurídico e produza um resumo {level} considerando:
    
    1. **Natureza do documento**
    2. **Principais informações**
    3. **Aspectos jurídicos relevantes**
    4. **Consequências práticas**
    5. **Pontos de atenção**
    
    Documento:
    {text}
    
    Instruções adicionais: {custom_instructions}
    
    Resumo {level}:
    """


class SummarizationService:
    """
    Advanced summarization service with specialized support for legal documents.
    
    Features:
    - Multiple summarization techniques
    - Legal document type detection
    - Quality metrics and validation
    - Caching and performance optimization
    - Batch processing capabilities
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        text_processor: Optional[TextProcessor] = None,
        cache_ttl: int = 3600
    ):
        """Initialize the summarization service."""
        self.llm = llm
        self.text_processor = text_processor or TextProcessor()
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[SummaryResult, datetime]] = {}
        self._stats = {
            'requests_total': 0,
            'requests_cached': 0,
            'processing_time_total': 0.0,
            'errors_total': 0
        }
        
        # Initialize components
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len
        )
        
        # Load NLP model for entity extraction
        try:
            self.nlp = spacy.load("pt_core_news_sm")
        except OSError:
            logger.warning("Portuguese spaCy model not found. Entity extraction will be limited.")
            self.nlp = None
            
        # ROUGE scorer for quality metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # Prompt templates
        self.templates = LegalPromptTemplates()
        
        logger.info("SummarizationService initialized successfully")
    
    async def summarize(
        self,
        request: SummaryRequest,
        use_cache: bool = True
    ) -> SummaryResult:
        """
        Generate summary for the given text.
        
        Args:
            request: Summarization request configuration
            use_cache: Whether to use cached results
            
        Returns:
            Complete summarization result
        """
        start_time = time.time()
        self._stats['requests_total'] += 1
        
        try:
            # Validate request
            self._validate_request(request)
            
            # Generate cache key
            cache_key = self._generate_cache_key(request)
            
            # Check cache
            if use_cache and cache_key in self._cache:
                cached_result, cached_time = self._cache[cache_key]
                if datetime.now() - cached_time < timedelta(seconds=self.cache_ttl):
                    self._stats['requests_cached'] += 1
                    logger.info(f"Returning cached summary for key: {cache_key[:16]}...")
                    return cached_result
            
            # Process summarization
            result = await self._process_summarization(request)
            
            # Cache result
            if use_cache:
                self._cache[cache_key] = (result, datetime.now())
                self._cleanup_cache()
            
            # Update statistics
            processing_time = time.time() - start_time
            self._stats['processing_time_total'] += processing_time
            
            logger.info(
                f"Summary generated successfully. "
                f"Compression: {result.metrics.compression_ratio:.2f}, "
                f"Time: {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self._stats['errors_total'] += 1
            logger.error(f"Error in summarization: {str(e)}")
            raise SummarizationError(f"Failed to generate summary: {str(e)}")
    
    async def batch_summarize(
        self,
        requests: List[SummaryRequest],
        max_workers: int = 5
    ) -> List[SummaryResult]:
        """
        Process multiple summarization requests in batch.
        
        Args:
            requests: List of summarization requests
            max_workers: Maximum number of concurrent workers
            
        Returns:
            List of summarization results
        """
        logger.info(f"Starting batch summarization for {len(requests)} documents")
        
        async def process_request(request: SummaryRequest) -> SummaryResult:
            try:
                return await self.summarize(request)
            except Exception as e:
                logger.error(f"Error processing batch item: {str(e)}")
                raise
        
        # Process requests concurrently
        semaphore = asyncio.Semaphore(max_workers)
        
        async def limited_process(request: SummaryRequest) -> SummaryResult:
            async with semaphore:
                return await process_request(request)
        
        tasks = [limited_process(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        processed_results = []
        errors = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors += 1
                logger.error(f"Failed to process request {i}: {str(result)}")
                # Create error result
                error_result = self._create_error_result(str(result))
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        logger.info(
            f"Batch summarization completed. "
            f"Success: {len(results) - errors}, Errors: {errors}"
        )
        
        return processed_results
    
    async def _process_summarization(self, request: SummaryRequest) -> SummaryResult:
        """Process the actual summarization."""
        # Detect document type if not specified
        if request.document_type == DocumentType.GENERAL:
            request.document_type = self._detect_document_type(request.text)
        
        # Preprocess text
        processed_text = self.text_processor.clean_text(request.text)
        
        # Count tokens
        original_tokens = len(self.tokenizer.encode(processed_text))
        
        # Handle long documents
        if original_tokens > settings.MAX_TOKENS_PER_REQUEST:
            chunks = self._split_text(processed_text)
            summary = await self._summarize_chunks(chunks, request)
            chunks_processed = len(chunks)
        else:
            summary = await self._summarize_single(processed_text, request)
            chunks_processed = 1
        
        # Extract additional information
        keywords = self._extract_keywords(processed_text, summary)
        entities = self._extract_entities(processed_text)
        topics = self._extract_topics(summary)
        legal_refs = self._extract_legal_references(processed_text)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            original_text=processed_text,
            summary=summary,
            processing_time=0.0  # Will be set by caller
        )
        
        # Count summary tokens
        summary_tokens = len(self.tokenizer.encode(summary))
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(
            original_tokens, summary_tokens, metrics
        )
        
        # Create result
        result = SummaryResult(
            summary=summary,
            keywords=keywords,
            key_entities=entities,
            main_topics=topics,
            legal_references=legal_refs,
            confidence_score=confidence,
            metrics=metrics,
            metadata={
                'document_type': request.document_type.value,
                'summary_type': request.summary_type.value,
                'summary_level': request.summary_level.value,
                'language': request.language,
                'timestamp': datetime.now().isoformat()
            },
            chunks_processed=chunks_processed,
            total_tokens=original_tokens,
            summary_tokens=summary_tokens
        )
        
        return result
    
    async def _summarize_single(
        self, 
        text: str, 
        request: SummaryRequest
    ) -> str:
        """Summarize a single text without chunking."""
        prompt_template = self._get_prompt_template(request)
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text", "level", "custom_instructions"]
        )
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=settings.DEBUG
        )
        
        try:
            result = await chain.arun(
                text=text,
                level=request.summary_level.value,
                custom_instructions=request.custom_instructions or "Nenhuma instrução adicional."
            )
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Error in single text summarization: {str(e)}")
            raise ProcessingError(f"Failed to summarize text: {str(e)}")
    
    async def _summarize_chunks(
        self, 
        chunks: List[str], 
        request: SummaryRequest
    ) -> str:
        """Summarize text using map-reduce approach for long documents."""
        # First pass: summarize each chunk
        chunk_summaries = []
        
        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)}")
            
            chunk_request = SummaryRequest(
                text=chunk,
                summary_type=request.summary_type,
                summary_level=SummaryLevel.DETAILED,  # More detailed for chunks
                document_type=request.document_type,
                custom_instructions=request.custom_instructions
            )
            
            chunk_summary = await self._summarize_single(chunk, chunk_request)
            chunk_summaries.append(chunk_summary)
        
        # Second pass: combine chunk summaries
        combined_text = "\n\n".join(chunk_summaries)
        
        final_request = SummaryRequest(
            text=combined_text,
            summary_type=request.summary_type,
            summary_level=request.summary_level,
            document_type=request.document_type,
            custom_instructions=f"{request.custom_instructions}\n\nEste texto já contém resumos parciais. Crie um resumo final coerente e completo."
        )
        
        return await self._summarize_single(combined_text, final_request)
    
    def _get_prompt_template(self, request: SummaryRequest) -> str:
        """Get appropriate prompt template based on document type."""
        template_map = {
            DocumentType.SENTENCA: self.templates.SENTENCA_TEMPLATE,
            DocumentType.ACORDAO: self.templates.ACORDAO_TEMPLATE,
            DocumentType.CONTRATO: self.templates.CONTRATO_TEMPLATE,
            DocumentType.GENERAL: self.templates.GENERAL_TEMPLATE
        }
        
        return template_map.get(request.document_type, self.templates.GENERAL_TEMPLATE)
    
    def _detect_document_type(self, text: str) -> DocumentType:
        """Detect the type of legal document based on content."""
        text_lower = text.lower()
        
        # Patterns for different document types
        patterns = {
            DocumentType.SENTENCA: [
                'sentença', 'julgo procedente', 'julgo improcedente', 
                'dispositivo', 'condenado', 'absolvo'
            ],
            DocumentType.ACORDAO: [
                'acórdão', 'tribunal', 'relator', 'recurso', 
                'apelação', 'agravo', 'embargos'
            ],
            DocumentType.CONTRATO: [
                'contrato', 'contratante', 'contratado', 'cláusula',
                'partes contratantes', 'objeto do contrato'
            ],
            DocumentType.PETICAO: [
                'petição', 'requer', 'nos termos do', 'código de processo',
                'requerente', 'requerido'
            ]
        }
        
        scores = {}
        for doc_type, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[doc_type] = score
        
        # Return the type with highest score, or GENERAL if no clear match
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return DocumentType.GENERAL
    
    def _extract_keywords(self, text: str, summary: str) -> List[str]:
        """Extract relevant keywords from text and summary."""
        # Simple keyword extraction using frequency and legal relevance
        legal_terms = [
            'direito', 'lei', 'artigo', 'código', 'jurisprudência',
            'tribunal', 'juiz', 'processo', 'ação', 'sentença',
            'acórdão', 'recurso', 'petição', 'contrato', 'responsabilidade'
        ]
        
        combined_text = f"{text} {summary}".lower()
        words = re.findall(r'\b\w{4,}\b', combined_text)
        
        # Count frequency
        word_freq = {}
        for word in words:
            if word in legal_terms or len(word) > 5:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10] if freq > 1]
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text[:1000000])  # Limit text size for spaCy
            
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'description': spacy.explain(ent.label_),
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            return entities[:20]  # Limit number of entities
            
        except Exception as e:
            logger.warning(f"Error extracting entities: {str(e)}")
            return []
    
    def _extract_topics(self, summary: str) -> List[str]:
        """Extract main topics from summary."""
        # Simple topic extraction based on sentence structure
        sentences = re.split(r'[.!?]+', summary)
        topics = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                # Extract potential topic from sentence beginning
                words = sentence.split()[:8]
                topic = ' '.join(words).strip()
                if topic and len(topic) > 10:
                    topics.append(topic)
        
        return topics[:5]
    
    def _extract_legal_references(self, text: str) -> List[str]:
        """Extract legal references (laws, articles, etc.)."""
        patterns = [
            r'Lei\s+n[ºª°]?\s*\d+[\./]\d+',
            r'Art\w*\.?\s*\d+[ºª°]?',
            r'Código\s+\w+',
            r'CF\/\d+',
            r'STF|STJ|TST|TSE',
            r'Súmula\s+\d+'
        ]
        
        references = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(matches)
        
        return list(set(references))[:10]
    
    def _calculate_metrics(
        self, 
        original_text: str, 
        summary: str, 
        processing_time: float
    ) -> SummaryMetrics:
        """Calculate quality metrics for the summary."""
        # Compression ratio
        compression_ratio = len(summary) / len(original_text) if original_text else 0
        
        # ROUGE scores (simplified)
        try:
            rouge_scores = self.rouge_scorer.score(original_text, summary)
            rouge_dict = {
                'rouge1': rouge_scores['rouge1'].fmeasure,
                'rouge2': rouge_scores['rouge2'].fmeasure,
                'rougeL': rouge_scores['rougeL'].fmeasure
            }
        except Exception:
            rouge_dict = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        # Simple quality metrics
        readability_score = min(1.0, len(summary.split()) / 100)  # Simplified
        coherence_score = 0.8  # Placeholder - would need more sophisticated analysis
        completeness_score = min(1.0, compression_ratio * 10)  # Heuristic
        legal_accuracy_score = 0.85  # Placeholder - would need legal validation
        model_confidence = 0.9  # Placeholder - from model if available
        
        return SummaryMetrics(
            compression_ratio=compression_ratio,
            readability_score=readability_score,
            coherence_score=coherence_score,
            completeness_score=completeness_score,
            legal_accuracy_score=legal_accuracy_score,
            processing_time=processing_time,
            model_confidence=model_confidence,
            rouge_scores=rouge_dict
        )
    
    def _calculate_confidence_score(
        self, 
        original_tokens: int, 
        summary_tokens: int, 
        metrics: SummaryMetrics
    ) -> float:
        """Calculate overall confidence score for the summary."""
        # Combine various factors
        length_score = min(1.0, summary_tokens / max(1, original_tokens * 0.3))
        rouge_score = (metrics.rouge_scores['rouge1'] + 
                      metrics.rouge_scores['rougeL']) / 2
        
        confidence = (
            length_score * 0.3 +
            rouge_score * 0.4 +
            metrics.legal_accuracy_score * 0.3
        )
        
        return min(1.0, max(0.0, confidence))
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into manageable chunks."""
        documents = [Document(page_content=text)]
        chunks = self.text_splitter.split_documents(documents)
        return [chunk.page_content for chunk in chunks]
    
    def _validate_request(self, request: SummaryRequest) -> None:
        """Validate summarization request."""
        if not request.text or not request.text.strip():
            raise ValidationError("Text cannot be empty")
        
        if len(request.text) < 50:
            raise ValidationError("Text too short for meaningful summarization")
        
        if len(request.text) > settings.MAX_TEXT_LENGTH:
            raise ValidationError(f"Text too long. Maximum length: {settings.MAX_TEXT_LENGTH}")
    
    def _generate_cache_key(self, request: SummaryRequest) -> str:
        """Generate cache key for request."""
        key_data = {
            'text_hash': hashlib.md5(request.text.encode()).hexdigest(),
            'summary_type': request.summary_type.value,
            'summary_level': request.summary_level.value,
            'document_type': request.document_type.value,
            'custom_instructions': request.custom_instructions or ""
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, (result, cached_time) in self._cache.items():
            if current_time - cached_time > timedelta(seconds=self.cache_ttl):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _create_error_result(self, error_message: str) -> SummaryResult:
        """Create error result for failed summarization."""
        return SummaryResult(
            summary=f"Erro na sumarização: {error_message}",
            keywords=[],
            key_entities=[],
            main_topics=[],
            legal_references=[],
            confidence_score=0.0,
            metrics=SummaryMetrics(
                compression_ratio=0.0,
                readability_score=0.0,
                coherence_score=0.0,
                completeness_score=0.0,
                legal_accuracy_score=0.0,
                processing_time=0.0,
                model_confidence=0.0,
                rouge_scores={'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
            ),
            metadata={'error': True, 'error_message': error_message},
            chunks_processed=0,
            total_tokens=0,
            summary_tokens=0
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        avg_processing_time = (
            self._stats['processing_time_total'] / max(1, self._stats['requests_total'])
        )
        
        cache_hit_rate = (
            self._stats['requests_cached'] / max(1, self._stats['requests_total'])
        )
        
        return {
            'requests_total': self._stats['requests_total'],
            'requests_cached': self._stats['requests_cached'],
            'cache_hit_rate': cache_hit_rate,
            'avg_processing_time': avg_processing_time,
            'errors_total': self._stats['errors_total'],
            'cache_size': len(self._cache),
            'uptime': datetime.now().isoformat()
        }
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        logger.info("Summarization cache cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the service."""
        try:
            # Test with a simple summarization
            test_request = SummaryRequest(
                text="Este é um texto de teste para verificar o funcionamento do serviço de sumarização.",
                summary_type=SummaryType.ABSTRACTIVE,
                summary_level=SummaryLevel.QUICK
            )
            
            start_time = time.time()
            await self.summarize(test_request, use_cache=False)
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'response_time': response_time,
                'cache_size': len(self._cache),
                'statistics': self.get_statistics()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'cache_size': len(self._cache)
            }


# Utility functions
def create_summarization_service(
    llm: BaseLLM,
    text_processor: Optional[TextProcessor] = None
) -> SummarizationService:
    """Factory function to create summarization service."""
    return SummarizationService(
        llm=llm,
        text_processor=text_processor
    )


async def quick_summarize(
    text: str,
    llm: BaseLLM,
    summary_level: SummaryLevel = SummaryLevel.DETAILED
) -> str:
    """Quick summarization utility function."""
    service = create_summarization_service(llm)
    
    request = SummaryRequest(
        text=text,
        summary_level=summary_level
    )
    
    result = await service.summarize(request)
    return result.summary


# Export main classes
__all__ = [
    'SummarizationService',
    'SummaryRequest',
    'SummaryResult',
    'SummaryMetrics',
    'SummaryType',
    'SummaryLevel',
    'DocumentType',
    'LegalPromptTemplates',
    'create_summarization_service',
    'quick_summarize'
]