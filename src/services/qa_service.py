"""
Question Answering Service for JurisOracle system.

This module provides comprehensive Q&A capabilities for legal documents,
including contextual understanding, citation management, and confidence scoring.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, validator

from ..config import get_settings
from ..core.vector_store import VectorStore
from ..core.hyde_retriever import HyDERetriever
from ..models.query import QueryRequest, QueryResponse, QueryType
from ..models.document import Document, DocumentMetadata
from ..utils.exceptions import JurisOracleError
from ..utils.text_processing import TextProcessor
from ..utils.validators import validate_legal_query

# Configure logging
logger = logging.getLogger(__name__)

class QAType(str, Enum):
    """Types of QA operations."""
    SIMPLE = "simple"
    CONTEXTUAL = "contextual"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    JURISPRUDENTIAL = "jurisprudential"

class ConfidenceLevel(str, Enum):
    """Confidence levels for answers."""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"           # 75-89%
    MEDIUM = "medium"       # 50-74%
    LOW = "low"            # 25-49%
    VERY_LOW = "very_low"  # 0-24%

@dataclass
class QAContext:
    """Context for QA operations."""
    documents: List[Document] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    legal_area: Optional[str] = None
    jurisdiction: Optional[str] = None
    date_range: Optional[Tuple[datetime, datetime]] = None
    previous_queries: List[str] = field(default_factory=list)

@dataclass
class Citation:
    """Legal citation information."""
    document_id: str
    document_title: str
    article_number: Optional[str] = None
    paragraph: Optional[str] = None
    page_number: Optional[int] = None
    relevance_score: float = 0.0
    text_excerpt: str = ""
    
class QAResponse(BaseModel):
    """Response model for QA operations."""
    question: str
    answer: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    qa_type: QAType
    citations: List[Citation] = Field(default_factory=list)
    related_topics: List[str] = Field(default_factory=list)
    legal_reasoning: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time: float = 0.0
    tokens_used: int = 0
    sources_count: int = 0
    
    @validator('confidence_level', pre=True, always=True)
    def set_confidence_level(cls, v, values):
        """Set confidence level based on score."""
        score = values.get('confidence_score', 0.0)
        if score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.75:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

class QACache:
    """Cache for QA responses."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[QAResponse, float]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
    def _generate_key(self, question: str, context: QAContext) -> str:
        """Generate cache key."""
        context_str = json.dumps({
            'legal_area': context.legal_area,
            'jurisdiction': context.jurisdiction,
            'doc_count': len(context.documents)
        }, sort_keys=True)
        
        combined = f"{question}:{context_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, question: str, context: QAContext) -> Optional[QAResponse]:
        """Get cached response."""
        key = self._generate_key(question, context)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                logger.debug(f"Cache hit for question: {question[:50]}...")
                return response
            else:
                del self.cache[key]
        return None
    
    def set(self, question: str, context: QAContext, response: QAResponse):
        """Cache response."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        key = self._generate_key(question, context)
        self.cache[key] = (response, time.time())
        logger.debug(f"Cached response for question: {question[:50]}...")

class QAService:
    """Comprehensive Question Answering service for legal documents."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        hyde_retriever: HyDERetriever,
        openai_client: Optional[AsyncOpenAI] = None,
        enable_cache: bool = True
    ):
        self.settings = get_settings()
        self.vector_store = vector_store
        self.hyde_retriever = hyde_retriever
        self.text_processor = TextProcessor()
        
        # Initialize OpenAI client
        self.openai_client = openai_client or AsyncOpenAI(
            api_key=self.settings.OPENAI_API_KEY
        )
        
        # Initialize cache
        self.cache = QACache() if enable_cache else None
        
        # QA prompts for different types
        self.qa_prompts = {
            QAType.SIMPLE: self._get_simple_qa_prompt(),
            QAType.CONTEXTUAL: self._get_contextual_qa_prompt(),
            QAType.COMPARATIVE: self._get_comparative_qa_prompt(),
            QAType.ANALYTICAL: self._get_analytical_qa_prompt(),
            QAType.PROCEDURAL: self._get_procedural_qa_prompt(),
            QAType.JURISPRUDENTIAL: self._get_jurisprudential_qa_prompt()
        }
        
        # Statistics
        self.stats = {
            'total_questions': 0,
            'cache_hits': 0,
            'avg_processing_time': 0.0,
            'confidence_distribution': {level.value: 0 for level in ConfidenceLevel}
        }
        
        logger.info("QA Service initialized successfully")
    
    async def answer_question(
        self, 
        question: str, 
        context: QAContext = None, 
        qa_type: QAType = QAType.SIMPLE
    ) -> QAResponse:
        """Asynchronously answer a question using the best available method."""
        logger.info(f"Received question: {question}")
        self.stats['total_questions'] += 1
        
        # Validate and preprocess question
        if not validate_legal_query(question):
            raise JurisOracleError("Invalid legal question format.")
        
        # Check cache first
        if self.cache:
            cached_response = self.cache.get(question, context)
            if cached_response:
                self.stats['cache_hits'] += 1
                logger.info(f"Cache hit for question: {question}")
                return cached_response
        
        # Determine QA method based on type
        if qa_type == QAType.SIMPLE:
            response = await self._perform_simple_qa(question, context)
        elif qa_type == QAType.CONTEXTUAL:
            response = await self._perform_contextual_qa(question, context)
        elif qa_type == QAType.COMPARATIVE:
            response = await self._perform_comparative_qa(question, context)
        elif qa_type == QAType.ANALYTICAL:
            response = await self._perform_analytical_qa(question, context)
        elif qa_type == QAType.PROCEDURAL:
            response = await self._perform_procedural_qa(question, context)
        elif qa_type == QAType.JURISPRUDENTIAL:
            response = await self._perform_jurisprudential_qa(question, context)
        else:
            raise JurisOracleError(f"Unsupported QA type: {qa_type}")
        
        # Cache the response if caching is enabled
        if self.cache:
            self.cache.set(question, context, response)
        
        return response
    
    async def _perform_simple_qa(
        self, question: str, context: QAContext
    ) -> QAResponse:
        """Perform simple QA without context."""
        start_time = time.time()
        documents = self.vector_store.search(question, top_k=5)
        answer = self._generate_answer(documents, question)
        response = QAResponse(
            question=question,
            answer=answer,
            confidence_score=self._estimate_confidence(answer),
            qa_type=QAType.SIMPLE,
            citations=self._extract_citations(documents),
            processing_time=time.time() - start_time,
            tokens_used=len(answer.split()),
            sources_count=len(documents)
        )
        logger.info(f"Simple QA performed in {response.processing_time:.2f}s")
        return response
    
    async def _perform_contextual_qa(
        self, question: str, context: QAContext
    ) -> QAResponse:
        """Perform QA using contextual information."""
        start_time = time.time()
        context_documents = context.documents or self.vector_store.search(question, top_k=5)
        answer = self._generate_answer(context_documents, question)
        response = QAResponse(
            question=question,
            answer=answer,
            confidence_score=self._estimate_confidence(answer),
            qa_type=QAType.CONTEXTUAL,
            citations=self._extract_citations(context_documents),
            processing_time=time.time() - start_time,
            tokens_used=len(answer.split()),
            sources_count=len(context_documents)
        )
        logger.info(f"Contextual QA performed in {response.processing_time:.2f}s")
        return response
    
    async def _perform_comparative_qa(
        self, question: str, context: QAContext
    ) -> QAResponse:
        """Perform comparative QA between documents."""
        start_time = time.time()
        if not context.documents or len(context.documents) < 2:
            raise JurisOracleError("Comparative QA requires at least 2 documents in context.")
        
        # Generate comparative answer
        answer = self._generate_comparative_answer(context.documents, question)
        response = QAResponse(
            question=question,
            answer=answer,
            confidence_score=self._estimate_confidence(answer),
            qa_type=QAType.COMPARATIVE,
            citations=self._extract_citations(context.documents),
            processing_time=time.time() - start_time,
            tokens_used=len(answer.split()),
            sources_count=len(context.documents)
        )
        logger.info(f"Comparative QA performed in {response.processing_time:.2f}s")
        return response
    
    async def _perform_analytical_qa(
        self, question: str, context: QAContext
    ) -> QAResponse:
        """Perform analytical QA requiring deep legal analysis."""
        start_time = time.time()
        documents = self.vector_store.search(question, top_k=5)
        answer = await self._generate_analytical_answer(documents, question)
        response = QAResponse(
            question=question,
            answer=answer,
            confidence_score=self._estimate_confidence(answer),
            qa_type=QAType.ANALYTICAL,
            citations=self._extract_citations(documents),
            processing_time=time.time() - start_time,
            tokens_used=len(answer.split()),
            sources_count=len(documents)
        )
        logger.info(f"Analytical QA performed in {response.processing_time:.2f}s")
        return response
    
    async def _perform_procedural_qa(
        self, question: str, context: QAContext
    ) -> QAResponse:
        """Perform procedural QA for legal procedures."""
        start_time = time.time()
        documents = self.vector_store.search(question, top_k=5)
        answer = self._generate_procedural_answer(documents, question)
        response = QAResponse(
            question=question,
            answer=answer,
            confidence_score=self._estimate_confidence(answer),
            qa_type=QAType.PROCEDURAL,
            citations=self._extract_citations(documents),
            processing_time=time.time() - start_time,
            tokens_used=len(answer.split()),
            sources_count=len(documents)
        )
        logger.info(f"Procedural QA performed in {response.processing_time:.2f}s")
        return response
    
    async def _perform_jurisprudential_qa(
        self, question: str, context: QAContext
    ) -> QAResponse:
        """Perform QA based on jurisprudential analysis."""
        start_time = time.time()
        documents = self.vector_store.search(question, top_k=5)
        answer = await self._generate_jurisprudential_answer(documents, question)
        response = QAResponse(
            question=question,
            answer=answer,
            confidence_score=self._estimate_confidence(answer),
            qa_type=QAType.JURISPRUDENTIAL,
            citations=self._extract_citations(documents),
            processing_time=time.time() - start_time,
            tokens_used=len(answer.split()),
            sources_count=len(documents)
        )
        logger.info(f"Jurisprudential QA performed in {response.processing_time:.2f}s")
        return response
    
    def _generate_answer(self, documents, question):
        # Implement logic to generate an answer based on the documents and the question
        # This could involve using a language model or other techniques
        pass
    
    def _generate_comparative_answer(self, documents, question):
        # Implement logic to generate a comparative answer based on the documents and the question
        pass
    
    async def _generate_analytical_answer(self, documents, question):
        """Generate an analytical answer using OpenAI."""
        prompt = self.qa_prompts[QAType.ANALYTICAL]
        response = await self.openai_client.completions.create(
            engine="davinci",
            prompt=prompt.format(question=question, context=documents),
            max_tokens=500,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n\n"]
        )
        answer = response.choices[0].text.strip()
        return answer
    
    async def _generate_jurisprudential_answer(self, documents, question):
        """Generate a jurisprudential answer using OpenAI."""
        prompt = self.qa_prompts[QAType.JURISPRUDENTIAL]
        response = await self.openai_client.completions.create(
            engine="davinci",
            prompt=prompt.format(question=question, context=documents),
            max_tokens=500,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n\n"]
        )
        answer = response.choices[0].text.strip()
        return answer

    def _extract_citations(self, documents):
        """Extract citations from documents."""
        citations = []
        for doc in documents:
            if isinstance(doc, Document) and doc.metadata.get('citations'):
                for cite in doc.metadata['citations']:
                    citations.append(Citation(
                        document_id=doc.id,
                        document_title=doc.title,
                        article_number=cite.get('article_number'),
                        paragraph=cite.get('paragraph'),
                        page_number=cite.get('page_number'),
                        relevance_score=cite.get('relevance_score', 0.0),
                        text_excerpt=cite.get('text_excerpt', '')
                    ))
        return citations
    
    def _get_simple_qa_prompt(self):
        """Get prompt template for simple QA."""
        return "Q: {question}\nA:"
    
    def _get_contextual_qa_prompt(self):
        """Get prompt template for contextual QA."""
        return "Given the context: {context}\nQ: {question}\nA:"
    
    def _get_comparative_qa_prompt(self):
        """Get prompt template for comparative QA."""
        return "Compare the following documents: {context}\nQ: {question}\nA:"
    
    def _get_analytical_qa_prompt(self):
        """Get prompt template for analytical QA."""
        return "Analyze the following documents for the question: {question}\nDocuments: {context}\nA:"
    
    def _get_procedural_qa_prompt(self):
        """Get prompt template for procedural QA."""
        return "Outline the procedure based on the following documents: {context}\nQ: {question}\nA:"
    
    def _get_jurisprudential_qa_prompt(self):
        """Get prompt template for jurisprudential QA."""
        return "Provide a jurisprudential analysis for the following documents: {context}\nQ: {question}\nA:"