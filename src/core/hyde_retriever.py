"""
HyDE (Hypothetical Document Embeddings) Retriever para JurisOracle
Implementa recuperação avançada usando documentos hipotéticos gerados por LLM
para melhorar a qualidade da busca semântica em documentos jurídicos
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, AutoModelForSequenceClassification
)
import torch
from sentence_transformers import SentenceTransformer, util
import re
from collections import defaultdict

from .rag_pipeline import JurisRAGPipeline, DocumentChunk, RetrievalResult, RAGResponse

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HypotheticalDocument:
    """Representa um documento hipotético gerado"""
    query: str
    content: str
    generation_method: str
    confidence: float
    tokens_used: int
    generation_time: float
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

@dataclass
class HyDEResult:
    """Resultado de recuperação HyDE"""
    original_result: RetrievalResult
    hyde_score: float
    direct_score: float
    combined_score: float
    retrieval_method: str  # 'hyde', 'direct', 'hybrid'
    hypothetical_docs: List[HypotheticalDocument]

@dataclass
class HyDEResponse:
    """Resposta completa do sistema HyDE"""
    query: str
    answer: str
    results: List[HyDEResult]
    hypothetical_documents: List[HypotheticalDocument]
    confidence: float
    processing_time: float
    total_tokens_used: int
    strategy_used: str
    timestamp: str

class JuridicalPromptTemplates:
    """Templates de prompts especializados para domínio jurídico"""
    
    CIVIL_LAW = """
    Como especialista em Direito Civil, escreva um artigo jurídico detalhado que responda à seguinte questão:
    
    Questão: {query}
    
    O artigo deve incluir:
    - Fundamentação legal relevante
    - Interpretação doutrinária
    - Precedentes jurisprudenciais aplicáveis
    - Análise prática da questão
    
    Artigo:"""
    
    CRIMINAL_LAW = """
    Como especialista em Direito Penal, elabore um parecer jurídico fundamentado sobre:
    
    Questão: {query}
    
    O parecer deve abordar:
    - Tipificação penal aplicável
    - Elementos do tipo penal
    - Excludentes de ilicitude ou culpabilidade
    - Jurisprudência dos tribunais superiores
    
    Parecer:"""
    
    CONSTITUTIONAL_LAW = """
    Como constitucionalista, redija uma análise jurídica sobre:
    
    Questão: {query}
    
    A análise deve contemplar:
    - Fundamentos constitucionais
    - Princípios constitucionais envolvidos
    - Controle de constitucionalidade
    - Decisões do STF relevantes
    
    Análise:"""
    
    ADMINISTRATIVE_LAW = """
    Como especialista em Direito Administrativo, prepare um estudo jurídico sobre:
    
    Questão: {query}
    
    O estudo deve incluir:
    - Princípios da administração pública
    - Competências administrativas
    - Processo administrativo
    - Controle judicial dos atos administrativos
    
    Estudo:"""
    
    GENERIC_LEGAL = """
    Como jurista experiente, redija um documento jurídico completo que responda à seguinte questão legal:
    
    Questão: {query}
    
    O documento deve ser fundamentado em:
    - Legislação aplicável
    - Doutrina especializada
    - Jurisprudência consolidada
    - Aspectos práticos relevantes
    
    Documento:"""

class HyDERetriever:
    """
    Retriever HyDE especializado para documentos jurídicos
    
    Funcionalidades:
    - Geração de documentos hipotéticos usando múltiplas estratégias
    - Classificação automática do domínio jurídico
    - Recuperação híbrida (HyDE + busca direta)
    - Reranking inteligente de resultados
    - Métricas de qualidade e confiança
    - Otimização para casos jurídicos brasileiros
    """
    
    def __init__(
        self,
        rag_pipeline: JurisRAGPipeline,
        llm_model: str = "microsoft/DialoGPT-medium",
        classifier_model: str = "neuralmind/bert-base-portuguese-cased",
        max_hypothetical_docs: int = 3,
        generation_temperature: float = 0.8,
        max_generation_tokens: int = 512,
        device: str = "auto"
    ):
        """
        Inicializa o HyDE Retriever
        
        Args:
            rag_pipeline: Pipeline RAG base
            llm_model: Modelo para geração de documentos hipotéticos
            classifier_model: Modelo para classificação de domínio jurídico
            max_hypothetical_docs: Número máximo de documentos hipotéticos
            generation_temperature: Temperatura para geração
            max_generation_tokens: Tokens máximos por geração
            device: Dispositivo (cpu/cuda/auto)
        """
        self.rag_pipeline = rag_pipeline
        self.llm_model_name = llm_model
        self.classifier_model_name = classifier_model
        self.max_hypothetical_docs = max_hypothetical_docs
        self.generation_temperature = generation_temperature
        self.max_generation_tokens = max_generation_tokens
        
        # Detecta dispositivo
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Inicializando HyDE Retriever com dispositivo: {self.device}")
        
        # Inicializa componentes
        self._initialize_llm()
        self._initialize_classifier()
        self._initialize_templates()
        
        # Métricas
        self.total_queries = 0
        self.total_generations = 0
        self.total_tokens_generated = 0
        self.performance_stats = defaultdict(list)
        
    def _initialize_llm(self):
        """Inicializa modelo de linguagem para geração"""
        try:
            logger.info(f"Carregando LLM para HyDE: {self.llm_model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            # Configura tokens especiais
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Configura para geração
            self.generation_config = {
                "temperature": self.generation_temperature,
                "max_new_tokens": self.max_generation_tokens,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "num_return_sequences": 1
            }
            
            logger.info("LLM para HyDE carregado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao carregar LLM para HyDE: {e}")
            # Fallback para pipeline
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.llm_model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else None
            )
    
    def _initialize_classifier(self):
        """Inicializa classificador de domínio jurídico"""
        try:
            logger.info("Inicializando classificador de domínio jurídico")
            
            # Para simplificar, vamos usar um classificador baseado em keywords
            # Em produção, seria melhor treinar um modelo específico
            self.domain_keywords = {
                "civil": [
                    "contrato", "responsabilidade civil", "danos", "indenização",
                    "propriedade", "posse", "família", "sucessões", "obrigações",
                    "pessoa jurídica", "pessoa física", "direitos reais"
                ],
                "criminal": [
                    "crime", "delito", "pena", "prisão", "homicídio", "furto",
                    "roubo", "estelionato", "tráfico", "penal", "criminal",
                    "processo penal", "investigação", "inquérito"
                ],
                "constitutional": [
                    "constituição", "constitucional", "direitos fundamentais",
                    "supremo tribunal federal", "STF", "controle de constitucionalidade",
                    "princípios constitucionais", "direitos humanos", "federação"
                ],
                "administrative": [
                    "administração pública", "servidor público", "licitação",
                    "concessão", "permissão", "ato administrativo", "processo administrativo",
                    "improbidade", "responsabilidade administrativa"
                ]
            }
            
            logger.info("Classificador de domínio inicializado")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar classificador: {e}")
            self.domain_keywords = {}
    
    def _initialize_templates(self):
        """Inicializa templates de prompts"""
        self.templates = JuridicalPromptTemplates()
        
        self.template_mapping = {
            "civil": self.templates.CIVIL_LAW,
            "criminal": self.templates.CRIMINAL_LAW,
            "constitutional": self.templates.CONSTITUTIONAL_LAW,
            "administrative": self.templates.ADMINISTRATIVE_LAW,
            "generic": self.templates.GENERIC_LEGAL
        }
    
    def _classify_legal_domain(self, query: str) -> str:
        """
        Classifica o domínio jurídico da query
        
        Args:
            query: Query para classificar
            
        Returns:
            Domínio identificado ('civil', 'criminal', etc.)
        """
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            logger.info(f"Domínio identificado: {best_domain} (score: {domain_scores[best_domain]})")
            return best_domain
        
        logger.info("Domínio não identificado, usando template genérico")
        return "generic"
    
    def _generate_hypothetical_document(
        self,
        query: str,
        template: str,
        method: str = "domain_specific"
    ) -> HypotheticalDocument:
        """
        Gera um documento hipotético baseado na query
        
        Args:
            query: Query original
            template: Template de prompt a usar
            method: Método de geração
            
        Returns:
            Documento hipotético gerado
        """
        start_time = datetime.now()
        
        try:
            # Prepara prompt
            prompt = template.format(query=query)
            
            # Tokeniza
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding=True
            )
            
            if self.device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Gera
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    **self.generation_config
                )
            
            # Decodifica
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Remove prompt da resposta
            if template.split("{query}")[0].strip() in generated_text:
                parts = generated_text.split(template.split("{query}")[0].strip())
                if len(parts) > 1:
                    generated_text = parts[-1].strip()
            
            # Limpa texto
            generated_text = self._clean_generated_text(generated_text)
            
            # Calcula métricas
            generation_time = (datetime.now() - start_time).total_seconds()
            tokens_used = len(outputs[0]) - len(inputs['input_ids'][0])
            confidence = self._calculate_generation_confidence(generated_text, query)
            
            self.total_generations += 1
            self.total_tokens_generated += tokens_used
            
            return HypotheticalDocument(
                query=query,
                content=generated_text,
                generation_method=method,
                confidence=confidence,
                tokens_used=tokens_used,
                generation_time=generation_time
            )
            
        except Exception as e:
            logger.error(f"Erro na geração de documento hipotético: {e}")
            return HypotheticalDocument(
                query=query,
                content=f"Documento jurídico sobre: {query}",
                generation_method=f"{method}_fallback",
                confidence=0.1,
                tokens_used=0,
                generation_time=0.0
            )
    
    def _clean_generated_text(self, text: str) -> str:
        """Limpa e formata texto gerado"""
        # Remove quebras de linha excessivas
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove espaços extras
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove início do prompt se ainda estiver presente
        if text.startswith("Questão:") or text.startswith("Como "):
            lines = text.split('\n')
            # Encontra onde começa o conteúdo real
            for i, line in enumerate(lines):
                if any(starter in line.lower() for starter in 
                      ['artigo:', 'parecer:', 'análise:', 'estudo:', 'documento:']):
                    text = '\n'.join(lines[i+1:])
                    break
        
        return text.strip()
    
    def _calculate_generation_confidence(self, generated_text: str, query: str) -> float:
        """
        Calcula confiança do documento gerado
        
        Args:
            generated_text: Texto gerado
            query: Query original
            
        Returns:
            Score de confiança (0-1)
        """
        if not generated_text or len(generated_text) < 50:
            return 0.1
        
        # Métricas de qualidade
        length_score = min(len(generated_text) / 500, 1.0)  # Textos maiores são melhores
        
        # Verifica se contém termos jurídicos
        legal_terms = [
            "lei", "artigo", "código", "jurisprudência", "tribunal",
            "direito", "legal", "normativo", "constitucional"
        ]
        legal_term_count = sum(1 for term in legal_terms 
                              if term in generated_text.lower())
        legal_score = min(legal_term_count / 5, 1.0)
        
        # Verifica estrutura (parágrafos, pontuação)
        structure_score = 0.5
        if '\n' in generated_text:
            structure_score += 0.2
        if any(punct in generated_text for punct in ['.', ':', ';']):
            structure_score += 0.3
        structure_score = min(structure_score, 1.0)
        
        # Combinação das métricas
        confidence = (length_score * 0.4 + legal_score * 0.4 + structure_score * 0.2)
        
        return max(0.1, min(1.0, confidence))
    
    def generate_multiple_hypothetical_documents(
        self,
        query: str,
        num_docs: Optional[int] = None
    ) -> List[HypotheticalDocument]:
        """
        Gera múltiplos documentos hipotéticos usando diferentes estratégias
        
        Args:
            query: Query original
            num_docs: Número de documentos (padrão: self.max_hypothetical_docs)
            
        Returns:
            Lista de documentos hipotéticos
        """
        if num_docs is None:
            num_docs = self.max_hypothetical_docs
            
        # Classifica domínio
        domain = self._classify_legal_domain(query)
        
        documents = []
        
        # Estratégia 1: Template específico do domínio
        if domain in self.template_mapping:
            doc1 = self._generate_hypothetical_document(
                query,
                self.template_mapping[domain],
                f"domain_specific_{domain}"
            )
            documents.append(doc1)
        
        # Estratégia 2: Template genérico (sempre inclui)
        if len(documents) < num_docs:
            doc2 = self._generate_hypothetical_document(
                query,
                self.template_mapping["generic"],
                "generic_legal"
            )
            documents.append(doc2)
        
        # Estratégia 3: Variação com temperatura diferente
        if len(documents) < num_docs:
            original_temp = self.generation_config["temperature"]
            self.generation_config["temperature"] = 0.5  # Mais conservador
            
            doc3 = self._generate_hypothetical_document(
                query,
                self.template_mapping.get(domain, self.template_mapping["generic"]),
                "low_temperature"
            )
            documents.append(doc3)
            
            # Restaura temperatura
            self.generation_config["temperature"] = original_temp
        
        return documents[:num_docs]
    
    def retrieve_with_hyde(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.1,
        strategy: str = "hybrid"  # "hyde", "direct", "hybrid"
    ) -> List[HyDEResult]:
        """
        Recuperação usando HyDE
        
        Args:
            query: Query original
            k: Número de resultados
            score_threshold: Score mínimo
            strategy: Estratégia de recuperação
            
        Returns:
            Lista de resultados HyDE
        """
        logger.info(f"Executando HyDE retrieval com estratégia: {strategy}")
        
        results_dict = {}  # chunk_id -> HyDEResult
        
        # Gera documentos hipotéticos
        hypothetical_docs = []
        if strategy in ["hyde", "hybrid"]:
            hypothetical_docs = self.generate_multiple_hypothetical_documents(query)
        
        # Recuperação com documentos hipotéticos
        hyde_results = []
        if hypothetical_docs:
            for hyp_doc in hypothetical_docs:
                hyde_retrieved = self.rag_pipeline.retrieve(
                    hyp_doc.content, k, score_threshold
                )
                hyde_results.extend(hyde_retrieved)
        
        # Recuperação direta
        direct_results = []
        if strategy in ["direct", "hybrid"]:
            direct_results = self.rag_pipeline.retrieve(query, k, score_threshold)
        
        # Combina resultados
        for result in hyde_results:
            chunk_id = result.chunk.id
            if chunk_id not in results_dict:
                results_dict[chunk_id] = HyDEResult(
                    original_result=result,
                    hyde_score=result.score,
                    direct_score=0.0,
                    combined_score=result.score,
                    retrieval_method="hyde",
                    hypothetical_docs=hypothetical_docs
                )
            else:
                # Atualiza score HyDE se for melhor
                if result.score > results_dict[chunk_id].hyde_score:
                    results_dict[chunk_id].hyde_score = result.score
        
        for result in direct_results:
            chunk_id = result.chunk.id
            if chunk_id not in results_dict:
                results_dict[chunk_id] = HyDEResult(
                    original_result=result,
                    hyde_score=0.0,
                    direct_score=result.score,
                    combined_score=result.score,
                    retrieval_method="direct",
                    hypothetical_docs=hypothetical_docs
                )
            else:
                # Atualiza score direto
                results_dict[chunk_id].direct_score = result.score
                results_dict[chunk_id].retrieval_method = "hybrid"
        
        # Calcula scores combinados e reordena
        final_results = list(results_dict.values())
        
        for result in final_results:
            # Combina scores com pesos
            if strategy == "hybrid":
                result.combined_score = (
                    result.hyde_score * 0.7 + 
                    result.direct_score * 0.3
                )
            elif strategy == "hyde":
                result.combined_score = result.hyde_score
            else:  # direct
                result.combined_score = result.direct_score
        
        # Ordena por score combinado
        final_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return final_results[:k]
    
    def query(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.1,
        strategy: str = "hybrid",
        generate_answer: bool = True
    ) -> HyDEResponse:
        """
        Executa query completa usando HyDE
        
        Args:
            query: Pergunta
            k: Número de resultados
            score_threshold: Score mínimo
            strategy: Estratégia ("hyde", "direct", "hybrid")
            generate_answer: Se deve gerar resposta
            
        Returns:
            Resposta completa HyDE
        """
        start_time = datetime.now()
        self.total_queries += 1
        
        # Recupera com HyDE
        hyde_results = self.retrieve_with_hyde(query, k, score_threshold, strategy)
        
        # Extrai documentos hipotéticos
        hypothetical_docs = []
        if hyde_results:
            hypothetical_docs = hyde_results[0].hypothetical_docs
        
        # Gera resposta se solicitado
        answer = ""
        total_tokens = sum(doc.tokens_used for doc in hypothetical_docs)
        
        if generate_answer and hyde_results:
            # Usa chunks dos melhores resultados
            chunks = [r.original_result.chunk for r in hyde_results]
            answer = self.rag_pipeline._generate_answer(query, chunks)
        elif generate_answer:
            answer = "Não foram encontrados documentos relevantes para sua pergunta."
        
        # Calcula confiança
        confidence = self._calculate_hyde_confidence(hyde_results, hypothetical_docs)
        
        # Tempo de processamento
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Atualiza estatísticas
        self.performance_stats['processing_time'].append(processing_time)
        self.performance_stats['confidence'].append(confidence)
        self.performance_stats['num_results'].append(len(hyde_results))
        
        return HyDEResponse(
            query=query,
            answer=answer,
            results=hyde_results,
            hypothetical_documents=hypothetical_docs,
            confidence=confidence,
            processing_time=processing_time,
            total_tokens_used=total_tokens,
            strategy_used=strategy,
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_hyde_confidence(
        self,
        results: List[HyDEResult],
        hypothetical_docs: List[HypotheticalDocument]
    ) -> float:
        """
        Calcula confiança da resposta HyDE
        
        Args:
            results: Resultados HyDE
            hypothetical_docs: Documentos hipotéticos
            
        Returns:
            Score de confiança (0-1)
        """
        if not results:
            return 0.0
        
        # Score baseado na qualidade dos resultados
        result_confidence = min(results[0].combined_score / 1.0, 1.0)
        
        # Score baseado na qualidade dos documentos hipotéticos
        hyp_confidence = 0.0
        if hypothetical_docs:
            hyp_confidence = np.mean([doc.confidence for doc in hypothetical_docs])
        
        # Score baseado na consistência dos resultados
        if len(results) > 1:
            scores = [r.combined_score for r in results]
            consistency = 1.0 - min(np.std(scores), 0.3) / 0.3
        else:
            consistency = 0.5
        
        # Combina métricas
        final_confidence = (
            result_confidence * 0.5 +
            hyp_confidence * 0.3 +
            consistency * 0.2
        )
        
        return max(0.0, min(1.0, final_confidence))
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance"""
        stats = {
            "total_queries": self.total_queries,
            "total_generations": self.total_generations,
            "total_tokens_generated": self.total_tokens_generated,
            "avg_tokens_per_generation": (
                self.total_tokens_generated / max(self.total_generations, 1)
            ),
            "llm_model": self.llm_model_name,
            "device": self.device
        }
        
        # Estatísticas de performance
        if self.performance_stats['processing_time']:
            stats.update({
                "avg_processing_time": np.mean(self.performance_stats['processing_time']),
                "avg_confidence": np.mean(self.performance_stats['confidence']),
                "avg_results_per_query": np.mean(self.performance_stats['num_results'])
            })
        
        return stats
    
    def clear_statistics(self):
        """Limpa estatísticas coletadas"""
        self.performance_stats = defaultdict(list)
        self.total_queries = 0
        self.total_generations = 0
        self.total_tokens_generated = 0