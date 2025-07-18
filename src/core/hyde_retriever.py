import asyncio
import hashlib
import json
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import openai
from openai import AsyncOpenAI
import google.generativeai as genai
from anthropic import AsyncAnthropic
import tiktoken

from ..config.settings import Settings

logger = logging.getLogger(__name__)

@dataclass
class HypotheticDocument:
    """Represents a generated hypothetical document."""
    content: str
    query: str
    model_used: str
    created_at: datetime
    confidence: float = 0.0
    tokens_used: int = 0

@dataclass
class HyDEConfig:
    """Configuration for HyDE generation."""
    max_tokens: int = 512
    temperature: float = 0.3
    model: str = "gpt-3.5-turbo"
    cache_ttl_hours: int = 24
    enable_cache: bool = True
    max_hypotheses: int = 3
    legal_domain_focus: bool = True

class HyDEPromptTemplates:
    """Templates for generating legal hypothetical documents."""
    
    JURISPRUDENCE_TEMPLATE = """
    Como um juiz experiente do direito brasileiro, elabore uma decisão judicial hipotética 
    que responderia à seguinte consulta jurídica:
    
    Consulta: {query}
    
    Elabore uma decisão fundamentada que:
    - Cite artigos relevantes da legislação brasileira
    - Mencione princípios jurídicos aplicáveis
    - Use linguagem técnica jurídica apropriada
    - Seja objetiva e precisa
    
    Decisão Hipotética:
    """
    
    LEGAL_ARTICLE_TEMPLATE = """
    Como um jurista especialista, redija um artigo jurídico técnico que responderia 
    completamente à seguinte questão:
    
    Questão: {query}
    
    O artigo deve:
    - Abordar os aspectos legais pertinentes
    - Citar legislação e jurisprudência relevantes
    - Usar terminologia jurídica precisa
    - Ser academicamente rigoroso
    
    Artigo:
    """
    
    LEGAL_OPINION_TEMPLATE = """
    Como um advogado sênior especialista em direito brasileiro, elabore um parecer 
    jurídico técnico que responda à seguinte consulta:
    
    Consulta: {query}
    
    O parecer deve:
    - Analisar os aspectos jurídicos relevantes
    - Citar fundamentação legal apropriada
    - Apresentar conclusões técnicas
    - Seguir estrutura de parecer profissional
    
    Parecer:
    """
    
    LEGAL_SUMMARY_TEMPLATE = """
    Como um especialista em direito, elabore um resumo técnico e detalhado 
    sobre o seguinte tema jurídico:
    
    Tema: {query}
    
    O resumo deve:
    - Cobrir os principais aspectos legais
    - Incluir referências normativas
    - Ser tecnicamente preciso
    - Usar linguagem jurídica apropriada
    
    Resumo:
    """

class LLMProvider:
    """Base class for LLM providers."""
    
    async def generate(self, prompt: str, config: HyDEConfig) -> str:
        raise NotImplementedError

class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""
    
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    async def generate(self, prompt: str, config: HyDEConfig) -> Dict[str, Any]:
        """Generate hypothetical document using OpenAI."""
        try:
            response = await self.client.chat.completions.create(
                model=config.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Você é um especialista em direito brasileiro com amplo conhecimento em legislação, jurisprudência e doutrina. Elabore respostas técnicas, precisas e bem fundamentadas."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            content = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            return {
                "content": content,
                "tokens_used": tokens_used,
                "model": config.model
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar hipótese com OpenAI: {str(e)}")
            raise

class GeminiProvider(LLMProvider):
    """Google Gemini provider."""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    async def generate(self, prompt: str, config: HyDEConfig) -> Dict[str, Any]:
        """Generate hypothetical document using Gemini."""
        try:
            # Executar de forma síncrona em thread pool
            loop = asyncio.get_event_loop()
            
            def _generate():
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=config.max_tokens,
                        temperature=config.temperature,
                        top_p=0.9,
                        top_k=40
                    )
                )
                return response
            
            response = await loop.run_in_executor(None, _generate)
            content = response.text.strip()
            
            # Estimar tokens (aproximação)
            tokens_used = len(content.split()) * 1.3  # Aproximação
            
            return {
                "content": content,
                "tokens_used": int(tokens_used),
                "model": "gemini-pro"
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar hipótese com Gemini: {str(e)}")
            raise

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)
    
    async def generate(self, prompt: str, config: HyDEConfig) -> Dict[str, Any]:
        """Generate hypothetical document using Claude."""
        try:
            response = await self.client.messages.create(
                model=config.model if "claude" in config.model else "claude-3-sonnet-20240229",
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                system="Você é um especialista em direito brasileiro com amplo conhecimento em legislação, jurisprudência e doutrina. Elabore respostas técnicas, precisas e bem fundamentadas.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text.strip()
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            return {
                "content": content,
                "tokens_used": tokens_used,
                "model": config.model
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar hipótese com Claude: {str(e)}")
            raise

class HyDECache:
    """In-memory cache for HyDE results with TTL."""
    
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
    
    def _generate_key(self, query: str, config: HyDEConfig) -> str:
        """Generate cache key from query and config."""
        config_str = f"{config.model}_{config.temperature}_{config.max_tokens}"
        content = f"{query}_{config_str}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, query: str, config: HyDEConfig) -> Optional[List[HypotheticDocument]]:
        """Get cached hypothetical documents."""
        key = self._generate_key(query, config)
        
        if key in self._cache:
            cache_data = self._cache[key]
            
            # Verificar TTL
            if datetime.now() - cache_data["created_at"] < timedelta(hours=config.cache_ttl_hours):
                logger.info(f"Cache hit para query: {query[:50]}...")
                return cache_data["documents"]
            else:
                # Cache expirado
                del self._cache[key]
                logger.info(f"Cache expirado para query: {query[:50]}...")
        
        return None
    
    def set(self, query: str, config: HyDEConfig, documents: List[HypotheticDocument]):
        """Cache hypothetical documents."""
        key = self._generate_key(query, config)
        self._cache[key] = {
            "documents": documents,
            "created_at": datetime.now()
        }
        logger.info(f"Cache armazenado para query: {query[:50]}...")
    
    def clear(self):
        """Clear all cache."""
        self._cache.clear()
        logger.info("Cache HyDE limpo")
    
    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)

class HyDERegressor:
    """
    HyDE (Hypothetical Document Embeddings) implementation for legal documents.
    
    Generates hypothetical documents that would answer the user's query,
    improving semantic search by creating more specific targets for retrieval.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache = HyDECache()
        self.templates = HyDEPromptTemplates()
        
        # Inicializar providers
        self.providers = {}
        self._init_providers()
        
        # Configuração padrão
        self.default_config = HyDEConfig(
            max_tokens=512,
            temperature=0.3,
            model=settings.OPENAI_MODEL or "gpt-3.5-turbo",
            cache_ttl_hours=24,
            enable_cache=True,
            max_hypotheses=3,
            legal_domain_focus=True
        )
        
        logger.info("HyDERegressor inicializado")
    
    def _init_providers(self):
        """Initialize available LLM providers."""
        if self.settings.OPENAI_API_KEY:
            self.providers["openai"] = OpenAIProvider(self.settings.OPENAI_API_KEY)
            logger.info("OpenAI provider inicializado")
        
        if self.settings.GOOGLE_API_KEY:
            self.providers["gemini"] = GeminiProvider(self.settings.GOOGLE_API_KEY)
            logger.info("Gemini provider inicializado")
        
        if self.settings.ANTHROPIC_API_KEY:
            self.providers["anthropic"] = AnthropicProvider(self.settings.ANTHROPIC_API_KEY)
            logger.info("Anthropic provider inicializado")
        
        if not self.providers:
            raise ValueError("Nenhum provider de LLM configurado. Configure pelo menos uma API key.")
    
    def _detect_query_type(self, query: str) -> str:
        """Detect the type of legal query to choose appropriate template."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["jurisprudência", "decisão", "acórdão", "sentença", "tribunal"]):
            return "jurisprudence"
        elif any(word in query_lower for word in ["parecer", "consulta", "análise jurídica", "opinião"]):
            return "legal_opinion"
        elif any(word in query_lower for word in ["artigo", "doutrina", "estudo", "pesquisa"]):
            return "legal_article"
        elif any(word in query_lower for word in ["resumo", "conceito", "definição", "explicação"]):
            return "legal_summary"
        else:
            return "legal_opinion"  # Default
    
    def _select_template(self, query_type: str) -> str:
        """Select appropriate template based on query type."""
        template_map = {
            "jurisprudence": self.templates.JURISPRUDENCE_TEMPLATE,
            "legal_article": self.templates.LEGAL_ARTICLE_TEMPLATE,
            "legal_opinion": self.templates.LEGAL_OPINION_TEMPLATE,
            "legal_summary": self.templates.LEGAL_SUMMARY_TEMPLATE
        }
        return template_map.get(query_type, self.templates.LEGAL_OPINION_TEMPLATE)
    
    def _get_provider(self, model: str) -> LLMProvider:
        """Get appropriate provider based on model name."""
        if "gpt" in model.lower() or "openai" in model.lower():
            if "openai" not in self.providers:
                raise ValueError("OpenAI provider não configurado")
            return self.providers["openai"]
        elif "gemini" in model.lower() or "google" in model.lower():
            if "gemini" not in self.providers:
                raise ValueError("Gemini provider não configurado")
            return self.providers["gemini"]
        elif "claude" in model.lower() or "anthropic" in model.lower():
            if "anthropic" not in self.providers:
                raise ValueError("Anthropic provider não configurado")
            return self.providers["anthropic"]
        else:
            # Usar o primeiro provider disponível
            if self.providers:
                return list(self.providers.values())[0]
            raise ValueError("Nenhum provider disponível")
    
    async def generate_hypotheses(
        self, 
        query: str, 
        config: Optional[HyDEConfig] = None,
        force_regenerate: bool = False
    ) -> List[HypotheticDocument]:
        """
        Generate hypothetical documents for a given query.
        
        Args:
            query: The user's legal query
            config: Configuration for generation
            force_regenerate: Bypass cache and force regeneration
            
        Returns:
            List of hypothetical documents
        """
        if not query or not query.strip():
            raise ValueError("Query não pode estar vazia")
        
        config = config or self.default_config
        
        # Verificar cache primeiro
        if config.enable_cache and not force_regenerate:
            cached = self.cache.get(query, config)
            if cached:
                return cached
        
        try:
            # Detectar tipo de consulta
            query_type = self._detect_query_type(query)
            logger.info(f"Tipo de consulta detectado: {query_type}")
            
            # Selecionar template
            template = self._select_template(query_type)
            prompt = template.format(query=query)
            
            # Obter provider
            provider = self._get_provider(config.model)
            
            hypotheses = []
            
            # Gerar múltiplas hipóteses se configurado
            for i in range(min(config.max_hypotheses, 3)):
                logger.info(f"Gerando hipótese {i+1}/{config.max_hypotheses}")
                
                # Adicionar variação no prompt para hipóteses múltiplas
                varied_prompt = prompt
                if i > 0:
                    varied_prompt += f"\n\nVersão {i+1}: Forneça uma perspectiva alternativa ou complementar."
                
                # Gerar hipótese
                result = await provider.generate(varied_prompt, config)
                
                # Criar documento hipotético
                hypothesis = HypotheticDocument(
                    content=result["content"],
                    query=query,
                    model_used=result["model"],
                    created_at=datetime.now(),
                    confidence=0.8 - (i * 0.1),  # Primeira hipótese tem maior confiança
                    tokens_used=result["tokens_used"]
                )
                
                hypotheses.append(hypothesis)
                logger.info(f"Hipótese {i+1} gerada com {hypothesis.tokens_used} tokens")
            
            # Armazenar no cache
            if config.enable_cache:
                self.cache.set(query, config, hypotheses)
            
            logger.info(f"Geradas {len(hypotheses)} hipóteses para query: {query[:50]}...")
            return hypotheses
            
        except Exception as e:
            logger.error(f"Erro ao gerar hipóteses: {str(e)}")
            raise
    
    async def generate_single_hypothesis(
        self, 
        query: str, 
        config: Optional[HyDEConfig] = None
    ) -> HypotheticDocument:
        """Generate a single hypothetical document."""
        hypotheses = await self.generate_hypotheses(query, config)
        return hypotheses[0] if hypotheses else None
    
    def extract_hypothesis_content(self, hypotheses: List[HypotheticDocument]) -> List[str]:
        """Extract just the content from hypothetical documents."""
        return [h.content for h in hypotheses]
    
    def get_best_hypothesis(self, hypotheses: List[HypotheticDocument]) -> HypotheticDocument:
        """Get the hypothesis with highest confidence score."""
        if not hypotheses:
            return None
        return max(hypotheses, key=lambda h: h.confidence)
    
    def clear_cache(self):
        """Clear the HyDE cache."""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": self.cache.size(),
            "providers_available": list(self.providers.keys()),
            "default_model": self.default_config.model
        }
    
    async def evaluate_hypothesis_quality(self, hypothesis: HypotheticDocument, original_query: str) -> float:
        """
        Evaluate the quality of a generated hypothesis.
        
        This is a simple heuristic-based evaluation. In production,
        you might want to use more sophisticated methods.
        """
        content = hypothesis.content.lower()
        query_terms = original_query.lower().split()
        
        # Calcular relevância baseada em termos em comum
        term_overlap = sum(1 for term in query_terms if term in content)
        relevance_score = term_overlap / len(query_terms) if query_terms else 0
        
        # Penalizar respostas muito curtas ou muito longas
        length_penalty = 1.0
        if len(hypothesis.content) < 100:
            length_penalty = 0.7
        elif len(hypothesis.content) > 2000:
            length_penalty = 0.8
        
        # Verificar se contém linguagem jurídica apropriada
        legal_terms = ["lei", "artigo", "código", "jurisprudência", "tribunal", "direito", "legal"]
        legal_score = sum(1 for term in legal_terms if term in content) / len(legal_terms)
        
        # Pontuação final combinada
        final_score = (relevance_score * 0.4 + legal_score * 0.3 + length_penalty * 0.3)
        
        return min(1.0, final_score)

class HydeRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query, top_k=5):
        """
        Retrieve relevant documents based on the query.

        Args:
            query (str): The query string to search for.
            top_k (int): The number of top relevant documents to retrieve.

        Returns:
            list: A list of relevant documents.
        """
        # Convert the query into a vector
        query_vector = self.vector_store.embed_query(query)
        
        # Retrieve the top_k relevant documents
        relevant_documents = self.vector_store.get_similar_documents(query_vector, top_k)
        
        return relevant_documents