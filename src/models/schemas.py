"""
Modelos de dados (Schemas) do JurisOracle
Definições Pydantic para estruturas de dados, validação e serialização
"""

import re
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Union, Literal
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator, EmailStr
from pydantic.types import SecretStr, constr, conint, confloat
from pydantic.networks import AnyHttpUrl

# ==========================================
# ENUMS E CONSTANTES
# ==========================================

class DocumentType(str, Enum):
    """Tipos de documentos jurídicos"""
    SENTENCA = "sentenca"
    ACORDAO = "acordao"
    PETICAO = "peticao"
    CONTRATO = "contrato"
    LEI = "lei"
    DECRETO = "decreto"
    PARECER = "parecer"
    DESPACHO = "despacho"
    DECISAO = "decisao"
    OUTROS = "outros"

class DocumentStatus(str, Enum):
    """Status de processamento de documentos"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"

class ChunkType(str, Enum):
    """Tipos de chunks de texto"""
    PARAGRAPH = "paragraph"
    HEADER = "header"
    TABLE = "table"
    LIST = "list"
    QUOTE = "quote"
    LEGAL_CITATION = "legal_citation"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    METADATA = "metadata"

class QueryStrategy(str, Enum):
    """Estratégias de query"""
    RAG_BASIC = "rag_basic"
    HYDE = "hyde"
    HYBRID = "hybrid"
    DIRECT = "direct"

class UserRole(str, Enum):
    """Tipos de usuários"""
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"
    VIEWER = "viewer"

class CourtType(str, Enum):
    """Tipos de tribunais"""
    STF = "stf"  # Supremo Tribunal Federal
    STJ = "stj"  # Superior Tribunal de Justiça
    TST = "tst"  # Tribunal Superior do Trabalho
    TSE = "tse"  # Tribunal Superior Eleitoral
    STM = "stm"  # Superior Tribunal Militar
    TRF = "trf"  # Tribunal Regional Federal
    TJ = "tj"    # Tribunal de Justiça
    TRT = "trt"  # Tribunal Regional do Trabalho
    TRE = "tre"  # Tribunal Regional Eleitoral
    OUTROS = "outros"

# ==========================================
# MODELOS BASE
# ==========================================

class BaseSchema(BaseModel):
    """Schema base com configurações comuns"""
    
    class Config:
        """Configuração Pydantic"""
        use_enum_values = True
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }

class TimestampMixin(BaseModel):
    """Mixin para timestamps"""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    @validator('updated_at', pre=True, always=True)
    def set_updated_at(cls, v):
        return v or datetime.now()

class UUIDMixin(BaseModel):
    """Mixin para UUIDs"""
    id: UUID = Field(default_factory=uuid4)

# ==========================================
# MODELOS DE USUÁRIO E AUTENTICAÇÃO
# ==========================================

class UserBase(BaseSchema):
    """Modelo base de usuário"""
    email: EmailStr
    full_name: constr(min_length=2, max_length=100)
    role: UserRole = UserRole.USER
    is_active: bool = True
    
    @validator('full_name')
    def validate_full_name(cls, v):
        if not v.strip():
            raise ValueError('Nome não pode estar vazio')
        return v.strip()

class UserCreate(UserBase):
    """Modelo para criação de usuário"""
    password: SecretStr = Field(..., min_length=8)
    confirm_password: str
    
    @validator('password')
    def validate_password(cls, v):
        password = v.get_secret_value()
        
        # Verifica comprimento mínimo
        if len(password) < 8:
            raise ValueError('Senha deve ter pelo menos 8 caracteres')
        
        # Verifica se tem pelo menos uma letra maiúscula
        if not re.search(r'[A-Z]', password):
            raise ValueError('Senha deve ter pelo menos uma letra maiúscula')
        
        # Verifica se tem pelo menos uma letra minúscula
        if not re.search(r'[a-z]', password):
            raise ValueError('Senha deve ter pelo menos uma letra minúscula')
        
        # Verifica se tem pelo menos um número
        if not re.search(r'\d', password):
            raise ValueError('Senha deve ter pelo menos um número')
        
        # Verifica se tem pelo menos um caractere especial
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise ValueError('Senha deve ter pelo menos um caractere especial')
        
        return v
    
    @root_validator
    def validate_passwords_match(cls, values):
        password = values.get('password')
        confirm_password = values.get('confirm_password')
        
        if password and confirm_password:
            if password.get_secret_value() != confirm_password:
                raise ValueError('Senhas não coincidem')
        
        return values

class UserUpdate(BaseSchema):
    """Modelo para atualização de usuário"""
    full_name: Optional[constr(min_length=2, max_length=100)] = None
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

class UserResponse(UserBase, UUIDMixin, TimestampMixin):
    """Modelo de resposta de usuário"""
    last_login: Optional[datetime] = None
    documents_count: int = 0
    queries_count: int = 0

class UserLogin(BaseSchema):
    """Modelo para login"""
    email: EmailStr
    password: SecretStr
    remember_me: bool = False

class Token(BaseSchema):
    """Modelo de token JWT"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None

class TokenPayload(BaseSchema):
    """Payload do token JWT"""
    sub: str  # user_id
    exp: datetime
    iat: datetime
    role: UserRole
    email: str

# ==========================================
# MODELOS DE DOCUMENTOS
# ==========================================

class DocumentMetadataBase(BaseSchema):
    """Metadados base de documento"""
    filename: str
    file_size: conint(ge=0)
    file_type: str
    mime_type: str
    pages: conint(ge=0) = 0
    word_count: conint(ge=0) = 0
    character_count: conint(ge=0) = 0
    language: str = "portuguese"
    
    @validator('filename')
    def validate_filename(cls, v):
        if not v.strip():
            raise ValueError('Nome do arquivo não pode estar vazio')
        
        # Remove caracteres perigosos
        safe_filename = re.sub(r'[<>:"/\|?*]', '_', v.strip())
        return safe_filename

class LegalMetadata(BaseSchema):
    """Metadados específicos jurídicos"""
    document_type: Optional[DocumentType] = None
    court: Optional[CourtType] = None
    case_number: Optional[str] = None
    parties: List[str] = Field(default_factory=list)
    legal_topics: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    jurisdiction: Optional[str] = None
    decision_date: Optional[date] = None
    
    @validator('case_number')
    def validate_case_number(cls, v):
        if v and not re.match(r'^\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}$', v):
            # Formato CNJ padrão: NNNNNNN-DD.AAAA.J.TR.OOOO
            raise ValueError('Número do processo deve seguir o padrão CNJ')
        return v

class DocumentMetadata(DocumentMetadataBase, LegalMetadata, TimestampMixin):
    """Metadados completos de documento"""
    hash_md5: str
    hash_sha256: str
    extracted_at: datetime = Field(default_factory=datetime.now)
    processing_version: str = "1.0.0"

class DocumentChunk(BaseSchema, UUIDMixin):
    """Chunk de documento"""
    content: constr(min_length=1, max_length=10000)
    start_page: conint(ge=1)
    end_page: conint(ge=1)
    start_position: conint(ge=0)
    end_position: conint(ge=0)
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    chunk_index: conint(ge=0)
    
    # Formatação e estilo
    formatting: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadados específicos do chunk
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Relacionamentos
    parent_document_id: UUID
    parent_section: Optional[str] = None
    
    # NLP enriquecimento
    entities: List[Dict[str, str]] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    sentiment: Optional[str] = None
    
    @validator('end_page')
    def validate_page_order(cls, v, values):
        start_page = values.get('start_page')
        if start_page and v < start_page:
            raise ValueError('Página final deve ser >= página inicial')
        return v
    
    @validator('end_position')
    def validate_position_order(cls, v, values):
        start_position = values.get('start_position')
        if start_position is not None and v < start_position:
            raise ValueError('Posição final deve ser >= posição inicial')
        return v

class ProcessedDocument(BaseSchema, UUIDMixin, TimestampMixin):
    """Documento processado completo"""
    metadata: DocumentMetadata
    chunks: List[DocumentChunk]
    processing_stats: Dict[str, Any] = Field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.COMPLETED
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Estrutura do documento
    has_table_of_contents: bool = False
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    images: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Relacionamentos
    user_id: UUID
    upload_session_id: Optional[str] = None

# ==========================================
# MODELOS DE QUERY E RETRIEVAL
# ==========================================

class QueryBase(BaseSchema):
    """Base para queries"""
    query: constr(min_length=3, max_length=2000)
    k: conint(ge=1, le=50) = 5
    score_threshold: confloat(ge=0.0, le=1.0) = 0.1
    
    @validator('query')
    def validate_query(cls, v):
        # Remove espaços extras e valida conteúdo
        cleaned = ' '.join(v.split())
        if not cleaned:
            raise ValueError('Query não pode estar vazia')
        return cleaned

class RAGQueryRequest(QueryBase):
    """Request para query RAG"""
    generate_answer: bool = True
    include_metadata: bool = True
    max_answer_length: conint(ge=50, le=2000) = 500

class HyDEQueryRequest(QueryBase):
    """Request para query HyDE"""
    strategy: QueryStrategy = QueryStrategy.HYBRID
    generate_answer: bool = True
    num_hypothetical_docs: conint(ge=1, le=5) = 3
    include_hypothetical_docs: bool = False

class RetrievalResult(BaseSchema):
    """Resultado de recuperação"""
    chunk_id: UUID
    content: str
    score: confloat(ge=0.0, le=1.0)
    rank: conint(ge=0)
    source_document_id: UUID
    page: conint(ge=1)
    chunk_type: ChunkType
    
    # Metadados adicionais
    source_filename: str
    document_type: Optional[DocumentType] = None
    highlight_positions: List[Dict[str, int]] = Field(default_factory=list)

class HyDEResult(RetrievalResult):
    """Resultado HyDE com scores específicos"""
    hyde_score: confloat(ge=0.0, le=1.0)
    direct_score: confloat(ge=0.0, le=1.0)
    combined_score: confloat(ge=0.0, le=1.0)
    retrieval_method: Literal["hyde", "direct", "hybrid"]

class HypotheticalDocument(BaseSchema, UUIDMixin):
    """Documento hipotético gerado pelo HyDE"""
    query: str
    content: str
    generation_method: str
    confidence: confloat(ge=0.0, le=1.0)
    tokens_used: conint(ge=0)
    generation_time: confloat(ge=0.0)
    model_used: str
    created_at: datetime = Field(default_factory=datetime.now)

class QueryResponse(BaseSchema, UUIDMixin):
    """Resposta base para queries"""
    query: str
    answer: str
    sources: List[RetrievalResult]
    confidence: confloat(ge=0.0, le=1.0)
    processing_time: confloat(ge=0.0)
    strategy_used: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Metadados da query
    total_documents_searched: conint(ge=0) = 0
    model_info: Dict[str, str] = Field(default_factory=dict)

class HyDEQueryResponse(QueryResponse):
    """Resposta específica para HyDE"""
    sources: List[HyDEResult]  # Override com tipo específico
    hypothetical_documents: List[HypotheticalDocument] = Field(default_factory=list)
    total_tokens_used: conint(ge=0) = 0

# ==========================================
# MODELOS DE UPLOAD E PROCESSAMENTO
# ==========================================

class DocumentUploadRequest(BaseSchema):
    """Request de upload de documento"""
    extract_tables: bool = True
    extract_images: bool = False
    enable_ocr: bool = False
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Configurações de processamento
    chunk_size: Optional[conint(ge=100, le=4000)] = None
    chunk_overlap: Optional[conint(ge=0, le=500)] = None
    
    # Classificação manual (opcional)
    document_type: Optional[DocumentType] = None
    court: Optional[CourtType] = None
    case_number: Optional[str] = None

class DocumentUploadResponse(BaseSchema, UUIDMixin):
    """Resposta de upload de documento"""
    filename: str
    file_size: int
    processing_status: DocumentStatus
    chunks_created: conint(ge=0)
    processing_time: confloat(ge=0.0)
    
    # Metadados extraídos
    extracted_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Estatísticas
    pages_processed: conint(ge=0) = 0
    tables_found: conint(ge=0) = 0
    images_found: conint(ge=0) = 0
    
    # Avisos e erros
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    
    # URLs para acesso
    download_url: Optional[AnyHttpUrl] = None
    preview_url: Optional[AnyHttpUrl] = None

class BatchUploadRequest(BaseSchema):
    """Request para upload em lote"""
    files: List[str]  # Lista de paths ou URLs
    common_metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_options: DocumentUploadRequest = Field(default_factory=DocumentUploadRequest)
    
    # Configurações de batch
    max_parallel_jobs: conint(ge=1, le=10) = 3
    stop_on_error: bool = False

class BatchUploadResponse(BaseSchema, UUIDMixin):
    """Resposta de upload em lote"""
    total_files: conint(ge=0)
    successful_uploads: conint(ge=0)
    failed_uploads: conint(ge=0)
    processing_time: confloat(ge=0.0)
    
    # Detalhes dos uploads
    upload_results: List[DocumentUploadResponse]
    batch_errors: List[str] = Field(default_factory=list)
    
    # Estatísticas consolidadas
    total_chunks_created: conint(ge=0) = 0
    total_pages_processed: conint(ge=0) = 0

# ==========================================
# MODELOS DE SISTEMA E MONITORAMENTO
# ==========================================

class SystemHealth(BaseSchema):
    """Status de saúde do sistema"""
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str
    uptime: str
    
    # Status dos componentes
    components: Dict[str, bool] = Field(default_factory=dict)
    
    # Métricas básicas
    memory_usage: Dict[str, Union[int, float]] = Field(default_factory=dict)
    disk_usage: Dict[str, Union[int, float]] = Field(default_factory=dict)
    
    # Dependências externas
    database_status: bool = True
    redis_status: bool = True
    model_status: Dict[str, bool] = Field(default_factory=dict)

class SystemMetrics(BaseSchema):
    """Métricas detalhadas do sistema"""
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Métricas de aplicação
    total_documents: conint(ge=0)
    total_chunks: conint(ge=0)
    total_users: conint(ge=0)
    active_sessions: conint(ge=0)
    
    # Métricas de performance
    avg_query_time: confloat(ge=0.0)
    avg_upload_time: confloat(ge=0.0)
    requests_per_minute: confloat(ge=0.0)
    
    # Métricas de ML
    model_load_time: confloat(ge=0.0)
    avg_embedding_time: confloat(ge=0.0)
    avg_generation_time: confloat(ge=0.0)
    
    # Cache e storage
    cache_hit_rate: confloat(ge=0.0, le=1.0)
    vector_store_size: conint(ge=0)
    storage_used: conint(ge=0)  # em bytes

class ErrorReport(BaseSchema, UUIDMixin, TimestampMixin):
    """Relatório de erro"""
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    
    # Contexto do erro
    user_id: Optional[UUID] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    
    # Metadados do erro
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    resolved: bool = False
    resolution_notes: Optional[str] = None

# ==========================================
# MODELOS DE CONFIGURAÇÃO
# ==========================================

class ModelConfig(BaseSchema):
    """Configuração de modelos ML"""
    embedding_model: str
    llm_model: str
    device: Literal["auto", "cpu", "cuda", "mps"]
    
    # Parâmetros de embedding
    embedding_dimension: conint(ge=1)
    max_sequence_length: conint(ge=1, le=8192) = 512
    
    # Parâmetros de geração
    max_generation_length: conint(ge=1, le=4096) = 512
    temperature: confloat(ge=0.1, le=2.0) = 0.7
    top_p: confloat(ge=0.1, le=1.0) = 0.9
    repetition_penalty: confloat(ge=1.0, le=2.0) = 1.1

class ProcessingConfig(BaseSchema):
    """Configuração de processamento"""
    chunk_size: conint(ge=100, le=4000) = 1000
    chunk_overlap: conint(ge=0, le=500) = 200
    max_workers: conint(ge=1, le=16) = 4
    
    # Configurações de extração
    extract_tables: bool = True
    extract_images: bool = False
    enable_ocr: bool = False
    enable_nlp: bool = True
    
    # Configurações de cache
    cache_enabled: bool = True
    cache_ttl: conint(ge=300, le=86400) = 3600  # segundos

class APIConfig(BaseSchema):
    """Configuração da API"""
    rate_limit_enabled: bool = True
    default_rate_limit: str = "100/hour"
    
    # Limites de requisição
    max_query_length: conint(ge=100, le=5000) = 2000
    max_file_size: conint(ge=1024, le=100*1024*1024) = 50*1024*1024  # bytes
    max_batch_size: conint(ge=1, le=100) = 10
    
    # Timeouts
    request_timeout: conint(ge=30, le=3600) = 300  # segundos
    upload_timeout: conint(ge=60, le=7200) = 1800  # segundos

# ==========================================
# MODELOS DE AUDITORIA E LOGS
# ==========================================

class AuditLog(BaseSchema, UUIDMixin, TimestampMixin):
    """Log de auditoria"""
    user_id: Optional[UUID] = None
    action: str
    resource_type: str
    resource_id: Optional[UUID] = None
    
    # Detalhes da ação
    details: Dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Resultado
    success: bool = True
    error_message: Optional[str] = None

class QueryLog(BaseSchema, UUIDMixin, TimestampMixin):
    """Log de queries"""
    user_id: UUID
    query_text: str
    strategy_used: QueryStrategy
    
    # Resultados
    results_count: conint(ge=0)
    confidence_score: confloat(ge=0.0, le=1.0)
    processing_time: confloat(ge=0.0)
    
    # Metadados
    model_used: str
    tokens_used: conint(ge=0) = 0
    cache_hit: bool = False

class AccessLog(BaseSchema, UUIDMixin):
    """Log de acesso"""
    timestamp: datetime = Field(default_factory=datetime.now)
    user_id: Optional[UUID] = None
    ip_address: str
    method: str
    endpoint: str
    status_code: conint(ge=100, le=599)
    response_time: confloat(ge=0.0)
    user_agent: Optional[str] = None
    referer: Optional[str] = None

# ==========================================
# MODELOS DE RESPOSTA DA API
# ==========================================

class PaginatedResponse(BaseSchema):
    """Resposta paginada genérica"""
    items: List[Any]
    total_count: conint(ge=0)
    page: conint(ge=1) = 1
    page_size: conint(ge=1, le=100) = 20
    total_pages: conint(ge=0)
    has_next: bool
    has_previous: bool

class DocumentListResponse(PaginatedResponse):
    """Lista paginada de documentos"""
    items: List[ProcessedDocument]

class QueryHistoryResponse(PaginatedResponse):
    """Histórico paginado de queries"""
    items: List[QueryLog]

class SuccessResponse(BaseSchema):
    """Resposta de sucesso genérica"""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseSchema):
    """Resposta de erro genérica"""
    success: bool = False
    error_code: str
    error_message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None

# ==========================================
# VALIDADORES CUSTOMIZADOS
# ==========================================

def validate_brazilian_cpf(cpf: str) -> str:
    """Valida CPF brasileiro"""
    # Remove caracteres não numéricos
    cpf = re.sub(r'[^0-9]', '', cpf)
    
    if len(cpf) != 11:
        raise ValueError('CPF deve ter 11 dígitos')
    
    # Verifica se todos os dígitos são iguais
    if cpf == cpf[0] * 11:
        raise ValueError('CPF inválido')
    
    # Calcula e verifica dígitos verificadores
    def calculate_digit(cpf_partial):
        sum_digits = sum(int(digit) * weight for digit, weight in zip(cpf_partial, range(len(cpf_partial) + 1, 1, -1)))
        remainder = sum_digits % 11
        return '0' if remainder < 2 else str(11 - remainder)
    
    if cpf[9] != calculate_digit(cpf[:9]) or cpf[10] != calculate_digit(cpf[:10]):
        raise ValueError('CPF inválido')
    
    return cpf

def validate_brazilian_cnpj(cnpj: str) -> str:
    """Valida CNPJ brasileiro"""
    # Remove caracteres não numéricos
    cnpj = re.sub(r'[^0-9]', '', cnpj)
    
    if len(cnpj) != 14:
        raise ValueError('CNPJ deve ter 14 dígitos')
    
    # Verifica se todos os dígitos são iguais
    if cnpj == cnpj[0] * 14:
        raise ValueError('CNPJ inválido')
    
    # Algoritmo de validação do CNPJ
    weights1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    weights2 = [6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9]
    
    def calculate_digit(cnpj_partial, weights):
        sum_digits = sum(int(digit) * weight for digit, weight in zip(cnpj_partial, weights))
        remainder = sum_digits % 11
        return '0' if remainder < 2 else str(11 - remainder)
    
    if (cnpj[12] != calculate_digit(cnpj[:12], weights1) or 
        cnpj[13] != calculate_digit(cnpj[:13], weights1 + [6])):
        raise ValueError('CNPJ inválido')
    
    return cnpj

# ==========================================
# SCHEMAS COM VALIDAÇÕES CUSTOMIZADAS
# ==========================================

class BrazilianPersonBase(BaseSchema):
    """Base para pessoa física brasileira"""
    cpf: Optional[str] = None
    
    @validator('cpf')
    def validate_cpf(cls, v):
        if v:
            return validate_brazilian_cpf(v)
        return v

class BrazilianCompanyBase(BaseSchema):
    """Base para pessoa jurídica brasileira"""
    cnpj: Optional[str] = None
    
    @validator('cnpj')
    def validate_cnpj(cls, v):
        if v:
            return validate_brazilian_cnpj(v)
        return v

class LegalEntitySchema(BaseSchema):
    """Entidade jurídica (pessoa física ou jurídica)"""
    name: constr(min_length=2, max_length=200)
    entity_type: Literal["person", "company", "government", "other"]
    cpf: Optional[str] = None
    cnpj: Optional[str] = None
    
    @root_validator
    def validate_documents(cls, values):
        entity_type = values.get('entity_type')
        cpf = values.get('cpf')
        cnpj = values.get('cnpj')
        
        if entity_type == 'person' and not cpf:
            raise ValueError('CPF é obrigatório para pessoa física')
        
        if entity_type == 'company' and not cnpj:
            raise ValueError('CNPJ é obrigatório para pessoa jurídica')
        
        if cpf:
            values['cpf'] = validate_brazilian_cpf(cpf)
        
        if cnpj:
            values['cnpj'] = validate_brazilian_cnpj(cnpj)
        
        return values

# ==========================================
# MODELOS DE EXPORTAÇÃO
# ==========================================

class ExportFormat(str, Enum):
    """Formatos de exportação"""
    JSON = "json"
    CSV = "csv"
    XLSX = "xlsx"
    PDF = "pdf"
    XML = "xml"

class ExportRequest(BaseSchema):
    """Request de exportação de dados"""
    format: ExportFormat
    include_metadata: bool = True
    date_range: Optional[Dict[str, date]] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    
    # Configurações específicas por formato
    csv_delimiter: str = ","
    excel_sheet_name: str = "Export"
    pdf_template: Optional[str] = None

class ExportResponse(BaseSchema, UUIDMixin):
    """Resposta de exportação"""
    filename: str
    format: ExportFormat
    file_size: conint(ge=0)
    records_count: conint(ge=0)
    download_url: AnyHttpUrl
    expires_at: datetime
    created_at: datetime = Field(default_factory=datetime.now)

# ==========================================
# MODELOS PARA MACHINE LEARNING
# ==========================================

class EmbeddingRequest(BaseSchema):
    """Request para gerar embeddings"""
    texts: List[constr(min_length=1, max_length=8192)]
    model: Optional[str] = None
    normalize: bool = True

class EmbeddingResponse(BaseSchema):
    """Resposta com embeddings"""
    embeddings: List[List[float]]
    model_used: str
    dimension: conint(ge=1)
    processing_time: confloat(ge=0.0)

class ModelBenchmark(BaseSchema):
    """Benchmark de modelo"""
    model_name: str
    task_type: Literal["embedding", "generation", "classification"]
    
    # Métricas de performance
    accuracy: Optional[confloat(ge=0.0, le=1.0)] = None
    precision: Optional[confloat(ge=0.0, le=1.0)] = None
    recall: Optional[confloat(ge=0.0, le=1.0)] = None
    f1_score: Optional[confloat(ge=0.0, le=1.0)] = None
    
    # Métricas de eficiência
    avg_inference_time: confloat(ge=0.0)
    memory_usage: conint(ge=0)  # MB
    
    # Configuração do teste
    test_dataset_size: conint(ge=1)
    test_date: datetime = Field(default_factory=datetime.now)

# ==========================================
# EXEMPLO DE USO E VALIDAÇÃO
# ==========================================

if __name__ == "__main__":
    # Exemplo de criação e validação de schemas
    
    # User creation
    try:
        user = UserCreate(
            email="teste@exemplo.com",
            full_name="João Silva",
            password=SecretStr("MinhaSenh@123"),
            confirm_password="MinhaSenh@123"
        )
        print("✅ Usuário válido:", user.email)
    except Exception as e:
        print("❌ Erro de validação:", e)
    
    # Document metadata
    try:
        doc_metadata = DocumentMetadata(
            filename="sentenca_001.pdf",
            file_size=1024000,
            file_type="PDF",
            mime_type="application/pdf",
            pages=10,
            word_count=5000,
            character_count=25000,
            hash_md5="abc123",
            hash_sha256="def456",
            document_type=DocumentType.SENTENCA,
            court=CourtType.TJ,
            case_number="1234567-89.2023.1.02.3456"
        )
        print("✅ Metadados válidos:", doc_metadata.filename)
    except Exception as e:
        print("❌ Erro de validação:", e)
    
    # Query request
    try:
        query = HyDEQueryRequest(
            query="Qual a jurisprudência sobre danos morais?",
            k=10,
            strategy=QueryStrategy.HYBRID
        )
        print("✅ Query válida:", query.query[:50] + "...")
    except Exception as e:
        print("❌ Erro de validação:", e)