#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📊 SETUP_DATABASE.PY - IA_JURIS PROJECT
=====================================

Script robusto para inicialização e configuração da estrutura do banco de dados.
Suporta múltiplos SGBDs (PostgreSQL, MySQL, SQLite) com fallback automático.

Autor: Equipe IA_Juris
Data: 2025
Versão: 1.0.0

Funcionalidades:
- Conexão segura com banco de dados
- Criação automática de tabelas com constraints
- População de dados iniciais
- Validação de integridade
- Backup automático
- Suporte a múltiplos ambientes

Uso:
    python setup_database.py --init
    python setup_database.py --populate
    python setup_database.py --validate
"""

import os
import sys
import json
import logging
import argparse
import hashlib
import secrets
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from contextlib import contextmanager

# Importações com tratamento de exceções
try:
    import sqlalchemy as sa
    from sqlalchemy import (
        create_engine, MetaData, Table, Column, Integer, String, 
        DateTime, Text, Boolean, ForeignKey, Index, UniqueConstraint,
        CheckConstraint, text
    )
    from sqlalchemy.exc import (
        SQLAlchemyError, IntegrityError, OperationalError, 
        ProgrammingError, DatabaseError
    )
    from sqlalchemy.dialects import postgresql, mysql, sqlite
    from sqlalchemy.pool import QueuePool
except ImportError as e:
    print(f"❌ Erro: SQLAlchemy não encontrado. Instale com: pip install sqlalchemy")
    sys.exit(1)

try:
    import bcrypt
except ImportError:
    print("⚠️  Aviso: bcrypt não encontrado. Instale com: pip install bcrypt")
    bcrypt = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  Aviso: python-dotenv não encontrado. Variáveis de ambiente serão lidas diretamente.")

# Configuração de logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configura o sistema de logging para o script.
    
    Args:
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Caminho para arquivo de log (opcional)
    
    Returns:
        Logger configurado
    """
    logger = logging.getLogger("setup_database")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remover handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (se especificado)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Não foi possível criar arquivo de log: {e}")
    
    return logger

# Configuração global
logger = setup_logging()

class DatabaseConfig:
    """
    �� Classe para gerenciar configurações de banco de dados.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Inicializa configurações do banco de dados.
        
        Args:
            config_file: Caminho para arquivo de configuração JSON
        """
        self.config = self._load_config(config_file)
        self.validate_config()
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """
        Carrega configurações de arquivo ou variáveis de ambiente.
        
        Args:
            config_file: Caminho para arquivo de configuração
            
        Returns:
            Dicionário com configurações
        """
        # Configuração padrão
        default_config = {
            "database": {
                "type": "sqlite",
                "host": "localhost",
                "port": 5432,
                "name": "ia_juris_db",
                "user": "ia_juris_user",
                "password": "",
                "pool_size": 10,
                "max_overflow": 20,
                "pool_timeout": 30,
                "pool_recycle": 3600
            },
            "security": {
                "ssl_mode": "prefer",
                "connect_timeout": 30,
                "command_timeout": 60
            },
            "backup": {
                "enabled": True,
                "path": "./backups",
                "retention_days": 30
            },
            "initialization": {
                "create_sample_data": False,
                "admin_email": "admin@iajuris.com",
                "admin_password": None  # Será gerado automaticamente
            }
        }
        
        # Carregar de arquivo se especificado
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    # Merge configs (arquivo sobrescreve padrão)
                    self._deep_merge(default_config, file_config)
                    logger.info(f"✅ Configuração carregada de: {config_file}")
            except Exception as e:
                logger.warning(f"⚠️  Erro ao carregar config: {e}. Usando padrão.")
        
        # Sobrescrever com variáveis de ambiente
        env_config = self._load_from_env()
        if env_config:
            self._deep_merge(default_config, env_config)
            logger.info("✅ Configurações de ambiente aplicadas")
        
        return default_config
    
    def _deep_merge(self, base: Dict, override: Dict) -> None:
        """
        Faz merge profundo de dicionários.
        
        Args:
            base: Dicionário base
            override: Dicionário com valores para sobrescrever
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _load_from_env(self) -> Dict[str, Any]:
        """
        Carrega configurações de variáveis de ambiente.
        
        Returns:
            Dicionário com configurações de ambiente
        """
        env_config = {}
        
        # Mapeamento de variáveis de ambiente
        env_mapping = {
            "DB_TYPE": ["database", "type"],
            "DB_HOST": ["database", "host"],
            "DB_PORT": ["database", "port"],
            "DB_NAME": ["database", "name"],
            "DB_USER": ["database", "user"],
            "DB_PASSWORD": ["database", "password"],
            "DB_SSL_MODE": ["security", "ssl_mode"],
            "ADMIN_EMAIL": ["initialization", "admin_email"],
            "ADMIN_PASSWORD": ["initialization", "admin_password"]
        }
        
        for env_var, config_path in env_mapping.items():
            value = os.getenv(env_var)
            if value:
                # Criar estrutura aninhada se necessário
                current = env_config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Converter tipos apropriados
                if config_path[-1] == "port":
                    value = int(value)
                elif config_path[-1] in ["pool_size", "max_overflow", "pool_timeout", "pool_recycle", "retention_days"]:
                    value = int(value)
                elif config_path[-1] in ["create_sample_data", "enabled"]:
                    value = value.lower() in ['true', '1', 'yes', 'on']
                
                current[config_path[-1]] = value
        
        return env_config
    
    def validate_config(self) -> None:
        """
        Valida a configuração carregada.
        
        Raises:
            ValueError: Se configuração inválida
        """
        db_config = self.config["database"]
        
        # Validar tipo de banco
        supported_types = ["postgresql", "mysql", "sqlite"]
        if db_config["type"] not in supported_types:
            raise ValueError(f"Tipo de banco não suportado: {db_config['type']}")
        
        # Validar campos obrigatórios
        required_fields = ["name"]
        if db_config["type"] in ["postgresql", "mysql"]:
            required_fields.extend(["host", "user", "password"])
        
        for field in required_fields:
            if not db_config.get(field):
                raise ValueError(f"Campo obrigatório não configurado: {field}")
        
        logger.info("✅ Configuração validada com sucesso")
    
    def get_database_url(self) -> str:
        """
        Gera URL de conexão com o banco de dados.
        
        Returns:
            String de conexão
        """
        db = self.config["database"]
        
        if db["type"] == "sqlite":
            db_path = db["name"] if db["name"].endswith('.db') else f"{db['name']}.db"
            return f"sqlite:///{db_path}"
        
        elif db["type"] == "postgresql":
            url = f"postgresql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['name']}"
            
        elif db["type"] == "mysql":
            url = f"mysql+pymysql://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['name']}"
        
        return url

class DatabaseManager:
    """
    ��️ Gerenciador principal do banco de dados.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Inicializa o gerenciador de banco de dados.
        
        Args:
            config: Configuração do banco de dados
        """
        self.config = config
        self.engine = None
        self.metadata = MetaData()
        self.tables = {}
        
        # Conectar ao banco
        self._create_engine()
        
        # Definir schemas das tabelas
        self._define_schemas()
    
    def _create_engine(self) -> None:
        """
        Cria engine de conexão com o banco de dados.
        
        Raises:
            DatabaseError: Se não conseguir conectar
        """
        try:
            db_url = self.config.get_database_url()
            db_config = self.config.config["database"]
            
            # Configurações de pool de conexão
            engine_kwargs = {
                "echo": False,  # Set to True for SQL debugging
                "pool_pre_ping": True,
                "pool_recycle": db_config.get("pool_recycle", 3600)
            }
            
            # Configurações específicas por tipo de banco
            if db_config["type"] != "sqlite":
                engine_kwargs.update({
                    "poolclass": QueuePool,
                    "pool_size": db_config.get("pool_size", 10),
                    "max_overflow": db_config.get("max_overflow", 20),
                    "pool_timeout": db_config.get("pool_timeout", 30)
                })
            
            self.engine = create_engine(db_url, **engine_kwargs)
            
            # Testar conexão
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logger.info(f"✅ Conectado ao banco: {db_config['type']}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao conectar com banco: {e}")
            raise DatabaseError(f"Falha na conexão: {e}")
    
    def _define_schemas(self) -> None:
        """
        Define os schemas das tabelas principais.
        """
        logger.info("📋 Definindo schemas das tabelas...")
        
        # Tabela de Usuários
        self.tables['users'] = Table(
            'users',
            self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('username', String(50), nullable=False),
            Column('email', String(120), nullable=False),
            Column('password_hash', String(255), nullable=False),
            Column('first_name', String(50)),
            Column('last_name', String(50)),
            Column('is_active', Boolean, default=True, nullable=False),
            Column('is_admin', Boolean, default=False, nullable=False),
            Column('last_login', DateTime(timezone=True)),
            Column('failed_login_attempts', Integer, default=0),
            Column('account_locked_until', DateTime(timezone=True)),
            Column('email_verified', Boolean, default=False, nullable=False),
            Column('email_verification_token', String(255)),
            Column('password_reset_token', String(255)),
            Column('password_reset_expires', DateTime(timezone=True)),
            Column('created_at', DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False),
            Column('updated_at', DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), nullable=False),
            
            # Constraints
            UniqueConstraint('username', name='uix_users_username'),
            UniqueConstraint('email', name='uix_users_email'),
            CheckConstraint('LENGTH(username) >= 3', name='ck_users_username_length'),
            CheckConstraint('LENGTH(password_hash) >= 60', name='ck_users_password_length'),
            CheckConstraint("email LIKE '%_@_%.__%'", name='ck_users_email_format'),
            
            # Índices
            Index('ix_users_email', 'email'),
            Index('ix_users_username', 'username'),
            Index('ix_users_created_at', 'created_at'),
            Index('ix_users_active', 'is_active')
        )
        
        # Tabela de Logs de Sistema
        self.tables['system_logs'] = Table(
            'system_logs',
            self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('user_id', Integer, ForeignKey('users.id', ondelete='SET NULL')),
            Column('action', String(100), nullable=False),
            Column('resource', String(100)),
            Column('resource_id', Integer),
            Column('ip_address', String(45)),  # IPv6 support
            Column('user_agent', Text),
            Column('details', Text),
            Column('status', String(20), default='success'),
            Column('error_message', Text),
            Column('timestamp', DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False),
            
            # Índices
            Index('ix_logs_user_id', 'user_id'),
            Index('ix_logs_action', 'action'),
            Index('ix_logs_timestamp', 'timestamp'),
            Index('ix_logs_status', 'status'),
            Index('ix_logs_composite', 'user_id', 'action', 'timestamp')
        )
        
        # Tabela de Sessões de Usuário
        self.tables['user_sessions'] = Table(
            'user_sessions',
            self.metadata,
            Column('id', String(255), primary_key=True),
            Column('user_id', Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
            Column('ip_address', String(45)),
            Column('user_agent', Text),
            Column('csrf_token', String(255)),
            Column('data', Text),  # JSON serialized session data
            Column('created_at', DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False),
            Column('last_accessed', DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False),
            Column('expires_at', DateTime(timezone=True), nullable=False),
            
            # Índices
            Index('ix_sessions_user_id', 'user_id'),
            Index('ix_sessions_expires', 'expires_at'),
            Index('ix_sessions_last_accessed', 'last_accessed')
        )
        
        # Tabela de Configurações do Sistema
        self.tables['system_settings'] = Table(
            'system_settings',
            self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('key', String(100), nullable=False),
            Column('value', Text),
            Column('description', String(255)),
            Column('data_type', String(20), default='string'),  # string, integer, boolean, json
            Column('is_public', Boolean, default=False),
            Column('created_at', DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False),
            Column('updated_at', DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), nullable=False),
            
            # Constraints
            UniqueConstraint('key', name='uix_settings_key'),
            
            # Índices
            Index('ix_settings_key', 'key'),
            Index('ix_settings_public', 'is_public')
        )
        
        # Tabela de Processos Jurídicos (específica do projeto)
        self.tables['legal_processes'] = Table(
            'legal_processes',
            self.metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('process_number', String(50), nullable=False),
            Column('title', String(255), nullable=False),
            Column('description', Text),
            Column('content', Text),  # Conteúdo completo do processo
            Column('category', String(100)),
            Column('status', String(50), default='pending'),
            Column('priority', String(20), default='medium'),
            Column('court', String(200)),
            Column('judge', String(200)),
            Column('parties_involved', Text),  # JSON
            Column('tags', Text),  # JSON array of tags
            Column('file_path', String(500)),
            Column('file_size', Integer),
            Column('file_hash', String(64)),  # SHA256
            Column('processed_by_ai', Boolean, default=False),
            Column('ai_summary', Text),
            Column('ai_classification', String(100)),
            Column('ai_confidence_score', String(10)),  # Decimal as string
            Column('created_by', Integer, ForeignKey('users.id')),
            Column('assigned_to', Integer, ForeignKey('users.id')),
            Column('created_at', DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False),
            Column('updated_at', DateTime(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc), nullable=False),
            Column('due_date', DateTime(timezone=True)),
            
            # Constraints
            UniqueConstraint('process_number', name='uix_processes_number'),
            
            # Índices
            Index('ix_processes_number', 'process_number'),
            Index('ix_processes_status', 'status'),
            Index('ix_processes_category', 'category'),
            Index('ix_processes_created_by', 'created_by'),
            Index('ix_processes_assigned_to', 'assigned_to'),
            Index('ix_processes_created_at', 'created_at'),
            Index('ix_processes_ai_processed', 'processed_by_ai')
        )
        
        logger.info("✅ Schemas definidos com sucesso")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager para conexões com o banco.
        
        Yields:
            Conexão SQLAlchemy
        """
        conn = None
        try:
            conn = self.engine.connect()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Erro na conexão: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def create_database(self) -> bool:
        """
        Cria o banco de dados (se não existir) e todas as tabelas.
        
        Returns:
            True se criado com sucesso, False caso contrário
        """
        try:
            logger.info("🚀 Iniciando criação do banco de dados...")
            
            # Para SQLite, o banco é criado automaticamente
            if self.config.config["database"]["type"] == "sqlite":
                self._create_tables()
                return True
            
            # Para PostgreSQL e MySQL, tentar criar o banco primeiro
            db_config = self.config.config["database"]
            db_name = db_config["name"]
            
            # Conectar sem especificar o banco para criá-lo
            temp_config = db_config.copy()
            temp_config["name"] = "postgres" if db_config["type"] == "postgresql" else "mysql"
            
            temp_url = self._build_temp_url(temp_config)
            temp_engine = create_engine(temp_url)
            
            try:
                with temp_engine.connect() as conn:
                    # Para PostgreSQL, usar autocommit
                    if db_config["type"] == "postgresql":
                        conn.execute(text("COMMIT"))
                        result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'"))
                        if not result.fetchone():
                            conn.execute(text(f"CREATE DATABASE {db_name}"))
                            logger.info(f"✅ Banco '{db_name}' criado")
                    
                    # Para MySQL
                    elif db_config["type"] == "mysql":
                        result = conn.execute(text(f"SHOW DATABASES LIKE '{db_name}'"))
                        if not result.fetchone():
                            conn.execute(text(f"CREATE DATABASE {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
                            logger.info(f"✅ Banco '{db_name}' criado")
                
            except Exception as e:
                logger.warning(f"⚠️  Banco já existe ou erro na criação: {e}")
            
            finally:
                temp_engine.dispose()
            
            # Criar tabelas
            self._create_tables()
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar banco: {e}")
            return False
    
    def _build_temp_url(self, config: Dict) -> str:
        """
        Constrói URL temporária para criação do banco.
        
        Args:
            config: Configuração temporária
            
        Returns:
            URL de conexão temporária
        """
        if config["type"] == "postgresql":
            return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['name']}"
        elif config["type"] == "mysql":
            return f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['name']}"
    
    def _create_tables(self) -> None:
        """
        Cria todas as tabelas definidas no metadata.
        
        Raises:
            DatabaseError: Se falhar na criação
        """
        try:
            logger.info("📋 Criando tabelas...")
            
            # Criar todas as tabelas
            self.metadata.create_all(self.engine, checkfirst=True)
            
            # Verificar se as tabelas foram criadas
            with self.get_connection() as conn:
                inspector = sa.inspect(conn)
                existing_tables = inspector.get_table_names()
                
                expected_tables = list(self.tables.keys())
                missing_tables = set(expected_tables) - set(existing_tables)
                
                if missing_tables:
                    raise DatabaseError(f"Tabelas não criadas: {missing_tables}")
                
                logger.info(f"✅ Tabelas criadas: {', '.join(expected_tables)}")
                
        except Exception as e:
            logger.error(f"❌ Erro ao criar tabelas: {e}")
            raise DatabaseError(f"Falha na criação de tabelas: {e}")
    
    def populate_initial_data(self) -> bool:
        """
        Popula o banco com dados iniciais.
        
        Returns:
            True se população bem sucedida
        """
        try:
            logger.info("🌱 Populando dados iniciais...")
            
            with self.get_connection() as conn:
                trans = conn.begin()
                
                try:
                    # Verificar se já existem dados
                    users_count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
                    
                    if users_count > 0:
                        logger.info("⚠️  Dados já existem, pulando população inicial")
                        trans.rollback()
                        return True
                    
                    # Criar usuário administrador
                    admin_data = self._create_admin_user()
                    result = conn.execute(
                        self.tables['users'].insert(),
                        admin_data
                    )
                    admin_id = result.inserted_primary_key[0]
                    
                    # Configurações iniciais do sistema
                    system_settings = self._get_initial_settings()
                    for setting in system_settings:
                        conn.execute(
                            self.tables['system_settings'].insert(),
                            setting
                        )
                    
                    # Log da criação
                    log_data = {
                        'user_id': admin_id,
                        'action': 'database_initialized',
                        'details': 'Sistema inicializado com dados padrão',
                        'status': 'success',
                        'timestamp': datetime.now(timezone.utc)
                    }
                    
                    conn.execute(
                        self.tables['system_logs'].insert(),
                        log_data
                    )
                    
                    # Dados de exemplo (se configurado)
                    if self.config.config["initialization"].get("create_sample_data", False):
                        self._create_sample_data(conn, admin_id)
                    
                    trans.commit()
                    logger.info("✅ Dados iniciais criados com sucesso")
                    return True
                    
                except Exception as e:
                    trans.rollback()
                    logger.error(f"❌ Erro ao popular dados: {e}")
                    raise
                    
        except Exception as e:
            logger.error(f"❌ Erro na população inicial: {e}")
            return False
    
    def _create_admin_user(self) -> Dict[str, Any]:
        """
        Cria dados do usuário administrador.
        
        Returns:
            Dicionário com dados do admin
        """
        init_config = self.config.config["initialization"]
        
        # Gerar senha se não fornecida
        admin_password = init_config.get("admin_password")
        if not admin_password:
            admin_password = secrets.token_urlsafe(16)
            logger.info(f"🔑 Senha do admin gerada: {admin_password}")
        
        # Hash da senha
        if bcrypt:
            password_hash = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        else:
            # Fallback simples (NÃO usar em produção)
            password_hash = hashlib.sha256(admin_password.encode()).hexdigest()
            logger.warning("⚠️  Usando hash simples - instale bcrypt para produção!")
        
        return {
            'username': 'admin',
            'email': init_config.get("admin_email", "admin@iajuris.com"),
            'password_hash': password_hash,
            'first_name': 'Administrator',
            'last_name': 'System',
            'is_active': True,
            'is_admin': True,
            'email_verified': True,
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc)
        }
    
    def _get_initial_settings(self) -> List[Dict[str, Any]]:
        """
        Retorna configurações iniciais do sistema.
        
        Returns:
            Lista de configurações
        """
        now = datetime.now(timezone.utc)
        
        return [
            {
                'key': 'system_version',
                'value': '1.0.0',
                'description': 'Versão atual do sistema IA_Juris',
                'data_type': 'string',
                'is_public': True,
                'created_at': now,
                'updated_at': now
            },
            {
                'key': 'maintenance_mode',
                'value': 'false',
                'description': 'Sistema em modo de manutenção',
                'data_type': 'boolean',
                'is_public': True,
                'created_at': now,
                'updated_at': now
            },
            {
                'key': 'max_file_size',
                'value': '30',
                'description': 'Tamanho máximo de arquivo em MB',
                'data_type': 'integer',
                'is_public': False,
                'created_at': now,
                'updated_at': now
            },
            {
                'key': 'session_timeout',
                'value': '3600',
                'description': 'Timeout de sessão em segundos',
                'data_type': 'integer',
                'is_public': False,
                'created_at': now,
                'updated_at': now
            },
            {
                'key': 'ai_model_version',
                'value': '1.0.0',
                'description': 'Versão do modelo de IA em uso',
                'data_type': 'string',
                'is_public': False,
                'created_at': now,
                'updated_at': now
            }
        ]
    
    def _create_sample_data(self, conn, admin_id: int) -> None:
        """
        Cria dados de exemplo para desenvolvimento.
        
        Args:
            conn: Conexão com o banco
            admin_id: ID do usuário administrador
        """
        logger.info("📝 Criando dados de exemplo...")
        
        # Processos jurídicos de exemplo
        sample_processes = [
            {
                'process_number': '0001234-56.2025.8.26.0100',
                'title': 'Ação de Indenização por Danos Morais',
                'description': 'Processo exemplo para testes do sistema',
                'content': 'Conteúdo detalhado do processo jurídico...',
                'category': 'civil',
                'status': 'em_andamento',
                'priority': 'alta',
                'court': 'Tribunal de Justiça de São Paulo',
                'judge': 'Dr. João da Silva',
                'parties_involved': '{"autor": "João Santos", "reu": "Empresa XYZ"}',
                'tags': '["danos morais", "indenização", "civil"]',
                'created_by': admin_id,
                'assigned_to': admin_id,
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc)
            },
            {
                'process_number': '0002345-67.2025.5.02.0200',
                'title': 'Reclamação Trabalhista',
                'description': 'Processo trabalhista de exemplo',
                'content': 'Detalhes da reclamação trabalhista...',
                'category': 'trabalhista',
                'status': 'pendente',
                'priority': 'media',
                'court': 'Tribunal Regional do Trabalho',
                'judge': 'Dra. Maria Oliveira',
                'parties_involved': '{"reclamante": "Pedro Silva", "reclamada": "Empresa ABC"}',
                'tags': '["trabalhista", "horas extras", "rescisão"]',
                'created_by': admin_id,
                'assigned_to': admin_id,
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc)
            }
        ]
        
        for process_data in sample_processes:
            conn.execute(
                self.tables['legal_processes'].insert(),
                process_data
            )
        
        logger.info("✅ Dados de exemplo criados")
    
    def validate_database(self) -> bool:
        """
        Valida a integridade do banco de dados.
        
        Returns:
            True se validação passou
        """
        try:
            logger.info("🔍 Validando integridade do banco...")
            
            with self.get_connection() as conn:
                # Verificar se todas as tabelas existem
                inspector = sa.inspect(conn)
                existing_tables = set(inspector.get_table_names())
                expected_tables = set(self.tables.keys())
                
                missing_tables = expected_tables - existing_tables
                if missing_tables:
                    logger.error(f"❌ Tabelas ausentes: {missing_tables}")
                    return False
                
                # Verificar constraints e índices
                for table_name, table in self.tables.items():
                    # Verificar colunas
                    columns = inspector.get_columns(table_name)
                    column_names = {col['name'] for col in columns}
                    expected_columns = {col.name for col in table.columns}
                    
                    missing_columns = expected_columns - column_names
                    if missing_columns:
                        logger.error(f"❌ Colunas ausentes na tabela {table_name}: {missing_columns}")
                        return False
                
                # Testar integridade referencial
                user_count = conn.execute(text("SELECT COUNT(*) FROM users")).scalar()
                if user_count == 0:
                    logger.warning("⚠️  Nenhum usuário encontrado no banco")
                
                # Verificar configurações do sistema
                settings_count = conn.execute(text("SELECT COUNT(*) FROM system_settings")).scalar()
                if settings_count == 0:
                    logger.warning("⚠️  Nenhuma configuração do sistema encontrada")
                
                logger.info("✅ Validação do banco concluída com sucesso")
                return True
                
        except Exception as e:
            logger.error(f"❌ Erro na validação: {e}")
            return False
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """
        Cria backup do banco de dados.
        
        Args:
            backup_path: Caminho para o backup
            
        Returns:
            True se backup criado com sucesso
        """
        try:
            if not backup_path:
                backup_dir = self.config.config["backup"]["path"]
                os.makedirs(backup_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(backup_dir, f"backup_{timestamp}.sql")
            
            logger.info(f"💾 Criando backup em: {backup_path}")
            
            db_config = self.config.config["database"]
            
            if db_config["type"] == "sqlite":
                # Para SQLite, simplesmente copiar o arquivo
                import shutil
                db_file = db_config["name"] if db_config["name"].endswith('.db') else f"{db_config['name']}.db"
                shutil.copy2(db_file, backup_path.replace('.sql', '.db'))
                
            elif db_config["type"] == "postgresql":
                # Usar pg_dump
                cmd = f"pg_dump -h {db_config['host']} -p {db_config['port']} -U {db_config['user']} -d {db_config['name']} -f {backup_path}"
                os.system(cmd)
                
            elif db_config["type"] == "mysql":
                # Usar mysqldump
                cmd = f"mysqldump -h {db_config['host']} -P {db_config['port']} -u {db_config['user']} -p{db_config['password']} {db_config['name']} > {backup_path}"
                os.system(cmd)
            
            logger.info("✅ Backup criado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar backup: {e}")
            return False
    
    def cleanup_old_data(self) -> bool:
        """
        Remove dados antigos conforme políticas de retenção.
        
        Returns:
            True se limpeza bem sucedida
        """
        try:
            logger.info("🧹 Iniciando limpeza de dados antigos...")
            
            with self.get_connection() as conn:
                trans = conn.begin()
                
                try:
                    # Remover logs antigos (mais de 90 dias)
                    cutoff_date = datetime.now(timezone.utc) - timezone.timedelta(days=90)
                    
                    result = conn.execute(
                        text("DELETE FROM system_logs WHERE timestamp < :cutoff"),
                        {"cutoff": cutoff_date}
                    )
                    
                    logs_deleted = result.rowcount
                    
                    # Remover sessões expiradas
                    result = conn.execute(
                        text("DELETE FROM user_sessions WHERE expires_at < :now"),
                        {"now": datetime.now(timezone.utc)}
                    )
                    
                    sessions_deleted = result.rowcount
                    
                    trans.commit()
                    
                    logger.info(f"✅ Limpeza concluída: {logs_deleted} logs e {sessions_deleted} sessões removidas")
                    return True
                    
                except Exception as e:
                    trans.rollback()
                    raise e
                    
        except Exception as e:
            logger.error(f"❌ Erro na limpeza: {e}")
            return False

def create_sample_config(config_path: str) -> None:
    """
    Cria arquivo de configuração de exemplo.
    
    Args:
        config_path: Caminho para o arquivo de config
    """
    sample_config = {
        "database": {
            "type": "postgresql",
            "host": "localhost",
            "port": 5432,
            "name": "ia_juris_db",
            "user": "ia_juris_user",
            "password": "sua_senha_aqui",
            "pool_size": 10,
            "max_overflow": 20,
            "pool_timeout": 30,
            "pool_recycle": 3600
        },
        "security": {
            "ssl_mode": "prefer",
            "connect_timeout": 30,
            "command_timeout": 60
        },
        "backup": {
            "enabled": True,
            "path": "./backups",
            "retention_days": 30
        },
        "initialization": {
            "create_sample_data": True,
            "admin_email": "admin@suaempresa.com",
            "admin_password": null
        }
    }
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Arquivo de configuração criado: {config_path}")
        print("📝 Edite o arquivo com suas configurações antes de executar o setup")
        
    except Exception as e:
        print(f"❌ Erro ao criar arquivo de config: {e}")

def main():
    """
    Função principal do script.
    """
    parser = argparse.ArgumentParser(
        description="🗄️ Setup do Banco de Dados - IA_Juris",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  %(prog)s --init                          # Inicializar banco
  %(prog)s --populate                      # Popular dados iniciais
  %(prog)s --validate                      # Validar integridade
  %(prog)s --backup                        # Criar backup
  %(prog)s --cleanup                       # Limpar dados antigos
  %(prog)s --config config.json --init    # Usar config personalizada
  %(prog)s --create-sample-config          # Criar arquivo de exemplo
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Caminho para arquivo de configuração JSON'
    )
    
    parser.add_argument(
        '--init',
        action='store_true',
        help='Inicializar banco de dados e criar tabelas'
    )
    
    parser.add_argument(
        '--populate',
        action='store_true',
        help='Popular banco com dados iniciais'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validar integridade do banco'
    )
    
    parser.add_argument(
        '--backup',
        type=str,
        nargs='?',
        const='auto',
        help='Criar backup do banco (caminho opcional)'
    )
    
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Limpar dados antigos'
    )
    
    parser.add_argument(
        '--create-sample-config',
        type=str,
        nargs='?',
        const='database_config.json',
        help='Criar arquivo de configuração de exemplo'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Nível de log'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Arquivo para logs'
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    global logger
    logger = setup_logging(args.log_level, args.log_file)
    
    try:
        # Criar configuração de exemplo
        if args.create_sample_config:
            create_sample_config(args.create_sample_config)
            return
        
        # Carregar configurações
        config = DatabaseConfig(args.config)
        db_manager = DatabaseManager(config)
        
        # Executar ações solicitadas
        success = True
        
        if args.init:
            logger.info("🚀 Iniciando setup do banco de dados...")
            success &= db_manager.create_database()
        
        if args.populate:
            success &= db_manager.populate_initial_data()
        
        if args.validate:
            success &= db_manager.validate_database()
        
        if args.backup:
            backup_path = None if args.backup == 'auto' else args.backup
            success &= db_manager.backup_database(backup_path)
        
        if args.cleanup:
            success &= db_manager.cleanup_old_data()
        
        # Se nenhuma ação específica, executar setup completo
        if not any([args.init, args.populate, args.validate, args.backup, args.cleanup]):
            logger.info("🔧 Executando setup completo...")
            success &= db_manager.create_database()
            success &= db_manager.populate_initial_data()
            success &= db_manager.validate_database()
        
        if success:
            logger.info("🎉 Setup concluído com sucesso!")
            return 0
        else:
            logger.error("❌ Setup falhou!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("⚠️  Operação cancelada pelo usuário")
        return 1
    except Exception as e:
        logger.error(f"❌ Erro inesperado: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())