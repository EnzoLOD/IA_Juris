"""
Módulo de inicialização para os middlewares da aplicação.
Centraliza a configuração e aplicação de Rate Limiter, Autenticação, e outros middlewares.
"""

import logging
from typing import Dict, Any, Optional, Union

# --- Importações Relativas ---
# Importa componentes do arquivo rate_limiter.py
from .rate_limiter import (
    RateLimiter,
    RateLimitRule,
    MemoryStorage,
    RedisStorage,
    FlaskRateLimitMiddleware,
    # Se estiver desenvolvendo para outros frameworks, adicione as classes de middleware aqui:
    # FastAPIRateLimitMiddleware,
    # DjangoRateLimitMiddleware,
)

# Importa o blueprint de autenticação e a função de inicialização do arquivo auth.py
# 'auth' é o Blueprint de autenticação, 'init_auth' é a função de inicialização
from .auth import auth, init_auth

# --- Configuração de Logging ---
# Configura um logger específico para o módulo de middlewares
logger = logging.getLogger(__name__)
# Define o nível de logging para INFO, para que mensagens informativas sejam exibidas
logger.setLevel(logging.INFO) 

# --- Função Principal de Inicialização dos Middlewares ---
def init_app_middlewares(app: Any, config: Optional[Dict[str, Any]] = None):
    """
    Inicializa e configura todos os middlewares da aplicação de forma centralizada.

    Esta função deve ser chamada no arquivo principal da sua aplicação (e.g., yourapplication/__init__.py)
    após a criação da instância do aplicativo Flask (ou outro framework) e o carregamento das configurações.

    Args:
        app (Any): A instância do aplicativo do framework (e.g., Flask, FastAPI, Django).
                   O tipo é 'Any' para manter a flexibilidade entre frameworks.
        config (Optional[Dict[str, Any]]): Um dicionário contendo as configurações específicas
                                           dos middlewares. Se None, a função tentará buscar
                                           as configurações em `app.config` (para Flask) ou
                                           equivalente no framework utilizado.

    Exemplo de uso na sua aplicação principal (Flask):
    ```python
    # yourapplication/__init__.py
    from flask import Flask
    from middleware import init_app_middlewares

    def create_app():
        app = Flask(__name__)
        # Carrega configurações da sua classe Config (ex: config.py)
        app.config.from_object('config.Config') 

        # Inicializa os middlewares da pasta 'middleware'
        init_app_middlewares(app)

        # ... (outras configurações, registro de blueprints, etc.)

        return app
    ```
    """
    # Se nenhuma configuração for explicitamente passada, tenta obtê-la do objeto 'app'.
    # Para Flask, `app.config` é um dicionário que contém as configurações.
    if config is None:
        config = getattr(app, 'config', {}) 
        if not config:
            logger.warning("Nenhuma configuração de middleware encontrada em 'app.config'. Certifique-se de carregar as configurações antes de inicializar os middlewares.")

    logger.info("Iniciando módulos de middleware...")

    # 1. Inicialização do Módulo de Autenticação (auth.py)
    # O módulo auth.py já contém uma função 'init_auth' que é responsável por:
    # - Registrar o Blueprint de autenticação no aplicativo.
    # - Inicializar extensões como Flask-Login (para gestão de sessões de usuário)
    # - Inicializar Flask-CSRFProtect (para proteção contra ataques CSRF).
    logger.info("Inicializando módulo de autenticação (auth.py)...")
    try:
        init_auth(app)
        logger.info("Módulo de autenticação inicializado com sucesso.")
    except Exception as e:
        logger.error(f"Falha ao inicializar o módulo de autenticação: {e}", exc_info=True)

    # 2. Inicialização do Middleware de Limitação de Taxa (rate_limiter.py)
    logger.info("Inicializando middleware de limitação de taxa (rate_limiter.py)...")
    # Obtém as configurações específicas para o Rate Limiter do dicionário de configurações
    rate_limiter_config = config.get('RATE_LIMITER', {})

    # Define o tipo de armazenamento para o Rate Limiter (in-memory ou Redis)
    storage_type = rate_limiter_config.get('STORAGE', 'memory').lower()
    storage: Union[MemoryStorage, RedisStorage] # Define o tipo da variável storage
    
    if storage_type == 'redis':
        redis_host = rate_limiter_config.get('REDIS_HOST', 'localhost')
        redis_port = rate_limiter_config.get('REDIS_PORT', 6379)
        redis_db = rate_limiter_config.get('REDIS_DB', 0)
        
        try:
            # Tenta importar a biblioteca 'redis'. Se não estiver instalada, um ImportError será levantado.
            import redis 
            # Cria uma instância do cliente Redis. 'decode_responses=True' garante que as respostas sejam strings.
            redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=True)
            # Tenta fazer um ping para verificar a conexão imediatamente.
            redis_client.ping() 
            storage = RedisStorage(redis_client=redis_client)
            logger.info(f"Rate Limiter configurado para usar RedisStorage em {redis_host}:{redis_port}/{redis_db}.")
        except ImportError:
            logger.warning("A biblioteca 'redis' não está instalada. O Rate Limiter usará 'MemoryStorage'. Instale com 'pip install redis' para usar Redis.")
            storage = MemoryStorage()
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Falha ao conectar ao servidor Redis para o Rate Limiter: {e}. Caindo de volta para 'MemoryStorage'.", exc_info=True)
            storage = MemoryStorage()
        except Exception as e:
            logger.error(f"Erro inesperado ao configurar Redis para o Rate Limiter: {e}. Caindo de volta para 'MemoryStorage'.", exc_info=True)
            storage = MemoryStorage()
    else:
        storage = MemoryStorage()
        logger.info("Rate Limiter configurado para usar MemoryStorage (armazenamento em memória).")

    # Inicializa a instância principal do RateLimiter com o armazenamento selecionado e uma regra padrão.
    # A regra padrão é aplicada a todas as requisições que não correspondem a nenhuma regra específica.
    default_requests = rate_limiter_config.get('DEFAULT_LIMIT', 100)
    default_window = rate_limiter_config.get('DEFAULT_WINDOW', 60)
    
    rate_limiter_instance = RateLimiter(
        storage=storage,
        default_rule=RateLimitRule(requests=default_requests, window=default_window)
    )
    logger.info(f"Regra de rate limit padrão definida: {default_requests} requisições por {default_window} segundos.")

    # Adiciona regras customizadas de limitação de taxa, lidas da configuração.
    # Essas regras permitem definir limites específicos para endpoints, métodos ou usuários.
    custom_rules_config = rate_limiter_config.get('RULES', [])
    for idx, rule_data in enumerate(custom_rules_config):
        try:
            # Cria uma instância de RateLimitRule a partir dos dados do dicionário.
            # As chaves do dicionário devem corresponder aos parâmetros do construtor de RateLimitRule.
            rule = RateLimitRule(**rule_data)
            rate_limiter_instance.add_rule(rule)
            logger.debug(f"Regra de rate limit personalizada #{idx+1} adicionada: {rule_data}")
        except TypeError as e:
            logger.error(f"Configuração inválida para regra de rate limit #{idx+1}: {rule_data}. Verifique os parâmetros. Erro: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Erro ao adicionar regra de rate limit #{idx+1}: {rule_data}. Erro: {e}", exc_info=True)

    # Adiciona IPs globais isentos de limitação de taxa.
    # Requisições originadas desses IPs não serão contabilizadas pelo Rate Limiter.
    exempt_ips = rate_limiter_config.get('EXEMPT_IPS', [])
    for ip in exempt_ips:
        rate_limiter_instance.add_exempt_ip(ip)
        logger.debug(f"IP isento global adicionado: {ip}")
    if exempt_ips:
        logger.info(f"Total de {len(exempt_ips)} IPs isentos de rate limit configurados.")

    # Aplica o middleware específico do framework (Flask neste exemplo).
    # Esta parte do código é crucial para integrar o RateLimiter ao ciclo de requisição/resposta do framework.
    try:
        # Importa Flask para verificar se a instância 'app' é de fato um aplicativo Flask.
        from flask import Flask 
        if isinstance(app, Flask):
            # Instancia e aplica o middleware FlaskRateLimitMiddleware ao aplicativo Flask.
            FlaskRateLimitMiddleware(app, rate_limiter_instance)
            logger.info("FlaskRateLimitMiddleware aplicado ao aplicativo Flask.")
        # Se você estiver usando FastAPI ou Django, adicione a lógica de integração aqui:
        # elif isinstance(app, FastAPI):
        #    # Para FastAPI, você adicionaria o middleware diretamente à instância do aplicativo.
        #    app.add_middleware(FastAPIRateLimitMiddleware, rate_limiter=rate_limiter_instance)
        #    logger.info("FastAPIRateLimitMiddleware aplicado ao aplicativo FastAPI.")
        # elif isinstance(app, Django): # Exemplo hipotético, Django tem um sistema de middleware diferente
        #    # Para Django, o middleware é geralmente adicionado na lista MIDDLEWARE em settings.py
        #    # e não precisa ser instanciado diretamente aqui, mas sim configurado.
        #    logger.warning("Para Django, configure o middleware de rate limit em settings.py.")
        else:
            logger.warning("Tipo de aplicativo desconhecido. O middleware de rate limit específico do framework não pôde ser aplicado.")
    except ImportError:
        logger.warning("A biblioteca 'Flask' não está instalada. Não foi possível aplicar o FlaskRateLimitMiddleware.")
    except Exception as e:
        logger.error(f"Falha ao aplicar o middleware de rate limit específico do framework: {e}", exc_info=True)

    logger.info("Todos os módulos de middleware inicializados com sucesso.")


# --- Exposição de Componentes (Opcional) ---
# A variável `__all__` define quais nomes serão importados quando um cliente fizer
# `from middleware import *`. Isso é útil para controlar a API pública do seu pacote.
__all__ = [
    'RateLimiter',
    'RateLimitRule',
    'MemoryStorage',
    'RedisStorage',
    'FlaskRateLimitMiddleware',
    # 'FastAPIRateLimitMiddleware',
    # 'DjangoRateLimitMiddleware',
    'auth', # O Blueprint de autenticação
    'init_auth', # A função de inicialização do módulo de autenticação
    'init_app_middlewares', # A função principal para inicializar todos os middlewares
]