"""
Rate Limiter Middleware para APIs
Implementação robusta com suporte a Redis e armazenamento em memória
Compatível com Flask, Django e FastAPI
"""

import time
import json
import hashlib
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import wraps
import threading
import logging
from abc import ABC, abstractmethod

# Imports condicionais para diferentes frameworks
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from flask import Flask, request, jsonify, g
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    from django.http import JsonResponse
    from django.utils.deprecation import MiddlewareMixin
    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False

try:
    from fastapi import Request, HTTPException
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RateLimitRule:
    """
    Representa uma regra de limitação de taxa
    """
    requests: int = 100          # Número máximo de requisições
    window: int = 60             # Janela de tempo em segundos
    per_ip: bool = True          # Aplicar por IP
    per_user: bool = False       # Aplicar por usuário autenticado
    endpoints: List[str] = field(default_factory=list)  # Endpoints específicos
    methods: List[str] = field(default_factory=lambda: ['GET', 'POST', 'PUT', 'DELETE'])
    exempt_ips: List[str] = field(default_factory=list)  # IPs isentos
    custom_key_func: Optional[Callable] = None  # Função customizada para gerar chave


class StorageBackend(ABC):
    """
    Interface abstrata para backends de armazenamento
    """
    
    @abstractmethod
    def increment(self, key: str, window: int) -> int:
        """Incrementa contador e retorna valor atual"""
        pass
    
    @abstractmethod
    def get_count(self, key: str, window: int) -> int:
        """Obtém contagem atual para a chave"""
        pass
    
    @abstractmethod
    def cleanup_expired(self):
        """Remove entradas expiradas"""
        pass


class MemoryStorage(StorageBackend):
    """
    Backend de armazenamento em memória usando estruturas nativas
    """
    
    def __init__(self):
        self._data: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 60  # Cleanup a cada 60 segundos
    
    def increment(self, key: str, window: int) -> int:
        with self._lock:
            current_time = time.time()
            
            # Auto-cleanup periódico
            if current_time - self._last_cleanup > self._cleanup_interval:
                self._cleanup_expired_internal()
                self._last_cleanup = current_time
            
            # Remove entradas expiradas para esta chave
            window_start = current_time - window
            while self._data[key] and self._data[key][0] < window_start:
                self._data[key].popleft()
            
            # Adiciona nova entrada
            self._data[key].append(current_time)
            
            return len(self._data[key])
    
    def get_count(self, key: str, window: int) -> int:
        with self._lock:
            current_time = time.time()
            window_start = current_time - window
            
            # Remove entradas expiradas
            while self._data[key] and self._data[key][0] < window_start:
                self._data[key].popleft()
            
            return len(self._data[key])
    
    def cleanup_expired(self):
        with self._lock:
            self._cleanup_expired_internal()
    
    def _cleanup_expired_internal(self):
        """Cleanup interno sem lock (assumindo que já está protegido)"""
        current_time = time.time()
        keys_to_remove = []
        
        for key, timestamps in self._data.items():
            # Remove timestamps antigos (assumindo janela máxima de 1 hora)
            max_window = 3600
            window_start = current_time - max_window
            
            while timestamps and timestamps[0] < window_start:
                timestamps.popleft()
            
            # Remove chaves vazias
            if not timestamps:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._data[key]


class RedisStorage(StorageBackend):
    """
    Backend de armazenamento usando Redis
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, 
                 prefix: str = "rate_limit:"):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis não está disponível. Instale com: pip install redis")
        
        self.redis = redis_client or redis.Redis(
            host='localhost', 
            port=6379, 
            db=0, 
            decode_responses=True
        )
        self.prefix = prefix
    
    def increment(self, key: str, window: int) -> int:
        pipe = self.redis.pipeline()
        full_key = f"{self.prefix}{key}"
        current_time = time.time()
        
        # Remove entradas expiradas
        pipe.zremrangebyscore(full_key, 0, current_time - window)
        
        # Adiciona nova entrada
        pipe.zadd(full_key, {str(current_time): current_time})
        
        # Define expiração
        pipe.expire(full_key, window + 10)  # +10 segundos de margem
        
        # Conta entradas atuais
        pipe.zcard(full_key)
        
        results = pipe.execute()
        return results[-1]  # Retorna o resultado do zcard
    
    def get_count(self, key: str, window: int) -> int:
        full_key = f"{self.prefix}{key}"
        current_time = time.time()
        
        # Remove expiradas e conta
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(full_key, 0, current_time - window)
        pipe.zcard(full_key)
        
        results = pipe.execute()
        return results[-1]
    
    def cleanup_expired(self):
        # Redis limpa automaticamente com TTL
        pass


class RateLimiter:
    """
    Classe principal do Rate Limiter
    """
    
    def __init__(self, storage: Optional[StorageBackend] = None, 
                 default_rule: Optional[RateLimitRule] = None):
        self.storage = storage or MemoryStorage()
        self.default_rule = default_rule or RateLimitRule()
        self.rules: List[RateLimitRule] = []
        self.global_exempt_ips = set()
        
    def add_rule(self, rule: RateLimitRule):
        """Adiciona uma regra de rate limiting"""
        self.rules.append(rule)
        
    def add_exempt_ip(self, ip: str):
        """Adiciona IP globalmente isento"""
        self.global_exempt_ips.add(ip)
    
    def get_client_ip(self, request_data: Dict[str, Any]) -> str:
        """
        Extrai IP do cliente considerando proxies
        """
        # Verifica headers de proxy comuns
        headers_to_check = [
            'X-Forwarded-For',
            'X-Real-IP',
            'X-Client-IP',
            'CF-Connecting-IP',  # Cloudflare
            'True-Client-IP'     # Akamai
        ]
        
        for header in headers_to_check:
            ip = request_data.get('headers', {}).get(header)
            if ip:
                # Pega o primeiro IP se houver múltiplos
                return ip.split(',')[0].strip()
        
        # Fallback para IP direto
        return request_data.get('remote_addr', '127.0.0.1')
    
    def get_applicable_rule(self, method: str, endpoint: str) -> RateLimitRule:
        """
        Encontra a regra mais específica aplicável
        """
        for rule in self.rules:
            # Verifica método
            if method.upper() not in [m.upper() for m in rule.methods]:
                continue
            
            # Verifica endpoint
            if rule.endpoints:
                endpoint_match = any(
                    endpoint.startswith(ep) for ep in rule.endpoints
                )
                if not endpoint_match:
                    continue
            
            return rule
        
        return self.default_rule
    
    def generate_cache_key(self, rule: RateLimitRule, request_data: Dict[str, Any]) -> str:
        """
        Gera chave de cache baseada na regra e dados da requisição
        """
        components = []
        
        if rule.custom_key_func:
            return rule.custom_key_func(request_data)
        
        if rule.per_ip:
            ip = self.get_client_ip(request_data)
            components.append(f"ip:{ip}")
        
        if rule.per_user and 'user_id' in request_data:
            components.append(f"user:{request_data['user_id']}")
        
        # Adiciona endpoint se especificado na regra
        if rule.endpoints:
            endpoint = request_data.get('endpoint', '')
            components.append(f"endpoint:{endpoint}")
        
        # Hash final para evitar chaves muito longas
        key_string = ":".join(components)
        if len(key_string) > 200:  # Limite para evitar problemas
            key_string = hashlib.md5(key_string.encode()).hexdigest()
        
        return key_string
    
    def is_exempt(self, request_data: Dict[str, Any], rule: RateLimitRule) -> bool:
        """
        Verifica se a requisição está isenta de rate limiting
        """
        ip = self.get_client_ip(request_data)
        
        # Verifica isenção global
        if ip in self.global_exempt_ips:
            return True
        
        # Verifica isenção da regra específica
        if ip in rule.exempt_ips:
            return True
        
        return False
    
    def check_rate_limit(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifica se a requisição excede o limite de taxa
        
        Returns:
            Dict com informações sobre o limite:
            - allowed: bool
            - limit: int 
            - remaining: int
            - reset_time: float
            - message: str
        """
        method = request_data.get('method', 'GET')
        endpoint = request_data.get('endpoint', '/')
        
        # Encontra regra aplicável
        rule = self.get_applicable_rule(method, endpoint)
        
        # Verifica isenção
        if self.is_exempt(request_data, rule):
            return {
                'allowed': True,
                'limit': rule.requests,
                'remaining': rule.requests,
                'reset_time': time.time() + rule.window,
                'message': 'Request allowed (exempt)'
            }
        
        # Gera chave de cache
        cache_key = self.generate_cache_key(rule, request_data)
        
        # Verifica limite
        current_count = self.storage.increment(cache_key, rule.window)
        remaining = max(0, rule.requests - current_count)
        reset_time = time.time() + rule.window
        
        allowed = current_count <= rule.requests
        
        result = {
            'allowed': allowed,
            'limit': rule.requests,
            'remaining': remaining,
            'reset_time': reset_time,
            'current_count': current_count
        }
        
        if not allowed:
            result['message'] = (
                f"Rate limit exceeded. "
                f"Limit: {rule.requests} requests per {rule.window} seconds. "
                f"Try again in {rule.window} seconds."
            )
        else:
            result['message'] = 'Request allowed'
        
        return result


# =============================================================================
# MIDDLEWARES PARA DIFERENTES FRAMEWORKS
# =============================================================================

if FLASK_AVAILABLE:
    class FlaskRateLimitMiddleware:
        """
        Middleware para Flask
        """
        
        def __init__(self, app: Flask, rate_limiter: RateLimiter):
            self.app = app
            self.rate_limiter = rate_limiter
            self.init_app(app)
        
        def init_app(self, app: Flask):
            app.before_request(self.before_request)
        
        def before_request(self):
            # Coleta dados da requisição
            request_data = {
                'method': request.method,
                'endpoint': request.endpoint or request.path,
                'remote_addr': request.remote_addr,
                'headers': dict(request.headers),
                'user_id': g.get('user_id') if hasattr(g, 'user_id') else None
            }
            
            # Verifica rate limit
            result = self.rate_limiter.check_rate_limit(request_data)
            
            if not result['allowed']:
                response = jsonify({
                    'error': 'Rate limit exceeded',
                    'message': result['message'],
                    'limit': result['limit'],
                    'reset_time': result['reset_time']
                })
                response.status_code = 429
                response.headers['X-RateLimit-Limit'] = str(result['limit'])
                response.headers['X-RateLimit-Remaining'] = str(result['remaining'])
                response.headers['X-RateLimit-Reset'] = str(int(result['reset_time']))
                response.headers['Retry-After'] = str(60)  # Default retry after
                return response


if DJANGO_AVAILABLE:
    class DjangoRateLimitMiddleware(MiddlewareMixin):
        """
        Middleware para Django
        """
        
        def __init__(self, get_response=None):
            super().__init__(get_response)
            # Configuração padrão - deve ser customizada via settings
            self.rate_limiter = RateLimiter()
        
        def process_request(self, request):
            # Coleta dados da requisição
            request_data = {
                'method': request.method,
                'endpoint': request.path,
                'remote_addr': self.get_client_ip(request),
                'headers': dict(request.META),
                'user_id': getattr(request.user, 'id', None) if hasattr(request, 'user') else None
            }
            
            # Verifica rate limit
            result = self.rate_limiter.check_rate_limit(request_data)
            
            if not result['allowed']:
                response_data = {
                    'error': 'Rate limit exceeded',
                    'message': result['message'],
                    'limit': result['limit'],
                    'reset_time': result['reset_time']
                }
                response = JsonResponse(response_data, status=429)
                response['X-RateLimit-Limit'] = str(result['limit'])
                response['X-RateLimit-Remaining'] = str(result['remaining'])
                response['X-RateLimit-Reset'] = str(int(result['reset_time']))
                response['Retry-After'] = str(60)
                return response
        
        def get_client_ip(self, request):
            """Extrai IP do cliente no Django"""
            x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
            if x_forwarded_for:
                ip = x_forwarded_for.split(',')[0]
            else:
                ip = request.META.get('REMOTE_ADDR')
            return ip


if FASTAPI_AVAILABLE:
    class FastAPIRateLimitMiddleware:
        """
        Middleware para FastAPI
        """
        
        def __init__(self, rate_limiter: RateLimiter):
            self.rate_limiter = rate_limiter
        
        async def __call__(self, request: Request, call_next):
            # Coleta dados da requisição
            request_data = {
                'method': request.method,
                'endpoint': str(request.url.path),
                'remote_addr': request.client.host if request.client else '127.0.0.1',
                'headers': dict(request.headers),
                'user_id': getattr(request.state, 'user_id', None)
            }
            
            # Verifica rate limit
            result = self.rate_limiter.check_rate_limit(request_data)
            
            if not result['allowed']:
                return JSONResponse(
                    status_code=429,
                    content={
                        'error': 'Rate limit exceeded',
                        'message': result['message'],
                        'limit': result['limit'],
                        'reset_time': result['reset_time']
                    },
                    headers={
                        'X-RateLimit-Limit': str(result['limit']),
                        'X-RateLimit-Remaining': str(result['remaining']),
                        'X-RateLimit-Reset': str(int(result['reset_time'])),
                        'Retry-After': '60'
                    }
                )
            
            response = await call_next(request)
            
            # Adiciona headers informativos
            response.headers['X-RateLimit-Limit'] = str(result['limit'])
            response.headers['X-RateLimit-Remaining'] = str(result['remaining'])
            response.headers['X-RateLimit-Reset'] = str(int(result['reset_time']))
            
            return response


# =============================================================================
# DECORADORES PARA USO DIRETO EM FUNÇÕES
# =============================================================================

def rate_limit(requests: int = 100, window: int = 60, 
               per_ip: bool = True, per_user: bool = False,
               storage: Optional[StorageBackend] = None):
    """
    Decorador para aplicar rate limiting diretamente em funções
    """
    if storage is None:
        storage = MemoryStorage()
    
    rule = RateLimitRule(requests=requests, window=window, per_ip=per_ip, per_user=per_user)
    limiter = RateLimiter(storage=storage, default_rule=rule)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Tenta extrair dados da requisição baseado no framework
            request_data = {'method': 'GET', 'endpoint': '/', 'remote_addr': '127.0.0.1', 'headers': {}}
            
            # Flask
            if FLASK_AVAILABLE and hasattr(request, 'method'):
                request_data = {
                    'method': request.method,
                    'endpoint': request.endpoint or request.path,
                    'remote_addr': request.remote_addr,
                    'headers': dict(request.headers)
                }
            
            result = limiter.check_rate_limit(request_data)
            
            if not result['allowed']:
                if FLASK_AVAILABLE:
                    response = jsonify({
                        'error': 'Rate limit exceeded',
                        'message': result['message']
                    })
                    response.status_code = 429
                    return response
                else:
                    raise Exception(f"Rate limit exceeded: {result['message']}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# EXEMPLOS DE CONFIGURAÇÃO E USO
# =============================================================================

def create_example_configurations():
    """
    Exemplos de configurações para diferentes cenários
    """
    
    # Configuração básica com armazenamento em memória
    basic_limiter = RateLimiter(
        storage=MemoryStorage(),
        default_rule=RateLimitRule(requests=100, window=60)
    )
    
    # Configuração com Redis (se disponível)
    if REDIS_AVAILABLE:
        redis_limiter = RateLimiter(
            storage=RedisStorage(),
            default_rule=RateLimitRule(requests=1000, window=60)
        )
        
        # Adiciona regras específicas
        redis_limiter.add_rule(RateLimitRule(
            requests=10,
            window=60,
            endpoints=['/api/upload', '/api/heavy-operation'],
            methods=['POST']
        ))
        
        redis_limiter.add_rule(RateLimitRule(
            requests=1000,
            window=3600,
            endpoints=['/api/data'],
            methods=['GET'],
            per_user=True
        ))
    
    # Configuração para APIs públicas vs autenticadas
    api_limiter = RateLimiter()
    
    # Limite restritivo para endpoints públicos
    api_limiter.add_rule(RateLimitRule(
        requests=20,
        window=60,
        endpoints=['/api/public'],
        per_ip=True
    ))
    
    # Limite mais generoso para usuários autenticados
    api_limiter.add_rule(RateLimitRule(
        requests=500,
        window=60,
        endpoints=['/api/private'],
        per_user=True,
        per_ip=False
    ))
    
    return basic_limiter, api_limiter


# =============================================================================
# EXEMPLOS DE USO COM DIFERENTES FRAMEWORKS
# =============================================================================

def flask_example():
    """Exemplo de uso com Flask"""
    if not FLASK_AVAILABLE:
        print("Flask não disponível")
        return
    
    app = Flask(__name__)
    
    # Configuração do rate limiter
    storage = RedisStorage() if REDIS_AVAILABLE else MemoryStorage()
    rate_limiter = RateLimiter(storage=storage)
    
    # Regras específicas
    rate_limiter.add_rule(RateLimitRule(
        requests=10,
        window=60,
        endpoints=['/api/upload'],
        methods=['POST']
    ))
    
    rate_limiter.add_rule(RateLimitRule(
        requests=100,
        window=60,
        endpoints=['/api'],
        methods=['GET', 'POST']
    ))
    
    # Adiciona middleware
    FlaskRateLimitMiddleware(app, rate_limiter)
    
    @app.route('/api/test')
    def test():
        return jsonify({'message': 'Success!'})
    
    @app.route('/api/upload', methods=['POST'])
    def upload():
        return jsonify({'message': 'File uploaded!'})
    
    # Uso com decorador
    @app.route('/api/decorated')
    @rate_limit(requests=5, window=60)
    def decorated_endpoint():
        return jsonify({'message': 'Decorated endpoint!'})
    
    return app


def django_example():
    """Exemplo de configuração para Django"""
    if not DJANGO_AVAILABLE:
        print("Django não disponível")
        return
    
    # No settings.py
    MIDDLEWARE_EXAMPLE = [
        'django.middleware.security.SecurityMiddleware',
        'path.to.rate_limiter.DjangoRateLimitMiddleware',  # Adicionar aqui
        'django.middleware.common.CommonMiddleware',
        # ... outros middlewares
    ]
    
    # Configuração customizada no settings.py
    RATE_LIMITER_CONFIG = {
        'STORAGE': 'redis',  # ou 'memory'
        'REDIS_HOST': 'localhost',
        'REDIS_PORT': 6379,
        'REDIS_DB': 0,
        'DEFAULT_LIMIT': 100,
        'DEFAULT_WINDOW': 60,
        'RULES': [
            {
                'requests': 10,
                'window': 60,
                'endpoints': ['/api/upload/'],
                'methods': ['POST']
            },
            {
                'requests': 1000,
                'window': 3600,
                'endpoints': ['/api/'],
                'per_user': True
            }
        ]
    }


def fastapi_example():
    """Exemplo de uso com FastAPI"""
    if not FASTAPI_AVAILABLE:
        print("FastAPI não disponível")
        return
    
    from fastapi import FastAPI
    
    app = FastAPI()
    
    # Configuração do rate limiter
    storage = RedisStorage() if REDIS_AVAILABLE else MemoryStorage()
    rate_limiter = RateLimiter(storage=storage)
    
    # Adiciona regras
    rate_limiter.add_rule(RateLimitRule(
        requests=100,
        window=60,
        endpoints=['/api'],
    ))
    
    # Adiciona middleware
    app.add_middleware(FastAPIRateLimitMiddleware, rate_limiter=rate_limiter)
    
    @app.get('/api/test')
    async def test():
        return {'message': 'Success!'}
    
    return app


# =============================================================================
# MONITORAMENTO E MÉTRICAS
# =============================================================================

class RateLimiterMetrics:
    """
    Classe para coletar métricas do rate limiter
    """
    
    def __init__(self):
        self.total_requests = 0
        self.blocked_requests = 0
        self.requests_by_endpoint = defaultdict(int)
        self.blocked_by_endpoint = defaultdict(int)
        self._lock = threading.Lock()
    
    def record_request(self, endpoint: str, blocked: bool = False):
        with self._lock:
            self.total_requests += 1
            self.requests_by_endpoint[endpoint] += 1
            
            if blocked:
                self.blocked_requests += 1
                self.blocked_by_endpoint[endpoint] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'total_requests': self.total_requests,
                'blocked_requests': self.blocked_requests,
                'block_rate': self.blocked_requests / max(1, self.total_requests),
                'requests_by_endpoint': dict(self.requests_by_endpoint),
                'blocked_by_endpoint': dict(self.blocked_by_endpoint)
            }


if __name__ == '__main__':
    # Teste básico
    print("🧪 Testando Rate Limiter...")
    
    # Teste com storage em memória
    limiter = RateLimiter(
        storage=MemoryStorage(),
        default_rule=RateLimitRule(requests=5, window=10)
    )
    
    # Simula requisições
    for i in range(8):
        request_data = {
            'method': 'GET',
            'endpoint': '/api/test',
            'remote_addr': '192.168.1.100',
            'headers': {}
        }
        
        result = limiter.check_rate_limit(request_data)
        status = "✅ ALLOWED" if result['allowed'] else "❌ BLOCKED"
        
        print(f"Request {i+1}: {status} - "
              f"Remaining: {result['remaining']}/{result['limit']}")
        
        time.sleep(1)
    
    print("\n✅ Teste concluído!")
    print("\n📚 Para usar em produção:")
    print("1. Configure Redis para alta performance")
    print("2. Ajuste as regras conforme seu caso de uso")
    print("3. Monitore métricas de bloqueio")
    print("4. Configure alertas para abusos")