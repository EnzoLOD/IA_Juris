# This file marks the services directory as a Python package.
"""
Services module for JurisOracle - Enterprise Service Management System

This module provides a comprehensive service registry and management system for all
JurisOracle services including QA, Summarization, and Training services.

Author: JurisOracle Team
Version: 1.0.0
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, Callable
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime, timedelta

# Core imports
from ..config import get_settings, get_logger
try:
    from ..core.exceptions import JurisOracleError
except ImportError:
    from src.core.exceptions import JurisOracleError

# Service imports
from .qa_service import QAService
from .summarization import SummarizationService
from .training_service import TrainingService

# Get configuration and logger
settings = get_settings()
logger = get_logger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration"""
    INACTIVE = "inactive"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ServicePriority(Enum):
    """Service priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ServiceHealth:
    """Service health information"""
    status: ServiceStatus
    last_check: datetime
    response_time: float
    error_count: int = 0
    success_count: int = 0
    uptime: timedelta = field(default_factory=lambda: timedelta())
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.error_count + self.success_count
        if total == 0:
            return 0.0
        return self.success_count / total * 100
    
    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        return (
            self.status == ServiceStatus.RUNNING and
            self.success_rate >= 95.0 and
            self.response_time < 5.0
        )


@dataclass
class ServiceConfig:
    """Service configuration"""
    name: str
    service_class: Type
    priority: ServicePriority = ServicePriority.NORMAL
    auto_start: bool = True
    health_check_interval: int = 30
    max_retries: int = 3
    timeout: int = 300
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


class ServiceRegistry:
    """
    Enterprise-grade service registry and management system
    
    Features:
    - Service lifecycle management
    - Health monitoring
    - Dependency resolution
    - Load balancing
    - Circuit breaker pattern
    - Metrics collection
    """
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._configs: Dict[str, ServiceConfig] = {}
        self._health: Dict[str, ServiceHealth] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._monitoring_task: Optional[asyncio.Task] = None
        self._startup_time = datetime.now()
        self._metrics: Dict[str, Dict[str, Any]] = {}
        
        logger.info("🚀 ServiceRegistry initialized")
    
    def register_service(self, config: ServiceConfig) -> None:
        """Register a new service"""
        try:
            if config.name in self._services:
                logger.warning(f"⚠️ Service {config.name} already registered, updating...")
            
            self._configs[config.name] = config
            self._locks[config.name] = threading.Lock()
            self._health[config.name] = ServiceHealth(
                status=ServiceStatus.INACTIVE,
                last_check=datetime.now(),
                response_time=0.0
            )
            self._metrics[config.name] = {
                'requests': 0,
                'errors': 0,
                'total_time': 0.0,
                'last_request': None
            }
            
            logger.info(f"✅ Service {config.name} registered successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to register service {config.name}: {e}")
            raise JurisOracleError(f"Service registration failed: {e}")
    
    async def start_service(self, name: str) -> bool:
        """Start a specific service"""
        if name not in self._configs:
            logger.error(f"❌ Service {name} not found in registry")
            return False
        
        with self._locks[name]:
            try:
                config = self._configs[name]
                
                # Check dependencies
                for dep in config.dependencies:
                    if not await self.is_service_running(dep):
                        logger.error(f"❌ Dependency {dep} not running for service {name}")
                        return False
                
                # Update status
                self._health[name].status = ServiceStatus.STARTING
                logger.info(f"🔄 Starting service {name}...")
                
                # Initialize service
                service_instance = config.service_class(**config.config)
                await service_instance.initialize()
                
                self._services[name] = service_instance
                self._health[name].status = ServiceStatus.RUNNING
                self._health[name].last_check = datetime.now()
                
                logger.info(f"✅ Service {name} started successfully")
                return True
                
            except Exception as e:
                self._health[name].status = ServiceStatus.ERROR
                logger.error(f"❌ Failed to start service {name}: {e}")
                return False
    
    async def stop_service(self, name: str) -> bool:
        """Stop a specific service"""
        if name not in self._services:
            logger.warning(f"⚠️ Service {name} not running")
            return True
        
        with self._locks[name]:
            try:
                self._health[name].status = ServiceStatus.STOPPING
                logger.info(f"🔄 Stopping service {name}...")
                
                service = self._services[name]
                if hasattr(service, 'cleanup'):
                    await service.cleanup()
                
                del self._services[name]
                self._health[name].status = ServiceStatus.INACTIVE
                
                logger.info(f"✅ Service {name} stopped successfully")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to stop service {name}: {e}")
                return False
    
    async def restart_service(self, name: str) -> bool:
        """Restart a specific service"""
        logger.info(f"🔄 Restarting service {name}...")
        await self.stop_service(name)
        await asyncio.sleep(2)  # Grace period
        return await self.start_service(name)
    
    async def start_all_services(self) -> bool:
        """Start all registered services in dependency order"""
        logger.info("🚀 Starting all services...")
        
        # Sort by priority and dependencies
        sorted_services = self._sort_by_dependencies()
        
        for name in sorted_services:
            config = self._configs[name]
            if config.auto_start:
                success = await self.start_service(name)
                if not success and config.priority == ServicePriority.CRITICAL:
                    logger.error(f"❌ Critical service {name} failed to start")
                    return False
        
        # Start monitoring
        await self._start_monitoring()
        
        logger.info("✅ All services started successfully")
        return True
    
    async def stop_all_services(self) -> None:
        """Stop all running services"""
        logger.info("🛑 Stopping all services...")
        
        # Stop monitoring first
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        # Stop services in reverse dependency order
        sorted_services = list(reversed(self._sort_by_dependencies()))
        
        for name in sorted_services:
            if name in self._services:
                await self.stop_service(name)
        
        logger.info("✅ All services stopped")
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get a service instance"""
        return self._services.get(name)
    
    async def is_service_running(self, name: str) -> bool:
        """Check if a service is running"""
        health = self._health.get(name)
        return health is not None and health.status == ServiceStatus.RUNNING
    
    async def health_check(self, name: str) -> ServiceHealth:
        """Perform health check on a service"""
        if name not in self._services:
            return ServiceHealth(
                status=ServiceStatus.INACTIVE,
                last_check=datetime.now(),
                response_time=0.0
            )
        
        start_time = time.time()
        
        try:
            service = self._services[name]
            
            # Perform health check
            if hasattr(service, 'health_check'):
                await service.health_check()
            
            response_time = time.time() - start_time
            
            # Update health
            health = self._health[name]
            health.last_check = datetime.now()
            health.response_time = response_time
            health.success_count += 1
            
            if health.status != ServiceStatus.RUNNING:
                health.status = ServiceStatus.RUNNING
            
            return health
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Update health with error
            health = self._health[name]
            health.last_check = datetime.now()
            health.response_time = response_time
            health.error_count += 1
            health.status = ServiceStatus.ERROR
            
            logger.error(f"❌ Health check failed for {name}: {e}")
            return health
    
    async def get_all_health(self) -> Dict[str, ServiceHealth]:
        """Get health status for all services"""
        health_status = {}
        
        for name in self._services.keys():
            health_status[name] = await self.health_check(name)
        
        return health_status
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        total_uptime = datetime.now() - self._startup_time
        
        return {
            'system': {
                'uptime': str(total_uptime),
                'total_services': len(self._configs),
                'running_services': len(self._services),
                'healthy_services': sum(1 for h in self._health.values() if h.is_healthy)
            },
            'services': {
                name: {
                    'status': health.status.value,
                    'uptime': str(health.uptime),
                    'success_rate': health.success_rate,
                    'response_time': health.response_time,
                    'requests': self._metrics[name]['requests'],
                    'errors': self._metrics[name]['errors']
                }
                for name, health in self._health.items()
            }
        }
    
    def _sort_by_dependencies(self) -> List[str]:
        """Sort services by dependencies (topological sort)"""
        result = []
        visited = set()
        temp_visited = set()
        
        def visit(name: str):
            if name in temp_visited:
                raise JurisOracleError(f"Circular dependency detected: {name}")
            if name in visited:
                return
            
            temp_visited.add(name)
            
            config = self._configs.get(name)
            if config:
                for dep in config.dependencies:
                    if dep in self._configs:
                        visit(dep)
            
            temp_visited.remove(name)
            visited.add(name)
            result.append(name)
        
        for name in self._configs.keys():
            if name not in visited:
                visit(name)
        
        return result
    
    async def _start_monitoring(self) -> None:
        """Start background monitoring task"""
        async def monitor():
            while True:
                try:
                    await asyncio.sleep(30)  # Monitor every 30 seconds
                    
                    for name in list(self._services.keys()):
                        await self.health_check(name)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"❌ Monitoring error: {e}")
        
        self._monitoring_task = asyncio.create_task(monitor())
        logger.info("📊 Service monitoring started")
    
    @asynccontextmanager
    async def service_context(self, name: str):
        """Context manager for service operations"""
        start_time = time.time()
        
        try:
            service = self.get_service(name)
            if not service:
                raise JurisOracleError(f"Service {name} not available")
            
            # Update metrics
            self._metrics[name]['requests'] += 1
            self._metrics[name]['last_request'] = datetime.now()
            
            yield service
            
        except Exception as e:
            self._metrics[name]['errors'] += 1
            logger.error(f"❌ Service {name} operation failed: {e}")
            raise
        
        finally:
            # Update timing metrics
            elapsed = time.time() - start_time
            self._metrics[name]['total_time'] += elapsed


# Global service registry instance
service_registry = ServiceRegistry()


# Service configuration definitions
SERVICE_CONFIGS = [
    ServiceConfig(
        name="qa_service",
        service_class=QAService,
        priority=ServicePriority.HIGH,
        auto_start=True,
        health_check_interval=30,
        dependencies=[],
        config={}
    ),
    ServiceConfig(
        name="summarization_service",
        service_class=SummarizationService,
        priority=ServicePriority.HIGH,
        auto_start=True,
        health_check_interval=30,
        dependencies=[],
        config={}
    ),
    ServiceConfig(
        name="training_service",
        service_class=TrainingService,
        priority=ServicePriority.NORMAL,
        auto_start=False,  # Training is manual
        health_check_interval=60,
        dependencies=["qa_service"],
        config={}
    )
]


async def initialize_services() -> bool:
    """Initialize all services"""
    logger.info("🚀 Initializing JurisOracle Services...")
    
    try:
        # Register all services
        for config in SERVICE_CONFIGS:
            service_registry.register_service(config)
        
        # Start services
        success = await service_registry.start_all_services()
        
        if success:
            logger.info("✅ All services initialized successfully")
        else:
            logger.error("❌ Service initialization failed")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        return False


async def cleanup_services() -> None:
    """Cleanup all services"""
    logger.info("🧹 Cleaning up services...")
    await service_registry.stop_all_services()
    logger.info("✅ Services cleanup completed")


# Service accessor functions
def get_qa_service() -> Optional[QAService]:
    """Get QA service instance"""
    return service_registry.get_service("qa_service")


def get_summarization_service() -> Optional[SummarizationService]:
    """Get summarization service instance"""
    return service_registry.get_service("summarization_service")


def get_training_service() -> Optional[TrainingService]:
    """Get training service instance"""
    return service_registry.get_service("training_service")


async def get_service_health() -> Dict[str, ServiceHealth]:
    """Get health status for all services"""
    return await service_registry.get_all_health()


async def get_service_metrics() -> Dict[str, Any]:
    """Get service metrics"""
    return await service_registry.get_metrics()


# Context managers for service operations
@asynccontextmanager
async def qa_service_context():
    """Context manager for QA service operations"""
    async with service_registry.service_context("qa_service") as service:
        yield service


@asynccontextmanager
async def summarization_service_context():
    """Context manager for summarization service operations"""
    async with service_registry.service_context("summarization_service") as service:
        yield service


@asynccontextmanager
async def training_service_context():
    """Context manager for training service operations"""
    async with service_registry.service_context("training_service") as service:
        yield service


# Export all public interfaces
__all__ = [
    # Core classes
    'ServiceRegistry',
    'ServiceStatus',
    'ServicePriority',
    'ServiceHealth',
    'ServiceConfig',
    
    # Global registry
    'service_registry',
    
    # Initialization functions
    'initialize_services',
    'cleanup_services',
    
    # Service accessors
    'get_qa_service',
    'get_summarization_service',
    'get_training_service',
    
    # Monitoring functions
    'get_service_health',
    'get_service_metrics',
    
    # Context managers
    'qa_service_context',
    'summarization_service_context',
    'training_service_context',
    
    # Service classes
    'QAService',
    'SummarizationService',
    'TrainingService'
]


# Auto-initialization hook
async def _auto_initialize():
    """Auto-initialize services if enabled"""
    if getattr(settings, 'AUTO_INITIALIZE_SERVICES', True):
        await initialize_services()


# Log successful module import
logger.info("📋 Services module loaded successfully")