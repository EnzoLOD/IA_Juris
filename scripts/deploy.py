#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚀 DEPLOY.PY - SISTEMA DE DEPLOY AUTOMATIZADO IA_JURIS
=====================================================

Este script automatiza o processo de deployment da aplicação IA_Juris.
Ele é projetado para ser robusto, flexível e aderir às melhores práticas
de segurança e operação.

Funcionalidades:
- Carregamento e validação de variáveis de ambiente necessárias.
- Build e push de imagens Docker.
- Execução de migrações de banco de dados.
- Deploy para ambientes de orquestração (via placeholders para comandos externos).
- Integração de modelos de IA treinados.
- Tratamento de exceções e logging detalhado.

Autor: Equipe IA_Juris
Data: 2025
Versão: 1.0.0

Uso:
  python deploy.py --env <ambiente>
  python deploy.py --env production --build-only
  python deploy.py --env staging --skip-build

Variáveis de Ambiente Necessárias:
  - DEPLOY_ENVIRONMENT: Define o ambiente de deploy (e.g., development, staging, production).

  - REGISTRY_URL: URL do registry de containers (e.g., docker.io/youruser, 123456789012.dkr.ecr.us-east-1.amazonaws.com).

  - REGISTRY_USERNAME: Usuário para autenticação no registry.

  - REGISTRY_PASSWORD: Senha para autenticação no registry.

  - DB_CONNECTION_STRING: String de conexão para o banco de dados (usada para migrações).

  - AWS_ACCESS_KEY_ID (opcional, para ECR/ECS/EKS)

  - AWS_SECRET_ACCESS_KEY (opcional, para ECR/ECS/EKS)
  
  - KUBECONFIG (opcional, para Kubernetes)
"""

# =============================================================================
# 1. IMPORTAÇÕES ORGANIZADAS
# =============================================================================

# Bibliotecas padrão
import os
import sys
import json
import logging
import argparse
import subprocess
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Bibliotecas de terceiros (com tratamento de exceções para dependências não críticas)
try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False
    print("⚠️ Aviso: `python-dotenv` não encontrado. Variáveis de ambiente devem ser carregadas manualmente.")

# Módulos do projeto
# Assumimos que 'setup_database.py' e 'train_model.py' estão na mesma pasta 'script'
# ou que seus caminhos podem ser inferidos/configurados.
# Para evitar circular imports e manter o deploy.py independente, não importamos
# diretamente setup_database ou train_model aqui, mas invocamos seus comandos via subprocess.
# Se esses scripts fossem funções, poderíamos importá-los, mas para um deploy,
# é mais comum invocá-los como executáveis independentes.


# =============================================================================
# 2. CONFIGURAÇÃO DE LOGGING
# =============================================================================

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configura o sistema de logging para o script de deploy.
    
    Args:
        level (str): Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL).

        log_file (Optional[str]): Caminho para o arquivo de log. Se None, loga apenas no console.
    
    Returns:
        logging.Logger: Instância do logger configurada.
    """
    logger = logging.getLogger("IA_Juris_Deploy")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove handlers existentes para evitar duplicação
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para arquivo
    if log_file:
        try:
            log_dir = Path(log_file).parent
            if log_dir: # Garante que não é apenas um nome de arquivo sem diretório
                log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Erro ao configurar log para arquivo '{log_file}': {e}")

    return logger

# Instância global do logger
logger = setup_logging()

# =============================================================================
# 3. UTILITIES
# =============================================================================

def get_env_var(name: str, required: bool = True, default: Optional[str] = None) -> str:
    """
    Obtém uma variável de ambiente, verificando se é obrigatória.
    
    Args:
        name (str): Nome da variável de ambiente.

        required (bool): Se a variável é obrigatória.

        default (Optional[str]): Valor padrão se a variável não for encontrada e não for obrigatória.
        
    Returns:
        str: Valor da variável de ambiente.
        
    Raises:
        ValueError: Se a variável obrigatória não for encontrada.
    """
    value = os.getenv(name)
    if value is None:
        if required:
            logger.error(f"Variável de ambiente obrigatória não definida: {name}")
            raise ValueError(f"Missing required environment variable: {name}")
        logger.warning(f"Variável de ambiente '{name}' não definida, usando valor padrão: '{default}'")
        return default
    return value

def run_command(command: Union[str, List[str]], cwd: Optional[str] = None, check: bool = True) -> str:
    """
    Executa um comando de shell e captura a saída.
    
    Args:
        command (Union[str, List[str]]): Comando a ser executado.

        cwd (Optional[str]): Diretório de trabalho atual para o comando.

        check (bool): Se True, levanta uma exceção se o comando retornar um status de erro.
        
    Returns:
        str: Saída padrão do comando.
        
    Raises:
        subprocess.CalledProcessError: Se o comando retornar um status de erro e `check` for True.
    """
    try:
        logger.debug(f"Executando comando: {' '.join(command) if isinstance(command, list) else command}")
        
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check,
            shell=isinstance(command, str) # Use shell=True se o comando for uma string
        )
        if result.stdout:
            logger.debug(f"Comando output: {result.stdout.strip()}")
        if result.stderr:
            logger.warning(f"Comando stderr: {result.stderr.strip()}")
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logger.error(f"Comando falhou com código {e.returncode}: {e.cmd}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error(f"Comando não encontrado: {command[0] if isinstance(command, list) else command.split(' ')[0]}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado ao executar comando: {e}")
        raise

# =============================================================================
# 4. FUNÇÕES DE DEPLOY ESPECÍFICAS
# =============================================================================

def _load_environment_variables(environment: str) -> Dict[str, str]:
    """
    Carrega e valida as variáveis de ambiente necessárias para o deploy.
    
    Args:
        environment (str): O ambiente de deploy (e.g., 'production', 'staging').
        
    Returns:
        Dict[str, str]: Dicionário de variáveis de ambiente validadas.
        
    Raises:
        ValueError: Se alguma variável de ambiente obrigatória estiver faltando.
    """
    logger.info(f"Carregando variáveis de ambiente para o ambiente: {environment.upper()}...")
    
    # Carregar .env se disponível
    if _DOTENV_AVAILABLE:
        dotenv_path = Path('.') / '.env'
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path)
            logger.info(".env carregado.")
        else:
            logger.warning(".env não encontrado. Verifique se as variáveis estão definidas no ambiente.")

    env_vars = {}
    
    # Variáveis de ambiente básicas
    env_vars['DEPLOY_ENVIRONMENT'] = environment
    env_vars['REGISTRY_URL'] = get_env_var('REGISTRY_URL')
    env_vars['REGISTRY_USERNAME'] = get_env_var('REGISTRY_USERNAME')
    env_vars['REGISTRY_PASSWORD'] = get_env_var('REGISTRY_PASSWORD')
    
    # Variáveis específicas do banco de dados (para migrações)
    env_vars['DB_CONNECTION_STRING'] = get_env_var('DB_CONNECTION_STRING')
    
    # Variáveis específicas de nuvem (ex: AWS)
    if environment == 'production': # Exemplo de variável específica de produção
        env_vars['AWS_ACCESS_KEY_ID'] = get_env_var('AWS_ACCESS_KEY_ID', required=False)
        env_vars['AWS_SECRET_ACCESS_KEY'] = get_env_var('AWS_SECRET_ACCESS_KEY', required=False)

    logger.info("Variáveis de ambiente carregadas com sucesso.")
    return env_vars

def _build_and_push_docker_image(env_vars: Dict[str, str]) -> str:
    """
    Constrói a imagem Docker da aplicação e a envia para o registry.
    
    Args:
        env_vars (Dict[str, str]): Variáveis de ambiente carregadas.
        
    Returns:
        str: A tag completa da imagem Docker construída e enviada.
        
    Raises:
        subprocess.CalledProcessError: Se os comandos Docker falharem.
    """
    logger.info("Iniciando build e push da imagem Docker...")
    
    repo_name = "ia_juris_app"
    version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
    image_tag = f"{env_vars['REGISTRY_URL']}/{repo_name}:{version}"
    
    try:
        logger.info("Autenticando no registry Docker...")
        run_command([
            "docker", "login", env_vars['REGISTRY_URL'],
            "-u", env_vars['REGISTRY_USERNAME'],
            "--password-stdin"
        ], input=env_vars['REGISTRY_PASSWORD']) # Passa a senha via stdin para segurança
        
        logger.info(f"Construindo imagem Docker: {image_tag}...")
        # Assume que o Dockerfile está na raiz do projeto
        run_command(["docker", "build", ".", "-t", image_tag])
        
        logger.info(f"Enviando imagem Docker: {image_tag}...")
        run_command(["docker", "push", image_tag])
        
        logger.info(f"✅ Imagem Docker construída e enviada com sucesso: {image_tag}")
        return image_tag
        
    except Exception as e:
        logger.error(f"Falha no build/push da imagem Docker: {e}")
        raise

def _run_database_migrations(env_vars: Dict[str, str]) -> None:
    """
    Executa as migrações de banco de dados usando o script setup_database.py.
    
    Args:
        env_vars (Dict[str, str]): Variáveis de ambiente carregadas.
        
    Raises:
        subprocess.CalledProcessError: Se o script de migração falhar.
    """
    logger.info("Executando migrações de banco de dados...")
    
    # Assumimos que setup_database.py está na pasta 'script'
    script_path = Path(__file__).parent / "setup_database.py"
    
    if not script_path.exists():
        logger.error(f"Script de setup de banco de dados não encontrado: {script_path}")
        raise FileNotFoundError(f"setup_database.py não encontrado em {script_path}")

    # Passa a string de conexão via variável de ambiente para o script de setup
    # Nota: para segurança, é melhor usar um arquivo de configuração para DB_CONNECTION_STRING
    # e passar o caminho para o setup_database.py
    
    # Definir um dicionário de variáveis de ambiente para o subprocesso
    sub_env = os.environ.copy()
    sub_env['DB_CONNECTION_STRING'] = env_vars['DB_CONNECTION_STRING']

    try:
        # A flag --init força a criação do banco/tabelas se não existirem
        # O script setup_database.py é robusto e cuidará das migrações
        run_command([sys.executable, str(script_path), "--init", "--log-level", logger.level_name], env=sub_env)
        
        # Opcional: popular dados iniciais se necessário, após as migrações
        # run_command([sys.executable, str(script_path), "--populate", "--log-level", logger.level_name], env=sub_env)
        
        logger.info("✅ Migrações de banco de dados executadas com sucesso.")
        
    except Exception as e:
        logger.error(f"Falha na execução das migrações de banco de dados: {e}")
        raise

def _deploy_to_orchestrator(image_tag: str, env_vars: Dict[str, str]) -> None:
    """
    Realiza o deploy da imagem Docker para o orquestrador (Kubernetes, ECS, etc.).
    
    Este é um placeholder e deve ser customizado para o seu ambiente.
    
    Args:
        image_tag (str): A tag completa da imagem Docker a ser implantada.

        env_vars (Dict[str, str]): Variáveis de ambiente carregadas.
        
    Raises:
        Exception: Se o deploy falhar.
    """
    logger.info(f"Iniciando deploy para o orquestrador ({env_vars['DEPLOY_ENVIRONMENT'].upper()})...")
    
    # Exemplo para Kubernetes (requer kubectl configurado)
    if env_vars['DEPLOY_ENVIRONMENT'] == 'production':
        logger.info("Executando deploy para Kubernetes Production...")
        try:
            # Substitua este comando pelo seu comando kubectl apply, helm upgrade, etc.
            # Você pode usar um template YAML e substituí-lo com a image_tag
            # Ex: sed -i "s|__IMAGE_TAG__|{image_tag}|g" k8s/deployment.yaml
            # Ex: run_command(["kubectl", "apply", "-f", "k8s/deployment.yaml", "-n", "production"])
            
            # Placeholder: Simular um deploy complexo
            logger.info(f"Simulando deploy de {image_tag} para Kubernetes...")
            time.sleep(5) # Simula o tempo de deploy
            logger.info("✅ Deploy para Kubernetes Production simulado com sucesso.")
            
        except Exception as e:
            logger.error(f"Falha no deploy para Kubernetes Production: {e}")
            raise
    
    elif env_vars['DEPLOY_ENVIRONMENT'] == 'staging':
        logger.info("Executando deploy para ECS Staging...")
        try:
            # Exemplo para AWS ECS
            # run_command([
            #     "aws", "ecs", "update-service",
            #     "--cluster", "your-staging-cluster",
            #     "--service", "your-staging-service",
            #     "--force-new-deployment",
            #     "--task-definition", "your-task-definition-family" # Você precisaria de um Task Definition com a nova imagem
            # ])
            
            # Placeholder: Simular um deploy
            logger.info(f"Simulando deploy de {image_tag} para ECS Staging...")
            time.sleep(3) # Simula o tempo de deploy
            logger.info("✅ Deploy para ECS Staging simulado com sucesso.")
            
        except Exception as e:
            logger.error(f"Falha no deploy para ECS Staging: {e}")
            raise
    else:
        logger.info("Nenhuma lógica de orquestrador definida para este ambiente. Deploy manual pode ser necessário.")
    
    logger.info("Processo de deploy para orquestrador concluído.")

def _integrate_ai_models(env_vars: Dict[str, str]) -> None:
    """
    Integra modelos de IA treinados (e.g., upload para S3/model registry, atualização de serviço).
    
    Este é um placeholder e deve ser customizado para o seu ambiente.
    
    Args:
        env_vars (Dict[str, str]): Variáveis de ambiente carregadas.
        
    Raises:
        Exception: Se a integração dos modelos falhar.
    """
    logger.info("Iniciando integração de modelos de IA...")
    
    # Exemplo: Upload de modelos para um bucket S3
    model_path = Path("./models/latest_model.pt") # Caminho hipotético do modelo
    if model_path.exists():
        logger.info(f"Encontrado modelo em {model_path}. Iniciando upload para S3...")
        try:
            # run_command([
            #     "aws", "s3", "cp", str(model_path),
            #     f"s3://your-model-bucket/{env_vars['DEPLOY_ENVIRONMENT']}/latest_model.pt"
            # ])
            logger.info(f"Simulando upload de modelo {model_path} para S3...")
            time.sleep(2)
            logger.info("✅ Upload de modelo para S3 simulado com sucesso.")
            
            # Após o upload, pode ser necessário atualizar um serviço de inferência
            # para carregar o novo modelo.
            # Ex: Fazer uma chamada de API para o serviço de inferência.
            
        except Exception as e:
            logger.error(f"Falha na integração do modelo de IA (upload para S3): {e}")
            raise
    else:
        logger.warning(f"Nenhum modelo encontrado em {model_path}. Pulando integração de modelos.")
    
    logger.info("Integração de modelos de IA concluída.")

def _health_check(service_url: str, max_retries: int = 10, retry_delay: int = 5) -> None:
    """
    Realiza uma verificação de saúde básica na URL do serviço.
    
    Args:
        service_url (str): URL do serviço a ser verificado.

        max_retries (int): Número máximo de tentativas.

        retry_delay (int): Atraso em segundos entre as tentativas.
        
    Raises:
        Exception: Se o serviço não estiver saudável após as retries.
    """
    logger.info(f"Executando verificação de saúde para: {service_url}...")
    
    # Requer a biblioteca 'requests'
    try:
        import requests
    except ImportError:
        logger.warning("Biblioteca 'requests' não encontrada. Verificação de saúde HTTP desabilitada.")
        return
        
    for i in range(max_retries):
        try:
            response = requests.get(service_url, timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ Verificação de saúde bem-sucedida para {service_url}.")
                return
            else:
                logger.warning(f"Verificação de saúde falhou com status {response.status_code}. Tentativa {i+1}/{max_retries}.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Verificação de saúde falhou: {e}. Tentativa {i+1}/{max_retries}.")
        
        time.sleep(retry_delay)
        
    logger.error(f"❌ Verificação de saúde falhou após {max_retries} tentativas para {service_url}.")
    raise Exception("Serviço não respondeu com status 200 OK.")

# =============================================================================
# 5. FUNÇÃO PRINCIPAL DEPLOY
# =============================================================================

def deploy(environment: str, skip_build: bool = False, build_only: bool = False,
           skip_migrations: bool = False, skip_orchestrator: bool = False,
           skip_ai_models: bool = False, skip_health_check: bool = False) -> None:
    """
    Função principal que orquestra o processo de deploy.
    
    Args:
        environment (str): O ambiente de destino do deploy (e.g., 'development', 'staging', 'production').

        skip_build (bool): Se True, pula a etapa de build e push da imagem Docker.

        build_only (bool): Se True, executa apenas a etapa de build e push da imagem Docker e sai.

        skip_migrations (bool): Se True, pula a execução de migrações de banco de dados.

        skip_orchestrator (bool): Se True, pula o deploy para o orquestrador.

        skip_ai_models (bool): Se True, pula a integração de modelos de IA.

        skip_health_check (bool): Se True, pula a verificação de saúde pós-deploy.
        
    Raises:
        Exception: Em caso de qualquer falha crítica no processo de deploy.
    """
    start_time = time.time()
    logger.info(f"🚀 Iniciando processo de deploy para o ambiente '{environment.upper()}'...")

    try:
        # 1. Carregar e validar variáveis de ambiente
        env_vars = _load_environment_variables(environment)
        
        # 2. Build e Push da Imagem Docker
        image_tag = None
        if not skip_build:
            image_tag = _build_and_push_docker_image(env_vars)
        else:
            logger.info("Build e push da imagem Docker pulados (--skip-build).")
            # Se pular o build, o image_tag deve vir de algum lugar (e.g., variável de ambiente)
            image_tag = get_env_var('EXISTING_IMAGE_TAG', required=False, default='latest')
            if image_tag == 'latest':
                logger.warning("Usando 'latest' como image_tag. Certifique-se de que é a imagem correta.")

        if build_only:
            logger.info("Modo 'build-only' ativado. Encerrando após build.")
            return
            
        # 3. Executar Migrações de Banco de Dados
        if not skip_migrations:
            _run_database_migrations(env_vars)
        else:
            logger.info("Migrações de banco de dados puladas (--skip-migrations).")

        # 4. Deploy para o Orquestrador
        if not skip_orchestrator:
            if image_tag: # Garante que temos uma tag de imagem para deploy
                _deploy_to_orchestrator(image_tag, env_vars)
            else:
                logger.error("Não há imagem Docker para implantar. O deploy para orquestrador será pulado.")
        else:
            logger.info("Deploy para orquestrador pulado (--skip-orchestrator).")

        # 5. Integração de Modelos de IA
        if not skip_ai_models:
            _integrate_ai_models(env_vars)
        else:
            logger.info("Integração de modelos de IA pulada (--skip-ai-models).")
            
        # 6. Verificação de Saúde Pós-Deploy
        service_url = get_env_var('SERVICE_HEALTH_URL', required=False) # URL do serviço para health check
        if not skip_health_check and service_url:
            _health_check(service_url)
        elif not service_url:
            logger.warning("SERVICE_HEALTH_URL não definida. Verificação de saúde pós-deploy pulada.")
        else:
            logger.info("Verificação de saúde pós-deploy pulada (--skip-health-check).")

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"🎉 Processo de deploy para '{environment.upper()}' concluído com sucesso em {duration:.2f} segundos!")

    except ValueError as ve:
        logger.critical(f"❌ Erro de configuração: {ve}")
        logger.critical("Deploy falhou devido a configuração ou variáveis de ambiente ausentes/inválidas.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.critical(f"❌ Erro de comando externo: {e.cmd} falhou com código {e.returncode}.")
        logger.critical("Verifique os logs acima para detalhes sobre a falha do comando.")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.critical(f"❌ Erro de arquivo: {e}. Certifique-se de que todos os scripts e arquivos necessários existam.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"❌ Erro inesperado durante o deploy: {e}")
        logger.critical(f"Traceback:\n{traceback.format_exc()}") # Loga o traceback completo para depuração
        sys.exit(1)

# =============================================================================
# 6. BLOCO PRINCIPAL DE EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🚀 Script de Deploy para IA_Juris",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python deploy.py --env development
  python deploy.py --env staging --skip-migrations --skip-health-check
  python deploy.py --env production --build-only
  python deploy.py --env production --log-level DEBUG --log-file deploy.log
        """
    )
    
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        choices=['development', 'staging', 'production', 'test'], # Adicione outros ambientes conforme necessário
        help="Ambiente de destino para o deploy."
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Pular a etapa de build e push da imagem Docker."
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Executar apenas o build e push da imagem Docker, e sair."
    )
    parser.add_argument(
        "--skip-migrations",
        action="store_true",
        help="Pular a execução de migrações de banco de dados."
    )
    parser.add_argument(
        "--skip-orchestrator",
        action="store_true",
        help="Pular o deploy para o orquestrador (Kubernetes, ECS, etc.)."
    )
    parser.add_argument(
        "--skip-ai-models",
        action="store_true",
        help="Pular a integração de modelos de IA."
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Pular a verificação de saúde pós-deploy."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nível de detalhe do log."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Caminho para o arquivo de log."
    )
    
    args = parser.parse_args()
    
    # Reconfigurar logger com base nos argumentos da linha de comando
    logger = setup_logging(args.log_level, args.log_file)
    
    # Chama a função principal de deploy com os argumentos fornecidos
    deploy(
        environment=args.env,
        skip_build=args.skip_build,
        build_only=args.build_only,
        skip_migrations=args.skip_migrations,
        skip_orchestrator=args.skip_orchestrator,
        skip_ai_models=args.skip_ai_models,
        skip_health_check=args.skip_health_check
    )