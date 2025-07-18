"""
JurisOracle Main Application Package (src)
========================================

Este é o pacote raiz da aplicação JurisOracle, representando o coração da lógica
de negócio. Ele organiza e expõe os principais componentes e funcionalidades do
sistema de forma estruturada e modular, facilitando a manutenção, escalabilidade
e testabilidade.

O pacote 'src' atua como o ponto de entrada principal para a lógica de negócio,
contendo submódulos dedicados a:
- **core**: Lógica central da aplicação e fluxo principal.
- **config**: Gerenciamento de configurações e variáveis de ambiente.
- **models**: Definição de modelos de dados e esquemas.
- **services**: Implementação de regras de negócio e orquestração de operações.
- **training**: Módulos para treinamento, avaliação e gerenciamento de modelos de IA.
- **utils**: Funções utilitárias e auxiliares de uso geral.

Recursos Chave:
- **Controle de Versão**: Define a versão global da aplicação para rastreamento e empacotamento.
- **Estrutura Modular**: Expõe submódulos chave, promovendo a separação de responsabilidades.
- **API Pública Clara**: Utiliza `__all__` para definir explicitamente a interface pública do pacote.
- **Configuração de Logging**: Garante que o pacote se integre bem com sistemas de logging externos.

Autor: JurisOracle Development Team
Version: 0.1.0
License: MIT
"""

import logging
import sys

# 1. Configura o logger para o pacote 'src'.
# É uma boa prática para pacotes/bibliotecas Python adicionar um `NullHandler`
# ao seu logger principal. Isso garante que o pacote não emita avisos de
# "No handlers could be found for logger..." se o aplicativo principal que o
# consome não configurar um manipulador de log para este logger específico.
# As mensagens de log geradas por este pacote só serão processadas se o
# aplicativo consumidor configurar um handler apropriado (e.g., para o logger
# raiz ou para 'src') e um nível de log.
logging.getLogger(__name__).addHandler(logging.NullHandler())


# 2. Define __version__ para o controle de versão do pacote.
# Esta variável fornece a versão atual da aplicação/pacote. É um metadado
# crucial para:
# - Rastreamento de mudanças e releases.
# - Compatibilidade entre diferentes versões do pacote.
# - Uso por ferramentas de empacotamento (e.g., setuptools, poetry) e gerenciadores
#   de pacotes (e.g., pip).
# Pode ser acessado programaticamente via `importlib.metadata.version('jurisoracle')`
# (Python 3.8+) ou `pkg_resources.get_distribution('jurisoracle').version` (setuptools).
__version__ = '0.1.0'


# 3. Importa os submódulos principais da aplicação.
# O ponto '.' indica uma importação relativa, referenciando módulos dentro
# do mesmo pacote (neste caso, dentro de 'src').
# Estas importações tornam os submódulos acessíveis como `src.core`, `src.utils`, etc.,
# permitindo uma estrutura de projeto clara e a separação de responsabilidades.
# Cada submódulo é projetado para conter uma parte específica da lógica da aplicação.
from . import core      # Lógica de negócio central e orquestração de alto nível.
from . import config    # Carregamento e gerenciamento de configurações da aplicação (e.g., variáveis de ambiente, segredos).
from . import models    # Definição de modelos de dados, esquemas de banco de dados, ou modelos Pydantic.
from . import services  # Implementação de serviços de negócio que interagem com modelos e outras camadas.
from . import training  # Módulos relacionados ao treinamento, avaliação e inferência de modelos de Machine Learning/IA.
from . import utils     # Funções utilitárias e auxiliares que podem ser usadas em várias partes da aplicação.


# 4. Importa e expõe um exemplo de função de um submódulo (re-exportação).
# Para permitir que funções ou classes frequentemente usadas sejam importadas
# diretamente do pacote raiz (e.g., `from src import clean_text`), elas devem
# ser importadas do seu submódulo de origem e então re-exportadas neste
# `__init__.py`. Isso simplifica o uso para o consumidor do pacote.
try:
    # `clean_text` é um exemplo de função utilitária que poderia estar em `src.utils`.
    # Ela poderia ser usada para pré-processamento de texto em NLP, removendo
    # caracteres especiais, normalizando espaços, etc.
    from .utils import clean_text as exemplo_funcao
except ImportError as e:
    # Este bloco try-except é um padrão robusto para lidar com a ausência de um
    # módulo ou objeto específico. Pode ocorrer se o submódulo 'utils' não puder
    # ser carregado, se 'clean_text' não estiver definido, ou se houver um erro
    # de dependência. Em vez de falhar a importação de todo o pacote, registramos
    # um aviso e definimos a função como None, permitindo que o restante do pacote
    # seja carregado.
    logging.warning(f"Não foi possível importar 'exemplo_funcao' (clean_text) do submódulo 'utils': {e}")
    exemplo_funcao = None # Define como None se a importação falhar para evitar NameError


# 5. Inclui um __all__ explícito para especificar quais módulos ou objetos devem ser expostos.
# A variável `__all__` é uma lista de strings que define a API pública do pacote.
# Quando um usuário faz `from src import *` (importação "star"), apenas os nomes
# listados em `__all__` serão importados para o namespace do chamador.
# Isso é uma boa prática para:
# - **Evitar poluição de namespace**: Impede a importação acidental de objetos internos.
# - **Documentar a API pública**: Deixa claro quais partes do pacote são destinadas
#   ao uso externo e quais são detalhes de implementação interna.
# - **Melhorar a legibilidade**: Ajuda outros desenvolvedores a entender rapidamente
#   o que o pacote oferece.
__all__ = [
    '__version__',      # Expondo a versão do pacote
    'core',             # Expondo o submódulo 'core'
    'config',           # Expondo o submódulo 'config'
    'models',           # Expondo o submódulo 'models'
    'services',         # Expondo o submódulo 'services'
    'training',         # Expondo o submódulo 'training'
    'utils',            # Expondo o submódulo 'utils'
    'exemplo_funcao',   # Expondo a função 'exemplo_funcao' diretamente para conveniência
]


# Mensagem de log para indicar que o pacote 'src' foi inicializado.
# Esta mensagem é útil para depuração e para confirmar que o pacote principal
# foi carregado com sucesso durante a inicialização da aplicação.
logging.info(f"Pacote 'src' (v{__version__}) inicializado com sucesso.")
