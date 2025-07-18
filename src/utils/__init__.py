"""
JurisOracle Utilities Package
===========================

Este pacote serve como um repositório centralizado para funcionalidades utilitárias
e auxiliares diversas, essenciais para o funcionamento robusto e eficiente da
aplicação JurisOracle. Ele agrupa código reutilizável que não pertence a um
domínio de negócio específico, mas que é amplamente utilizado em várias partes
do sistema.

O principal objetivo é promover a reusabilidade, a consistência e a manutenibilidade
do código, evitando duplicação e garantindo que operações comuns, como validação
de dados, processamento de texto e tratamento de exceções, sejam tratadas de forma
uniforme em toda a aplicação.

Conteúdo Detalhado:
------------------
- `validators`: Oferece um conjunto de funções para garantir a integridade e o formato
  correto de dados de entrada, prevenindo erros e vulnerabilidades. Inclui validações
  para e-mails, senhas, tipos numéricos, datas e URLs.
- `text_processing`: Contém ferramentas para limpeza, normalização e manipulação de
  strings, como remoção de pontuação, caracteres especiais, diacríticos e espaços
  extras, além de funções para análise textual básica.
- `exceptions`: Define classes de exceção personalizadas para a aplicação, permitindo
  um tratamento de erros mais granular, semântico e fácil de depurar, distinguindo
  entre diferentes tipos de falhas operacionais e de negócio.

As funcionalidades mais importantes e frequentemente utilizadas são expostas
diretamente no nível superior do pacote 'utils' para facilitar a importação
e o uso, tornando a API mais conveniente para os desenvolvedores.

Exemplos de Importação:
----------------------
Para importar funcionalidades diretamente expostas pelo pacote:
    >>> from utils import is_valid_email, clean_text, NotFoundException

Para importar um submódulo inteiro ou funcionalidades específicas de um submódulo:
    >>> import utils.validators
    >>> is_strong = utils.validators.is_strong_password("MyStrongP@ssw0rd")

    >>> from utils.exceptions import ApplicationException
    >>> raise ApplicationException("Ocorreu um erro genérico na aplicação.")

Author: JurisOracle Team
Version: 1.0.0
License: MIT
"""

import logging

# Configuração de logging para o pacote 'utils'.
# É uma prática recomendada para bibliotecas e pacotes Python usar `NullHandler`.
# Isso impede que mensagens como "No handlers could be found for logger..." sejam
# exibidas caso o aplicativo principal não configure um handler de logging para
# este pacote. As mensagens de log deste pacote serão processadas apenas se o
# aplicativo consumidor configurar um handler de logging para o logger raiz ou
# para o logger específico deste pacote (`utils`).
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Variáveis de metadados do pacote
# Estas variáveis são padrões na comunidade Python e fornecem informações essenciais
# sobre o pacote, úteis para ferramentas de empacotamento, documentação e identificação.
__version__ = "1.0.0"
__author__ = "JurisOracle Team"
__email__ = "dev@jurisoracle.com"
__license__ = "MIT"
__status__ = "Production" # Pode ser "Development", "Beta", "Production", etc.

# Bloco de importação de submódulos e exposição de APIs.
# Este bloco `try...except ImportError` é fundamental para a robustez do pacote.
# Ele garante que, mesmo que um submódulo específico esteja ausente (por exemplo,
# devido a uma instalação incompleta, corrupção de arquivos ou um erro durante o
# desenvolvimento/deploy), o pacote `utils` ainda possa ser importado, embora
# com funcionalidade reduzida. Isso evita falhas catastróficas na inicialização
# da aplicação e fornece mensagens de erro claras, indicando qual submódulo falhou.
try:
    # Importa os submódulos inteiros. Isso permite acesso granular, e.g., `utils.validators.is_valid_email`.
    from . import validators
    from . import text_processing
    from . import exceptions

    # Expondo funções e classes selecionadas diretamente no namespace do pacote 'utils'.
    # Esta prática, conhecida como "flattening" a API do pacote, permite que os
    # consumidores do pacote importem diretamente as funcionalidades mais comumente
    # usadas, como `from utils import is_valid_email`, sem a necessidade de especificar
    # o submódulo (`from utils.validators import is_valid_email`). Isso melhora a
    # usabilidade e a legibilidade do código cliente, mas requer atenção para evitar
    # colisões de nomes entre diferentes submódulos.

    # Do módulo validators.py
    from .validators import (
        ValidationError, # Exceção base para erros de validação
        is_valid_email,
        is_strong_password,
        is_valid_integer,
        is_valid_float,
        is_valid_string,
        is_valid_date,
        is_valid_url,
    )

    # Do módulo text_processing.py
    from .text_processing import (
        to_lowercase,
        remove_punctuation,
        remove_numbers,
        remove_extra_spaces,
        remove_diacritics,
        clean_text, # Função de limpeza de texto abrangente
        count_words,
        count_characters,
        get_unique_words,
    )

    # Do módulo exceptions.py
    from .exceptions import (
        ApplicationException, # Exceção base para a aplicação
        NotFoundException,
        ValidationException,
        AuthenticationException,
        AuthorizationException,
        ConflictException,
        ServiceUnavailableException,
        InternalServerErrorException,
        handle_exception_chain, # Utilitário para rastrear cadeias de exceção
        log_exception, # Utilitário para logar exceções de forma padronizada
    )

except ImportError as e:
    # Registra um erro detalhado se um submódulo não puder ser importado.
    # Isso é crucial para depuração em ambientes onde a estrutura do pacote pode estar comprometida.
    logging.error(f"Erro fatal ao importar um submódulo do pacote 'utils': {e}. "
                  "Verifique a integridade dos arquivos do pacote e suas dependências.")
    # Dependendo da criticidade, pode-se optar por re-levantar uma exceção mais específica
    # ou permitir que o pacote seja carregado parcialmente, mas com um aviso claro.
    # Por exemplo: raise RuntimeError(f"Falha na inicialização do pacote 'utils' devido a: {e}")

# Definição da variável __all__ para controlar o que é importado com 'from utils import *'.
# A variável `__all__` é uma convenção Python que define a API pública de um pacote
# quando a sintaxe `from package import *` é utilizada. Embora o uso de `import *`
# seja geralmente desencorajado em código de produção devido à poluição do namespace,
# definir `__all__` é uma boa prática para:
#   1. Clareza da API: Documenta explicitamente quais objetos são considerados parte
#      da interface pública do pacote.
#   2. Ferramentas: Ajuda IDEs e ferramentas de análise estática a entenderem melhor
#      a estrutura do pacote e a oferecerem autocompletar mais preciso.
#   3. Controle: Evita a importação acidental de objetos internos ou auxiliares que
#      não deveriam ser expostos.
__all__ = [
    # Metadados do pacote (opcional, mas útil para introspecção)
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__status__",

    # Módulos inteiros (para acesso mais granular, e.g., `utils.validators.is_valid_email`)
    "validators",
    "text_processing",
    "exceptions",

    # Funções e classes expostas diretamente (para acesso direto, e.g., `utils.is_valid_email`)

    # De validators.py
    "ValidationError",
    "is_valid_email",
    "is_strong_password",
    "is_valid_integer",
    "is_valid_float",
    "is_valid_string",
    "is_valid_date",
    "is_valid_url",

    # De text_processing.py
    "to_lowercase",
    "remove_punctuation",
    "remove_numbers",
    "remove_extra_spaces",
    "remove_diacritics",
    "clean_text",
    "count_words",
    "count_characters",
    "get_unique_words",

    # De exceptions.py
    "ApplicationException",
    "NotFoundException",
    "ValidationException",
    "AuthenticationException",
    "AuthorizationException",
    "ConflictException",
    "ServiceUnavailableException",
    "InternalServerErrorException",
    "handle_exception_chain",
    "log_exception",
]

# Mensagem de inicialização do pacote.
# Esta mensagem de log é útil para confirmar a correta inicialização do pacote
# durante o startup da aplicação ou em ambientes de desenvolvimento/depuração.
# Em produção, seu nível de log pode ser ajustado para evitar logs excessivos.
logging.info(f"Pacote 'utils' (v{__version__}) inicializado com sucesso.")
