"""
Validators Module - JurisOracle Utils
====================================

Este módulo fornece um conjunto abrangente de funções de validação reutilizáveis
para dados comuns em aplicações Python, especialmente adequadas para sistemas
jurídicos e aplicações empresariais.

Todas as funções de validação retornam True em caso de sucesso ou levantam
ValidationError com mensagens descritivas em caso de falha.

Exemplo de uso:
    >>> from utils.validators import is_valid_email, ValidationError
    >>> 
    >>> try:
    >>>     is_valid_email("usuario@exemplo.com")
    >>>     print("Email válido!")
    >>> except ValidationError as e:
    >>>     print(f"Email inválido: {e}")

Author: JurisOracle Team
Version: 1.0.0
"""

import re
import urllib.parse
from datetime import datetime
from typing import Union, Optional, Any


class ValidationError(Exception):
    """
    Exceção customizada para erros de validação.
    
    Esta exceção é levantada quando uma função de validação
    encontra dados que não atendem aos critérios especificados.
    
    Attributes:
        message (str): Mensagem descritiva do erro de validação
        field (str, optional): Nome do campo que falhou na validação
        value (Any, optional): Valor que causou a falha na validação
    """
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        """
        Inicializa a exceção ValidationError.
        
        Args:
            message: Mensagem descritiva do erro
            field: Nome do campo que falhou (opcional)
            value: Valor que causou o erro (opcional)
        """
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Retorna representação string da exceção."""
        if self.field:
            return f"Erro de validação no campo '{self.field}': {self.message}"
        return f"Erro de validação: {self.message}"


def is_valid_email(email: str) -> bool:
    """
    Valida o formato de um endereço de e-mail usando regex robusta.
    
    Esta função utiliza uma expressão regular otimizada que segue
    as especificações RFC 5322 de forma prática, evitando complexidade
    excessiva que poderia levar a vulnerabilidades ReDoS.
    
    Args:
        email (str): O endereço de e-mail a ser validado
        
    Returns:
        bool: True se o email for válido
        
    Raises:
        ValidationError: Se o email for inválido ou None
        TypeError: Se o parâmetro não for uma string
        
    Examples:
        >>> is_valid_email("usuario@exemplo.com")
        True
        >>> is_valid_email("email.invalido")
        ValidationError: Formato de email inválido
    """
    if not isinstance(email, str):
        raise TypeError("Email deve ser uma string")
    
    if not email or not email.strip():
        raise ValidationError("Email não pode ser vazio", field="email", value=email)
    
    # Regex otimizada para validação de email (evita ReDoS)
    # Padrão baseado em RFC 5322 simplificado
    email_pattern = re.compile(
        r'^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?@[a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,}$',
        re.IGNORECASE
    )
    
    email = email.strip()
    
    # Validações básicas de comprimento
    if len(email) > 254:  # RFC 5321 limite
        raise ValidationError("Email muito longo (máximo 254 caracteres)", field="email", value=email)
    
    if '@' not in email:
        raise ValidationError("Email deve conter o símbolo '@'", field="email", value=email)
    
    local_part, domain = email.rsplit('@', 1)
    
    # Validar parte local (antes do @)
    if len(local_part) > 64:  # RFC 5321 limite
        raise ValidationError("Parte local do email muito longa (máximo 64 caracteres)", field="email", value=email)
    
    # Validar domínio
    if len(domain) > 253:
        raise ValidationError("Domínio do email muito longo (máximo 253 caracteres)", field="email", value=email)
    
    if not email_pattern.match(email):
        raise ValidationError("Formato de email inválido", field="email", value=email)
    
    return True


def is_strong_password(
    password: str,
    min_length: int = 8,
    require_uppercase: bool = True,
    require_lowercase: bool = True,
    require_digit: bool = True,
    require_special_char: bool = True
) -> bool:
    """
    Verifica a força de uma senha com critérios configuráveis.
    
    Esta função valida se uma senha atende aos critérios de segurança
    especificados, incluindo comprimento mínimo e tipos de caracteres.
    
    Args:
        password (str): A senha a ser validada
        min_length (int): Comprimento mínimo da senha (padrão: 8)
        require_uppercase (bool): Exigir ao menos uma letra maiúscula (padrão: True)
        require_lowercase (bool): Exigir ao menos uma letra minúscula (padrão: True)
        require_digit (bool): Exigir ao menos um dígito (padrão: True)
        require_special_char (bool): Exigir ao menos um caractere especial (padrão: True)
        
    Returns:
        bool: True se a senha for forte o suficiente
        
    Raises:
        ValidationError: Se a senha não atender aos critérios
        TypeError: Se o parâmetro não for uma string
        
    Examples:
        >>> is_strong_password("MinhaSenh@123")
        True
        >>> is_strong_password("senha")
        ValidationError: Senha deve ter pelo menos 8 caracteres
    """
    if not isinstance(password, str):
        raise TypeError("Senha deve ser uma string")
    
    if not password:
        raise ValidationError("Senha não pode ser vazia", field="password")
    
    errors = []
    
    # Verificar comprimento mínimo
    if len(password) < min_length:
        errors.append(f"deve ter pelo menos {min_length} caracteres")
    
    # Verificar caracteres maiúsculos
    if require_uppercase and not re.search(r'[A-Z]', password):
        errors.append("deve conter pelo menos uma letra maiúscula")
    
    # Verificar caracteres minúsculos
    if require_lowercase and not re.search(r'[a-z]', password):
        errors.append("deve conter pelo menos uma letra minúscula")
    
    # Verificar dígitos
    if require_digit and not re.search(r'\d', password):
        errors.append("deve conter pelo menos um dígito")
    
    # Verificar caracteres especiais
    if require_special_char and not re.search(r'[!@#$%^&*()_+\-=\[\]{};:"\|,.<>?]', password):
        errors.append("deve conter pelo menos um caractere especial (!@#$%^&*()_+-=[]{}|;:,.<>?)")
    
    if errors:
        error_message = "Senha " + " e ".join(errors)
        raise ValidationError(error_message, field="password", value="[OCULTA]")
    
    return True


def is_valid_integer(
    value: Any,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None
) -> bool:
    """
    Valida se um valor é um inteiro e opcionalmente verifica se está dentro de um range.
    
    Esta função aceita strings numéricas, floats sem parte decimal e inteiros,
    convertendo-os conforme necessário para validação.
    
    Args:
        value: O valor a ser validado
        min_val (int, optional): Valor mínimo permitido (inclusive)
        max_val (int, optional): Valor máximo permitido (inclusive)
        
    Returns:
        bool: True se o valor for um inteiro válido
        
    Raises:
        ValidationError: Se o valor não for um inteiro válido ou estiver fora do range
        
    Examples:
        >>> is_valid_integer("123")
        True
        >>> is_valid_integer(45.0)
        True
        >>> is_valid_integer("abc")
        ValidationError: Valor deve ser um número inteiro
    """
    if value is None:
        raise ValidationError("Valor não pode ser None", field="integer", value=value)
    
    # Tentar converter para inteiro
    try:
        if isinstance(value, str):
            if not value.strip():
                raise ValidationError("String não pode ser vazia", field="integer", value=value)
            # Remover espaços em branco
            value = value.strip()
            # Verificar se contém apenas dígitos (e opcionalmente sinal)
            if not re.match(r'^[+-]?\d+$', value):
                raise ValidationError("Valor deve ser um número inteiro", field="integer", value=value)
            int_value = int(value)
        elif isinstance(value, float):
            # Verificar se o float não tem parte decimal
            if not value.is_integer():
                raise ValidationError("Valor float deve ser um número inteiro (sem casas decimais)", field="integer", value=value)
            int_value = int(value)
        elif isinstance(value, int):
            int_value = value
        else:
            raise ValidationError(f"Tipo de valor inválido: {type(value).__name__}", field="integer", value=value)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Não foi possível converter para inteiro: {str(e)}", field="integer", value=value)
    
    # Verificar range mínimo
    if min_val is not None and int_value < min_val:
        raise ValidationError(f"Valor deve ser maior ou igual a {min_val}", field="integer", value=value)
    
    # Verificar range máximo
    if max_val is not None and int_value > max_val:
        raise ValidationError(f"Valor deve ser menor ou igual a {max_val}", field="integer", value=value)
    
    return True


def is_valid_float(
    value: Any,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> bool:
    """
    Valida se um valor é um float e opcionalmente verifica se está dentro de um range.
    
    Esta função aceita strings numéricas, inteiros e floats,
    convertendo-os conforme necessário para validação.
    
    Args:
        value: O valor a ser validado
        min_val (float, optional): Valor mínimo permitido (inclusive)
        max_val (float, optional): Valor máximo permitido (inclusive)
        
    Returns:
        bool: True se o valor for um float válido
        
    Raises:
        ValidationError: Se o valor não for um float válido ou estiver fora do range
        
    Examples:
        >>> is_valid_float("123.45")
        True
        >>> is_valid_float(67)
        True
        >>> is_valid_float("abc")
        ValidationError: Valor deve ser um número
    """
    if value is None:
        raise ValidationError("Valor não pode ser None", field="float", value=value)
    
    # Tentar converter para float
    try:
        if isinstance(value, str):
            if not value.strip():
                raise ValidationError("String não pode ser vazia", field="float", value=value)
            # Remover espaços em branco
            value = value.strip()
            # Verificar formato numérico válido
            if not re.match(r'^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$', value):
                raise ValidationError("Valor deve ser um número", field="float", value=value)
            float_value = float(value)
        elif isinstance(value, (int, float)):
            float_value = float(value)
        else:
            raise ValidationError(f"Tipo de valor inválido: {type(value).__name__}", field="float", value=value)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Não foi possível converter para float: {str(e)}", field="float", value=value)
    
    # Verificar se é um número válido (não NaN ou infinito)
    if not isinstance(float_value, (int, float)) or float_value != float_value:  # NaN check
        raise ValidationError("Valor não é um número válido", field="float", value=value)
    
    if float_value == float('inf') or float_value == float('-inf'):
        raise ValidationError("Valor não pode ser infinito", field="float", value=value)
    
    # Verificar range mínimo
    if min_val is not None and float_value < min_val:
        raise ValidationError(f"Valor deve ser maior ou igual a {min_val}", field="float", value=value)
    
    # Verificar range máximo
    if max_val is not None and float_value > max_val:
        raise ValidationError(f"Valor deve ser menor ou igual a {max_val}", field="float", value=value)
    
    return True


def is_valid_string(
    text: str,
    min_length: int = 1,
    max_length: Optional[int] = None,
    allow_empty: bool = False,
    allowed_chars: Optional[str] = None
) -> bool:
    """
    Valida strings quanto ao comprimento, se é vazio e se contém apenas caracteres permitidos.
    
    Esta função oferece validação flexível de strings com múltiplos critérios
    configuráveis para atender diferentes necessidades de validação.
    
    Args:
        text (str): A string a ser validada
        min_length (int): Comprimento mínimo da string (padrão: 1)
        max_length (int, optional): Comprimento máximo da string
        allow_empty (bool): Se strings vazias são permitidas (padrão: False)
        allowed_chars (str, optional): String contendo caracteres permitidos
        
    Returns:
        bool: True se a string for válida
        
    Raises:
        ValidationError: Se a string não atender aos critérios
        TypeError: Se o parâmetro não for uma string
        
    Examples:
        >>> is_valid_string("Texto válido")
        True
        >>> is_valid_string("", allow_empty=True)
        True
        >>> is_valid_string("abc123", allowed_chars="abcdefghijklmnopqrstuvwxyz0123456789")
        True
    """
    if not isinstance(text, str):
        raise TypeError("Valor deve ser uma string")
    
    # Verificar se string vazia é permitida
    if not text and not allow_empty:
        raise ValidationError("String não pode ser vazia", field="string", value=text)
    
    # Se string vazia é permitida e o texto está vazio, retornar True
    if not text and allow_empty:
        return True
    
    # Verificar comprimento mínimo
    if len(text) < min_length:
        raise ValidationError(f"String deve ter pelo menos {min_length} caracteres", field="string", value=text)
    
    # Verificar comprimento máximo
    if max_length is not None and len(text) > max_length:
        raise ValidationError(f"String deve ter no máximo {max_length} caracteres", field="string", value=text)
    
    # Verificar caracteres permitidos
    if allowed_chars is not None:
        invalid_chars = set(text) - set(allowed_chars)
        if invalid_chars:
            invalid_list = sorted(list(invalid_chars))
            raise ValidationError(
                f"String contém caracteres não permitidos: {invalid_list}",
                field="string",
                value=text
            )
    
    return True


def is_valid_date(date_string: str, date_format: str = '%Y-%m-%d') -> bool:
    """
    Valida se uma string é uma data no formato especificado.
    
    Esta função utiliza o módulo datetime para parsing seguro de datas,
    garantindo que a data seja válida (por exemplo, 31 de fevereiro será rejeitado).
    
    Args:
        date_string (str): A string de data a ser validada
        date_format (str): Formato esperado da data (padrão: '%Y-%m-%d')
        
    Returns:
        bool: True se a data for válida
        
    Raises:
        ValidationError: Se a data for inválida ou não estiver no formato correto
        TypeError: Se o parâmetro não for uma string
        
    Examples:
        >>> is_valid_date("2023-12-25")
        True
        >>> is_valid_date("25/12/2023", "%d/%m/%Y")
        True
        >>> is_valid_date("2023-02-30")
        ValidationError: Data inválida
    """
    if not isinstance(date_string, str):
        raise TypeError("Data deve ser uma string")
    
    if not date_string or not date_string.strip():
        raise ValidationError("String de data não pode ser vazia", field="date", value=date_string)
    
    date_string = date_string.strip()
    
    try:
        # Tentar fazer o parsing da data
        parsed_date = datetime.strptime(date_string, date_format)
        
        # Verificar se a data parseada corresponde à string original
        # Isso evita problemas como "2023-02-30" ser convertido para "2023-03-02"
        reformatted = parsed_date.strftime(date_format)
        if reformatted != date_string:
            raise ValidationError("Data inválida (exemplo: 30 de fevereiro)", field="date", value=date_string)
        
    except ValueError as e:
        raise ValidationError(f"Formato de data inválido. Esperado: {date_format}. Erro: {str(e)}", field="date", value=date_string)
    
    return True


def is_valid_url(url: str) -> bool:
    """
    Valida o formato de uma URL, incluindo esquemas comuns (http, https).
    
    Esta função utiliza o módulo urllib.parse para validação robusta de URLs,
    verificando componentes essenciais como esquema, netloc e formato geral.
    
    Args:
        url (str): A URL a ser validada
        
    Returns:
        bool: True se a URL for válida
        
    Raises:
        ValidationError: Se a URL for inválida
        TypeError: Se o parâmetro não for uma string
        
    Examples:
        >>> is_valid_url("https://www.exemplo.com")
        True
        >>> is_valid_url("http://exemplo.com/path?param=value")
        True
        >>> is_valid_url("exemplo.com")
        ValidationError: URL deve incluir o protocolo (http:// ou https://)
    """
    if not isinstance(url, str):
        raise TypeError("URL deve ser uma string")
    
    if not url or not url.strip():
        raise ValidationError("URL não pode ser vazia", field="url", value=url)
    
    url = url.strip()
    
    # Verificar comprimento básico
    if len(url) > 2048:  # Limite prático para URLs
        raise ValidationError("URL muito longa (máximo 2048 caracteres)", field="url", value=url)
    
    try:
        # Parse da URL
        parsed = urllib.parse.urlparse(url)
        
        # Verificar se tem esquema
        if not parsed.scheme:
            raise ValidationError("URL deve incluir o protocolo (http:// ou https://)", field="url", value=url)
        
        # Verificar esquemas permitidos
        allowed_schemes = {'http', 'https', 'ftp', 'ftps'}
        if parsed.scheme.lower() not in allowed_schemes:
            raise ValidationError(
                f"Protocolo não suportado: {parsed.scheme}. Protocolos permitidos: {', '.join(allowed_schemes)}",
                field="url",
                value=url
            )
        
        # Verificar se tem netloc (domínio)
        if not parsed.netloc:
            raise ValidationError("URL deve incluir um domínio válido", field="url", value=url)
        
        # Verificar formato básico do domínio
        netloc = parsed.netloc.lower()
        
        # Remover porta se existir
        if ':' in netloc:
            domain_part = netloc.split(':')[0]
            port_part = netloc.split(':')[1]
            # Verificar se porta é numérica
            if not port_part.isdigit() or not (1 <= int(port_part) <= 65535):
                raise ValidationError("Porta inválida na URL", field="url", value=url)
        else:
            domain_part = netloc
        
        # Verificar formato básico do domínio
        if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?$', domain_part):
            raise ValidationError("Formato de domínio inválido", field="url", value=url)
        
        # Verificar se tem pelo menos um ponto no domínio (exceto localhost e IPs)
        if not re.match(r'^\d+\.\d+\.\d+\.\d+$', domain_part) and domain_part != 'localhost':
            if '.' not in domain_part:
                raise ValidationError("Domínio deve conter pelo menos um ponto", field="url", value=url)
    
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Formato de URL inválido: {str(e)}", field="url", value=url)
    
    return True


# Exemplo de uso e testes
if __name__ == "__main__":
    """
    Exemplos práticos de uso das funções de validação.
    
    Este bloco demonstra como usar cada função de validação,
    incluindo casos de sucesso e tratamento de erros.
    """
    
    print("🔍 DEMONSTRAÇÃO DO MÓDULO VALIDATORS")
    print("=" * 50)
    
    # Função auxiliar para executar testes
    def test_validator(func, test_cases, description):
        """Executa testes para uma função de validação."""
        print(f"\n📋 {description}")
        print("-" * 30)
        
        for i, (args, kwargs, expected_result, case_description) in enumerate(test_cases, 1):
            try:
                if isinstance(args, tuple):
                    result = func(*args, **kwargs)
                else:
                    result = func(args, **kwargs)
                
                if expected_result:
                    print(f"✅ Teste {i}: {case_description} - SUCESSO")
                else:
                    print(f"❌ Teste {i}: {case_description} - ERRO: Deveria ter falhado")
            except ValidationError as e:
                if not expected_result:
                    print(f"✅ Teste {i}: {case_description} - FALHA ESPERADA: {e.message}")
                else:
                    print(f"❌ Teste {i}: {case_description} - ERRO INESPERADO: {e.message}")
            except Exception as e:
                print(f"❌ Teste {i}: {case_description} - ERRO CRÍTICO: {e}")
    
    # Testes de validação de email
    email_tests = [
        ("usuario@exemplo.com", {}, True, "Email válido simples"),
        ("test.email+tag@exemplo.com.br", {}, True, "Email com caracteres especiais"),
        ("usuario@sub.exemplo.com", {}, True, "Email com subdomínio"),
        ("email_invalido", {}, False, "Email sem @"),
        ("@exemplo.com", {}, False, "Email sem parte local"),
        ("usuario@", {}, False, "Email sem domínio"),
        ("", {}, False, "Email vazio"),
        ("a" * 65 + "@exemplo.com", {}, False, "Parte local muito longa"),
    ]
    test_validator(is_valid_email, email_tests, "Validação de Email")
    
    # Testes de validação de senha
    password_tests = [
        ("MinhaSenh@123", {}, True, "Senha forte padrão"),
        ("senha", {}, False, "Senha muito simples"),
        ("SOMENTEMAIUSCULA123!", {}, False, "Sem minúsculas"),
        ("somenteminus123!", {}, False, "Sem maiúsculas"),
        ("MinhaSenh", {}, False, "Sem números e caracteres especiais"),
        ("Abc@1", {"min_length": 5}, True, "Senha curta com critérios relaxados"),
        ("simples", {"require_uppercase": False, "require_digit": False, "require_special_char": False}, True, "Critérios relaxados"),
    ]
    test_validator(is_strong_password, password_tests, "Validação de Senha")
    
    # Testes de validação de inteiro
    integer_tests = [
        (123, {}, True, "Inteiro positivo"),
        (-456, {}, True, "Inteiro negativo"),
        ("789", {}, True, "String numérica"),
        (45.0, {}, True, "Float sem decimais"),
        ("abc", {}, False, "String não numérica"),
        (45.7, {}, False, "Float com decimais"),
        (100, {"min_val": 50, "max_val": 150}, True, "Inteiro no range"),
        (200, {"min_val": 50, "max_val": 150}, False, "Inteiro fora do range"),
    ]
    test_validator(is_valid_integer, integer_tests, "Validação de Inteiro")
    
    # Testes de validação de float
    float_tests = [
        (123.45, {}, True, "Float válido"),
        ("67.89", {}, True, "String float"),
        (100, {}, True, "Inteiro como float"),
        ("abc", {}, False, "String não numérica"),
        (50.5, {"min_val": 10.0, "max_val": 100.0}, True, "Float no range"),
        (150.5, {"min_val": 10.0, "max_val": 100.0}, False, "Float fora do range"),
    ]
    test_validator(is_valid_float, float_tests, "Validação de Float")
    
    # Testes de validação de string
    string_tests = [
        ("Texto válido", {}, True, "String válida padrão"),
        ("", {"allow_empty": True}, True, "String vazia permitida"),
        ("", {"allow_empty": False}, False, "String vazia não permitida"),
        ("abc", {"min_length": 5}, False, "String muito curta"),
        ("texto longo", {"max_length": 5}, False, "String muito longa"),
        ("abc123", {"allowed_chars": "abcdefghijklmnopqrstuvwxyz0123456789"}, True, "Caracteres permitidos"),
        ("abc@123", {"allowed_chars": "abcdefghijklmnopqrstuvwxyz0123456789"}, False, "Caracteres não permitidos"),
    ]
    test_validator(is_valid_string, string_tests, "Validação de String")
    
    # Testes de validação de data
    date_tests = [
        ("2023-12-25", {}, True, "Data ISO válida"),
        ("25/12/2023", {"date_format": "%d/%m/%Y"}, True, "Data brasileira válida"),
        ("2023-02-30", {}, False, "Data inválida (30 de fevereiro)"),
        ("2023-13-01", {}, False, "Mês inválido"),
        ("abc", {}, False, "String não é data"),
        ("", {}, False, "String vazia"),
    ]
    test_validator(is_valid_date, date_tests, "Validação de Data")
    
    # Testes de validação de URL
    url_tests = [
        ("https://www.exemplo.com", {}, True, "URL HTTPS válida"),
        ("http://exemplo.com/path?param=value", {}, True, "URL HTTP com parâmetros"),
        ("https://sub.exemplo.com:8080", {}, True, "URL com porta"),
        ("exemplo.com", {}, False, "URL sem protocolo"),
        ("http://", {}, False, "URL sem domínio"),
        ("https://exemplo", {}, False, "Domínio sem TLD"),
        ("", {}, False, "URL vazia"),
    ]
    test_validator(is_valid_url, url_tests, "Validação de URL")
    
    print("\n🎉 DEMONSTRAÇÃO CONCLUÍDA!")
    print("=" * 50)
    print("📝 Todas as funções de validação foram testadas com casos de sucesso e falha.")
    print("💡 Use estas funções em suas aplicações para garantir a integridade dos dados.")