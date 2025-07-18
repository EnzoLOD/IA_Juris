"""
Text Processing Utilities - JurisOracle
=====================================

Este módulo fornece um conjunto completo e robusto de funções para tarefas comuns
de processamento de texto. Ele é otimizado para clareza, eficiência e documentação,
servindo como uma biblioteca de utilitários de texto confiável e reutilizável.

Funcionalidades principais:
- Normalização de texto (minúsculas, remoção de acentos)
- Limpeza de texto (pontuação, números, espaços extras)
- Análise de texto (contagem de palavras/caracteres, palavras únicas)
- Função de limpeza completa que combina todas as operações

Exemplo de uso:
    >>> from src.utils.text_processing import clean_text, count_words
    >>> texto = "Olá! Este é um TEXTO de exemplo... 123"
    >>> texto_limpo = clean_text(texto)
    >>> print(texto_limpo)  # "ola este e um texto de exemplo"
    >>> print(count_words(texto_limpo))  # 6

Author: JurisOracle Team
Version: 1.0.0
License: MIT
"""

import re
import string
import unicodedata
from typing import List


def to_lowercase(text: str) -> str:
    """
    Converte todo o texto para minúsculas.
    
    Esta função trata adequadamente caracteres Unicode e fornece uma maneira
    consistente de normalizar o case de strings.
    
    Args:
        text (str): O texto a ser convertido para minúsculas.
        
    Returns:
        str: O texto convertido para minúsculas.
        
    Raises:
        TypeError: Se o input não for uma string.
        
    Examples:
        >>> to_lowercase("HELLO WORLD")
        'hello world'
        >>> to_lowercase("Olá MUNDO!")
        'olá mundo!'
        >>> to_lowercase("")
        ''
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    
    return text.lower()


def remove_punctuation(text: str) -> str:
    """
    Remove todos os caracteres de pontuação do texto.
    
    Utiliza a definição de pontuação do módulo `string`, que inclui
    caracteres como .,;:!?"'()[]{}...
    
    Args:
        text (str): O texto do qual remover pontuação.
        
    Returns:
        str: O texto sem caracteres de pontuação.
        
    Raises:
        TypeError: Se o input não for uma string.
        
    Examples:
        >>> remove_punctuation("Hello, world!")
        'Hello world'
        >>> remove_punctuation("Texto com... pontuação?!")
        'Texto com pontuação'
        >>> remove_punctuation("Sem pontuacao")
        'Sem pontuacao'
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    
    # Usa translate() que é mais eficiente que replace() em loop
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def remove_numbers(text: str) -> str:
    """
    Remove todos os dígitos numéricos do texto.
    
    Esta função remove apenas dígitos (0-9), preservando letras e outros caracteres.
    
    Args:
        text (str): O texto do qual remover números.
        
    Returns:
        str: O texto sem dígitos numéricos.
        
    Raises:
        TypeError: Se o input não for uma string.
        
    Examples:
        >>> remove_numbers("Hello123 World456")
        'Hello World'
        >>> remove_numbers("Texto com 123 números 456")
        'Texto com  números '
        >>> remove_numbers("SemNumeros")
        'SemNumeros'
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    
    # Usa regex para eficiência
    return re.sub(r'\d', '', text)


def remove_extra_spaces(text: str) -> str:
    """
    Remove espaços extras e normaliza espaçamento.
    
    - Substitui sequências de múltiplos espaços por um único espaço
    - Remove espaços no início e final do texto
    - Converte tabs e quebras de linha em espaços únicos
    
    Args:
        text (str): O texto a ser normalizado.
        
    Returns:
        str: O texto com espaçamento normalizado.
        
    Raises:
        TypeError: Se o input não for uma string.
        
    Examples:
        >>> remove_extra_spaces("  Hello    world  ")
        'Hello world'
        >>> remove_extra_spaces("Texto\tcom\nespaços")
        'Texto com espaços'
        >>> remove_extra_spaces("")
        ''
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    
    # Substitui qualquer sequência de whitespace por um espaço único
    # e remove espaços do início e fim
    return re.sub(r'\s+', ' ', text).strip()


def remove_diacritics(text: str) -> str:
    """
    Remove acentos e caracteres diacríticos do texto.
    
    Converte caracteres como á, é, ñ, ç para suas versões sem acento (a, e, n, c).
    Utiliza normalização Unicode NFD para separar caracteres base de diacríticos.
    
    Args:
        text (str): O texto do qual remover diacríticos.
        
    Returns:
        str: O texto sem acentos e diacríticos.
        
    Raises:
        TypeError: Se o input não for uma string.
        
    Examples:
        >>> remove_diacritics("café com açúcar")
        'cafe com acucar'
        >>> remove_diacritics("naïve résumé")
        'naive resume'
        >>> remove_diacritics("España")
        'Espana'
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    
    # Normalização NFD separa caracteres base de diacríticos
    nfd = unicodedata.normalize('NFD', text)
    
    # Remove apenas caracteres de marca diacrítica (categoria Mn)
    without_diacritics = ''.join(
        char for char in nfd 
        if unicodedata.category(char) != 'Mn'
    )
    
    return without_diacritics


def clean_text(text: str) -> str:
    """
    Aplica uma limpeza completa do texto.
    
    Executa sequencialmente as seguintes operações:
    1. Converte para minúsculas
    2. Remove pontuação
    3. Remove números
    4. Remove diacríticos/acentos
    5. Normaliza espaçamento
    
    Args:
        text (str): O texto a ser limpo.
        
    Returns:
        str: O texto completamente processado e limpo.
        
    Raises:
        TypeError: Se o input não for uma string.
        
    Examples:
        >>> clean_text("Olá! Este é um TEXTO de exemplo... 123")
        'ola este e um texto de exemplo'
        >>> clean_text("  Café com açúcar, por favor!!!  ")
        'cafe com acucar por favor'
        >>> clean_text("")
        ''
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    
    # Pipeline de limpeza em sequência otimizada
    result = text
    result = to_lowercase(result)
    result = remove_punctuation(result)
    result = remove_numbers(result)
    result = remove_diacritics(result)
    result = remove_extra_spaces(result)
    
    return result


def count_words(text: str) -> int:
    """
    Conta o número total de palavras no texto.
    
    Considera palavras como sequências de caracteres não-whitespace.
    Texto vazio ou apenas com espaços retorna 0.
    
    Args:
        text (str): O texto no qual contar palavras.
        
    Returns:
        int: O número de palavras encontradas.
        
    Raises:
        TypeError: Se o input não for uma string.
        
    Examples:
        >>> count_words("Hello world")
        2
        >>> count_words("Palavra única")
        2
        >>> count_words("   ")
        0
        >>> count_words("")
        0
        >>> count_words("Uma, duas... três palavras!")
        4
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    
    # Remove espaços extras e divide por whitespace
    cleaned = text.strip()
    if not cleaned:
        return 0
    
    # split() sem argumentos divide por qualquer whitespace e remove strings vazias
    return len(cleaned.split())


def count_characters(text: str, include_spaces: bool = True) -> int:
    """
    Conta o número total de caracteres no texto.
    
    Args:
        text (str): O texto no qual contar caracteres.
        include_spaces (bool): Se True, inclui espaços na contagem.
                              Se False, conta apenas caracteres não-espaço.
                              Default: True.
        
    Returns:
        int: O número de caracteres encontrados.
        
    Raises:
        TypeError: Se text não for string ou include_spaces não for boolean.
        
    Examples:
        >>> count_characters("Hello world")
        11
        >>> count_characters("Hello world", include_spaces=False)
        10
        >>> count_characters("   ")
        3
        >>> count_characters("   ", include_spaces=False)
        0
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str for text, got {type(text).__name__}")
    
    if not isinstance(include_spaces, bool):
        raise TypeError(f"Expected bool for include_spaces, got {type(include_spaces).__name__}")
    
    if include_spaces:
        return len(text)
    else:
        # Remove apenas espaços em branco (space, tab, newline, etc.)
        return len(re.sub(r'\s', '', text))


def get_unique_words(text: str) -> List[str]:
    """
    Retorna uma lista de palavras únicas em minúsculas.
    
    - Converte texto para minúsculas
    - Remove pontuação
    - Extrai palavras únicas (sem duplicatas)
    - Retorna lista ordenada alfabeticamente
    
    Args:
        text (str): O texto do qual extrair palavras únicas.
        
    Returns:
        List[str]: Lista de palavras únicas em ordem alfabética.
        
    Raises:
        TypeError: Se o input não for uma string.
        
    Examples:
        >>> get_unique_words("Hello world hello")
        ['hello', 'world']
        >>> get_unique_words("Café, café e mais café!")
        ['cafe', 'e', 'mais']
        >>> get_unique_words("")
        []
        >>> get_unique_words("Uma duas uma três duas")
        ['duas', 'tres', 'uma']
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    
    if not text.strip():
        return []
    
    # Limpa o texto mantendo apenas letras e espaços
    cleaned = to_lowercase(text)
    cleaned = remove_punctuation(cleaned)
    cleaned = remove_numbers(cleaned)
    cleaned = remove_diacritics(cleaned)
    cleaned = remove_extra_spaces(cleaned)
    
    if not cleaned:
        return []
    
    # Extrai palavras únicas e ordena
    words = cleaned.split()
    unique_words = sorted(set(words))
    
    return unique_words


# Exemplo de uso e demonstração das funcionalidades
if __name__ == "__main__":
    print("=== DEMONSTRAÇÃO: Text Processing Utilities ===\n")
    
    # Texto de exemplo para demonstrações
    exemplo_texto = "  Olá, Mundo! Este é um TEXTO de exemplo... com 123 números e ção especiais!  "
    print(f"📝 Texto original:")
    print(f"   '{exemplo_texto}'\n")
    
    # Demonstração de cada função
    print("🔧 Aplicando funções individuais:")
    
    lower = to_lowercase(exemplo_texto)
    print(f"• to_lowercase(): '{lower}'")
    
    no_punct = remove_punctuation(exemplo_texto)
    print(f"• remove_punctuation(): '{no_punct}'")
    
    no_numbers = remove_numbers(exemplo_texto)
    print(f"• remove_numbers(): '{no_numbers}'")
    
    no_spaces = remove_extra_spaces(exemplo_texto)
    print(f"• remove_extra_spaces(): '{no_spaces}'")
    
    no_diacritics = remove_diacritics(exemplo_texto)
    print(f"• remove_diacritics(): '{no_diacritics}'")
    
    print()
    
    # Demonstração da função de limpeza completa
    print("🧹 Limpeza completa:")
    texto_limpo = clean_text(exemplo_texto)
    print(f"• clean_text(): '{texto_limpo}'\n")
    
    # Demonstração de análise de texto
    print("📊 Análise de texto:")
    palavras = count_words(exemplo_texto)
    chars_com_espaco = count_characters(exemplo_texto, include_spaces=True)
    chars_sem_espaco = count_characters(exemplo_texto, include_spaces=False)
    palavras_unicas = get_unique_words(exemplo_texto)
    
    print(f"• Número de palavras: {palavras}")
    print(f"• Caracteres (com espaços): {chars_com_espaco}")
    print(f"• Caracteres (sem espaços): {chars_sem_espaco}")
    print(f"• Palavras únicas: {palavras_unicas}")
    
    print("\n" + "="*60)
    
    # Casos especiais e edge cases
    print("\n🧪 TESTANDO CASOS ESPECIAIS:")
    
    casos_teste = [
        "",  # String vazia
        "   ",  # Apenas espaços
        "123",  # Apenas números
        "!!!",  # Apenas pontuação
        "a",  # Caractere único
        "Ção São João",  # Muitos diacríticos
    ]
    
    for i, caso in enumerate(casos_teste, 1):
        print(f"\nCaso {i}: '{caso}'")
        try:
            limpo = clean_text(caso)
            palavras = count_words(caso)
            print(f"  → Limpo: '{limpo}'")
            print(f"  → Palavras: {palavras}")
        except Exception as e:
            print(f"  → Erro: {e}")
    
    print("\n✅ Demonstração concluída!")