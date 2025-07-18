# JurisOracle

<p align="center">
  <img src="https://img.shields.io/badge/status-Em%20Desenvolvimento-orange" alt="Status do Projeto">
  <img src="https://img.shields.io/badge/licen%C3%A7a-MIT-blue" alt="Licença">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python Version">
</p>

### Sumário
*   [Descrição do Projeto](#descrição-do-projeto)
*   [Status do Projeto](#status-do-projeto)
*   [Funcionalidades Principais](#funcionalidades-principais)
*   [Tecnologias Utilizadas](#tecnologias-utilizadas)
*   [Pré-requisitos](#pré-requisitos)
*   [Instalação](#instalação)
*   [Uso](#uso)
*   [Configuração](#configuração)
*   [Executando os Testes](#executando-os-testes)
*   [Estrutura do Projeto](#estrutura-do-projeto)
*   [Como Contribuir](#como-contribuir)
*   [Licença](#licença)
*   [Contato](#contato)

## Descrição do Projeto
O **Juris Oracle** é um sistema de Inteligência Artificial avançado, projetado para analisar, processar e extrair insights de grandes volumes de documentos jurídicos. A plataforma visa resolver o desafio de lidar com a complexidade e a extensão de textos legais, como processos, petições e jurisprudências, oferecendo resumos coesos e um sistema de perguntas e respostas (Q&A) contextual.

**Público-alvo:** Advogados, analistas jurídicos, estudantes de direito e desenvolvedores de software que trabalham com tecnologia para a área jurídica (Legal Tech).

## Status do Projeto
<p>
  <img src="https://img.shields.io/badge/status-Em%20Desenvolvimento-orange" alt="Status do Projeto">
</p>

O projeto está atualmente em fase de **desenvolvimento ativo**. As funcionalidades principais estão sendo implementadas e aprimoradas.

### Roadmap Futuro
-   [ ] **Fase 1:** Estabilização da API principal e do fluxo de processamento de documentos.
-   [ ] **Fase 2:** Aprimoramento do modelo de fine-tuning e pipeline de avaliação.
-   [ ] **Fase 3:** Implementação de um dashboard web para interação visual com os documentos e resultados.
-   [ ] **Fase 4:** Suporte a novos formatos de documentos e fontes de dados (e.g., integração direta com sistemas de tribunais).

## Funcionalidades Principais
*   **Processamento de Documentos:** Capacidade de fazer upload e processar arquivos nos formatos `.pdf` e `.docx`.
*   **Sumarização Inteligente:** Geração de resumos automáticos e concisos de processos jurídicos, destacando os pontos-chave.
*   **Q&A Contextual:** Sistema de perguntas e respostas que permite ao usuário "conversar" com os documentos, obtendo respostas precisas com base no conteúdo.
*   **Embeddings de Alta Qualidade:** Utilização de técnicas avançadas de embeddings para representar semanticamente o conteúdo dos textos.
*   **Recuperação Otimizada com HyDE:** Implementação de *Hypothetical Document Embeddings* para melhorar a relevância dos documentos recuperados durante as buscas.
*   **Treinamento e Fine-Tuning:** Pipeline completo para treinar e ajustar modelos de linguagem (LLMs) com dados jurídicos específicos, melhorando a performance para o domínio.
*   **API Robusta:** Exposição de todas as funcionalidades através de uma API RESTful segura e escalável.

## Tecnologias Utilizadas
### O projeto é construído sobre um stack de tecnologias modernas e robustas:

| *   **Backend:** Python 3.10+, FastAPI |
|---|


*   **Inteligência Artificial & NLP:**
    *   Hugging Face Transformers, Tokenizers
    *   PyTorch ou TensorFlow
    *   Scikit-learn
*   **Banco de Dados Vetorial:** FAISS, Milvus (ou similar)
*   **Banco de Dados Relacional:** PostgreSQL (via SQLAlchemy)
*   **DevOps & Infraestrutura:** Docker, Docker Compose
*   **Testes:** Pytest
*   **Linting & Formatação:** Ruff, Black

## Pré-requisitos
### Antes de começar, certifique-se de ter as seguintes ferramentas instaladas em seu sistema:

| *   **Python:** Versão 3.10 ou superior. |
|---|


    ```bash
    python --version
    ```
*   **Docker e Docker Compose:** Essencial para executar o ambiente containerizado.
    ```bash
    docker --version
    docker-compose --version
    ```
*   **Git:** Para clonar o repositório.
    ```bash
    git --version
    ```

## Instalação
Siga os passos abaixo para configurar o ambiente de desenvolvimento local.

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/seu-usuario/juris_oracle.git
    cd juris_oracle
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python -m venv venv
    # No Windows:
    # venv\Scripts\activate
    # No macOS/Linux:
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    Para produção, instale `requirements.txt`. Para desenvolvimento (incluindo testes e ferramentas de linting), instale `requirements-dev.txt`.
    ```bash
    pip install -r requirements-dev.txt
    ```

4.  **Configure as variáveis de ambiente:**
    Copie o arquivo de exemplo e preencha com suas chaves e configurações. Veja a seção [Configuração](#configuração) para mais detalhes.
    ```bash
    cp .env.example .env
    ```

5.  **Inicie os serviços com Docker Compose:**
    Este comando irá construir as imagens e iniciar os contêineres da aplicação, banco de dados e outros serviços necessários.
    ```bash
    docker-compose up -d --build
    ```
    A aplicação estará disponível em `http://localhost:8000`.

## Uso
Após a instalação, a principal forma de interagir com o Juris Oracle é através da sua API RESTful.

### Acessando a Documentação da API
### A API possui documentação interativa (Swagger UI) gerada automaticamente pelo FastAPI. Acesse-a no seu navegador:

| `http://localhost:8000/docs` |
|---|



### Exemplos de Requisições (usando `curl`)

#### 1. Fazer upload de um documento
```bash
curl -X 'POST' \
  'http://localhost:8000/api/documents/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@"/caminho/para/seu/processo.pdf";type=application/pdf'
```

#### 2. Solicitar um resumo
```bash
curl -X 'POST' \
  'http://localhost:8000/api/services/summarize/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "document_id": "ID_DO_DOCUMENTO_RECEBIDO_NO_UPLOAD",
    "max_length": 150
  }'
```

#### 3. Fazer uma pergunta sobre o documento
```bash
curl -X 'POST' \
  'http://localhost:8000/api/services/qa/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "document_id": "ID_DO_DOCUMENTO_RECEBIDO_NO_UPLOAD",
    "question": "Qual foi a decisão do juiz na primeira instância?"
  }'
```

### Executando Scripts
O projeto inclui scripts para tarefas específicas, como treinar o modelo. Para executá-los:
```bash
# Exemplo de como rodar o script de treinamento
python scripts/train_model.py --data-path data/processed/ --output-path data/models/
```

## Configuração
A configuração da aplicação é gerenciada por meio de variáveis de ambiente. Crie um arquivo `.env` na raiz do projeto a partir do `.env.example` e preencha as seguintes variáveis:

```dotenv
# .env

# Configuração do Banco de Dados
DATABASE_URL="postgresql://user:password@db:5432/juris_oracle_db"

# Chaves de API de serviços externos (ex: OpenAI, Hugging Face Hub)
AI_MODEL_API_KEY="sua_chave_de_api"

# Configurações da Aplicação
SECRET_KEY="sua_chave_secreta_para_jwt"
ENVIRONMENT="development"
```

## Executando os Testes
Para garantir a qualidade e a estabilidade do código, o projeto utiliza uma suíte de testes automatizados com `pytest`.

Para executar todos os testes (unitários, integração e E2E), rode o seguinte comando na raiz do projeto:
```bash
pytest
```

## Estrutura do Projeto
A organização do código-fonte segue uma estrutura modular e escalável:

```
juris_oracle/
├── src/                # Código-fonte principal da aplicação
│   ├── api/            # Camada da API (FastAPI, rotas, middleware)
│   ├── config/         # Configurações da aplicação e logging
│   ├── core/           # Lógica central (processamento, embeddings, etc.)
│   ├── models/         # Modelos de dados (Pydantic, SQLAlchemy)
│   ├── services/       # Lógica de negócio (serviços de sumarização, Q&A)
│   ├── training/       # Pipeline de treinamento e avaliação de modelos
│   └── utils/          # Funções utilitárias e exceções customizadas
├── tests/              # Suíte de testes automatizados
├── data/               # Dados brutos, processados e modelos treinados
├── scripts/            # Scripts autônomos para tarefas de automação
├── docker/             # Arquivos Docker e Docker Compose
├── docs/               # Documentação adicional do projeto
├── requirements.txt    # Dependências de produção
└── README.md           # Este arquivo
```

## Licença
Este projeto é distribuído sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Contato
**Enzo** - *Líder do Projeto*

Para dúvidas, sugestões ou suporte, a melhor forma de contato é abrindo uma [Issue no GitHub](https://github.com/seu-usuario/juris_oracle/issues).