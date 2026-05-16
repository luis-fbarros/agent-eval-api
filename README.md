# AI Agent Evaluation Pipeline

Uma API de avaliação e observabilidade para agentes de Inteligência Artificial. Esta aplicação processa rastreamentos (traces) de execução de agentes, executa uma série de avaliações (estáticas, semânticas com DeepEval e personalizadas com Gemini), e registra os resultados em um painel do MLflow via DagsHub.

## 🎯 Escopo da Aplicação

O objetivo principal desta API é fornecer um motor centralizado para avaliar o desempenho, a segurança e a eficácia de agentes de IA em tempo real ou em lote (batch). O fluxo funciona da seguinte maneira:
1. **Ingestão**: A API recebe os dados de execução do agente (prompt do usuário, resposta do agente, duração, tokens, saída de ferramentas).
2. **Avaliação Assíncrona**: O pipeline é processado em background para não bloquear a requisição.
3. **Métricas Estáticas**: Avalia tempo de resposta (latência), eficiência de tokens e verbosidade.
4. **Métricas Semânticas (LLM-as-a-Judge)**: Utiliza a biblioteca DeepEval e o modelo Gemini para calcular relevância da resposta, fidelidade (faithfulness) ao contexto, alucinações, completude da tarefa, clareza e segurança.
5. **Observabilidade**: Todos os resultados, métricas e tags são enviados para o MLflow (hospedado no DagsHub), permitindo a visualização em dashboards.

## 🚀 Como Rodar o Projeto

### Pré-requisitos
- Python 3.11+
- Conta no [DagsHub](https://dagshub.com/) (com um repositório MLflow criado)
- Chave de API do Google Gemini (`GOOGLE_API_KEY`)

### Configuração de Variáveis de Ambiente
Crie um arquivo `.env` na raiz do projeto com o seguinte conteúdo:
```env
GOOGLE_API_KEY=sua_chave_do_gemini
DAGSHUB_USER_TOKEN=seu_token_do_dagshub
```

### Rodando Localmente (Desenvolvimento)
1. Crie um ambiente virtual e ative-o:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements_api.txt
   ```
3. Inicie o servidor:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080 --reload
   ```

### Rodando com Docker
1. Construa a imagem Docker:
   ```bash
   docker build -t agent-eval-api .
   ```
2. Rode o container:
   ```bash
   docker run -p 8080:8080 --env-file .env agent-eval-api
   ```

A API estará disponível em `http://localhost:8080`.
Você pode acessar a documentação interativa do Swagger em `http://localhost:8080/docs`.
