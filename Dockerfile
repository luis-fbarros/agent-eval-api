FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema (necessárias para compilar algumas libs do deepeval/mlflow)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python antes de copiar o código
# (aproveita cache do Docker enquanto requirements não mudam)
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copia apenas o que a API precisa — SDK e mlflow_server ficam de fora
COPY main.py .
COPY eval_pipeline/ eval_pipeline/

# Cloud Run injeta $PORT em runtime (padrão 8080).
# Localmente, PORT não é setada e cai no default 8080.
EXPOSE 8080

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
