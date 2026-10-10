FROM python:3.11-slim

LABEL maintainer="DiegoSanch18"
LABEL description="Atomic AI Proxy (:8000) - Cognitive Task Decomposition & SLM Micro-Prompts"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PROXY_HOST=0.0.0.0 \
    PROXY_PORT=8000

WORKDIR /app

# Install dependencies first for optimal Docker layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
