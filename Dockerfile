FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for layer caching
COPY requirements.txt pyproject.toml ./

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy full source
COPY . .

# Make the server package importable from /app root
ENV PYTHONPATH=/app

EXPOSE 7860

# Health check (OpenEnv validate pings /reset)
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -sf -X POST http://localhost:7860/reset \
        -H "Content-Type: application/json" -d '{}' || exit 1

CMD ["python", "-m", "server.app"]
