# ── Stage 1: Builder ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy runtime requirements
COPY requirements.txt .

# Install production Python dependencies into /install directory
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Install runtime utilities (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed dependencies from builder stage
COPY --from=builder /install /usr/local

# Create non-root system user for container security compliance
RUN useradd -m -u 10001 -s /bin/bash appuser

# Copy application source code, versioned model artifacts, and processed data
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    MODEL_VERSION=latest

# Expose service port
EXPOSE 8000

# Container Healthcheck Probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch Uvicorn ASGI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
