# ============================================================================
# Dockerfile para API de Predicción de Adherencia de Pacientes
# Multi-stage build para optimizar tamaño de imagen
# ============================================================================

# ============================================================================
# Stage 1: Build stage - Instalar dependencias y compilar
# ============================================================================
FROM python:3.10-slim as builder

# Información del mantenedor
LABEL maintainer="Data Engineering Team"
LABEL description="API de predicción de adherencia de pacientes oncológicos"
LABEL version="1.0.0"

# Variables de entorno para build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Instalar dependencias del sistema necesarias para compilar
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt pyproject.toml ./

# Instalar dependencias de Python
RUN pip install --upgrade pip setuptools wheel && \
    pip install --user --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime stage - Imagen final optimizada
# ============================================================================
FROM python:3.10-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONHASHSEED=random

# Instalar solo dependencias de runtime necesarias
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/models /app/logs /app/data && \
    chown -R appuser:appuser /app

# Copiar dependencias instaladas desde builder
COPY --from=builder /root/.local /root/.local

# Establecer directorio de trabajo
WORKDIR /app

# Copiar código de la aplicación
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser alembic/ ./alembic/
COPY --chown=appuser:appuser alembic.ini ./

# Copiar modelos entrenados si existen (opcional, pueden montarse como volumen)
# Usar shell para manejar archivos que pueden no existir
RUN chown -R appuser:appuser /app && \
    if [ -f "models/modelo_rf.pkl" ]; then cp models/modelo_rf.pkl models/ || true; fi && \
    if [ -f "feature_names.pkl" ]; then cp feature_names.pkl models/ || true; fi

# Cambiar a usuario no-root
USER appuser

# Exponer puerto 8000 para FastAPI
EXPOSE 8000

# Health check mejorado
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Comando para iniciar el servidor
# Usar uvicorn con configuración optimizada
# En producción, considerar usar gunicorn con uvicorn workers
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "5"]
