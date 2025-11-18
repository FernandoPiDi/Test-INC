# ============================================================================
# Dockerfile para API de Predicción de Adherencia de Pacientes
# Optimizado para Python 3.13 y uv
# ============================================================================

FROM python:3.13-slim

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PYTHONHASHSEED=random

# Instalar dependencias del sistema (incluyendo cliente de postgres para el entrypoint)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq5 \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/models /app/logs /app/data && \
    chown -R appuser:appuser /app

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de configuración
COPY --chown=appuser:appuser pyproject.toml ./

# Instalar uv
RUN pip install --no-cache-dir uv

# Instalar dependencias usando uv
RUN uv pip install --system --no-cache -r pyproject.toml

# Copiar código de la aplicación
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser alembic/ ./alembic/
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser entrypoint.sh ./

# Hacer ejecutable el entrypoint
RUN chmod +x /app/entrypoint.sh

# Crear directorios necesarios con permisos
RUN mkdir -p /app/models /app/logs /app/data && \
    chown -R appuser:appuser /app

# Cambiar a usuario no-root
USER appuser

# Exponer puerto 8000 para FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Punto de entrada para ejecutar migraciones antes de iniciar la API
ENTRYPOINT ["/app/entrypoint.sh"]
