#!/bin/sh
# Entrypoint script para API de adherencia
# Espera a que postgres esté listo y ejecuta migraciones antes de iniciar la API

set -e

echo "Esperando a que PostgreSQL esté disponible..."

# Esperar a que postgres esté listo
until PGPASSWORD=$DB_PASSWORD psql -h "$DB_HOST" -U "$DB_USERNAME" -d "$DB_NAME" -c '\q' 2>/dev/null; do
  echo "PostgreSQL no está disponible todavía - esperando..."
  sleep 2
done

echo "✅ PostgreSQL está disponible"

echo "Ejecutando migraciones de Alembic..."
alembic upgrade head

echo "✅ Migraciones completadas"

echo "Iniciando API de FastAPI..."
exec python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

