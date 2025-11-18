"""
API de Predicción de Adherencia de Pacientes
FastAPI application con dos routers principales: datos e inferencia
"""

import time
from datetime import datetime

import uvicorn
from fastapi import FastAPI
from sqlmodel import Session, text

from app_types.monitoring import HealthStatus
from routes import data_router, inference_router
from services.inference import get_model_loader
from utils.logging_config import get_logger, setup_logging
from utils.settings import engine

# Configurar sistema de logging
setup_logging()
logger = get_logger(__name__)

# Registrar tiempo de inicio para cálculo de uptime
startup_time = time.time()

# ============================================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================================

app = FastAPI(
    title="API de Predicción de Adherencia",
    description="API para predecir la adherencia de pacientes a tratamiento oncológico usando Random Forest",
    version="1.0.0",
)

# Incluir routers
app.include_router(data_router)
app.include_router(inference_router)


@app.on_event("startup")
async def startup_event():
    """Inicializar servicios al arrancar la aplicación"""
    logger.info("Iniciando API...")
    # Precargar modelo
    try:
        model_loader = get_model_loader()
        if model_loader.is_loaded():
            logger.info("Modelo cargado exitosamente en el arranque")
        else:
            logger.warning("No se cargó ningún modelo en el arranque")
    except Exception as e:
        logger.error(f"Error cargando modelo en el arranque: {e}")


@app.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Endpoint de verificación de salud del sistema

    Verifica el estado de la API, modelo y base de datos.
    """
    model_loader = get_model_loader()
    model_loaded = model_loader.is_loaded()
    model_info = model_loader.get_model_info()
    model_version = model_info.model_version if model_info else None

    # Verificar conexión a base de datos
    db_connected = False
    try:
        with Session(engine) as session:
            session.execute(text("SELECT 1"))
            db_connected = True
    except Exception as e:
        logger.error(f"Falló verificación de conexión a base de datos: {e}")

    # Determinar estado general del sistema
    if model_loaded and db_connected:
        status = "healthy"
    elif model_loaded or db_connected:
        status = "degraded"
    else:
        status = "unhealthy"

    uptime = time.time() - startup_time

    return HealthStatus(
        status=status,
        model_loaded=model_loaded,
        model_version=model_version,
        database_connected=db_connected,
        timestamp=datetime.now(),
        uptime_seconds=round(uptime, 2),
        version="1.0.0",
    )


# ============================================================================
# EJECUTAR SERVIDOR
# ============================================================================

if __name__ == "__main__":
    # Ejecutar servidor con uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Desactivar en producción
    )
