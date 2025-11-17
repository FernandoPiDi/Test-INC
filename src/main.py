"""
API de Inferencia para Predicción de Adherencia de Pacientes
Despliegue del modelo Random Forest usando FastAPI

Endpoints principales:
- POST /api/v1/predict: Predicción de adherencia
- POST /laboratorio/datos: Sube datos desde archivo Excel
- POST /laboratorio/modelado/entrenar: Entrena modelos
- GET /health: Health check
"""

import time
from datetime import datetime
from types.monitoring import HealthStatus

import uvicorn
from fastapi import FastAPI
from sqlmodel import Session, text

from routes.inference import router as inference_router
from routes.main import router as laboratorio_router
from services.inference import get_model_loader
from utils.logging_config import get_logger, setup_logging
from utils.settings import engine

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Track startup time
startup_time = time.time()

# ============================================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================================

app = FastAPI(
    title="API de Predicción de Adherencia",
    description="API para predecir la adherencia de pacientes a tratamiento oncológico usando Random Forest",
    version="1.0.0",
)

# Include routers
app.include_router(laboratorio_router)
app.include_router(inference_router)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting up API...")
    # Pre-load model
    try:
        model_loader = get_model_loader()
        if model_loader.is_loaded():
            logger.info("Model loaded successfully on startup")
        else:
            logger.warning("No model loaded on startup")
    except Exception as e:
        logger.error(f"Error loading model on startup: {e}")


@app.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Health check endpoint

    Verifica el estado de la API, modelo y base de datos.
    """
    model_loader = get_model_loader()
    model_loaded = model_loader.is_loaded()
    model_info = model_loader.get_model_info()
    model_version = model_info.model_version if model_info else None

    # Check database connection
    db_connected = False
    try:
        with Session(engine) as session:
            session.execute(text("SELECT 1"))
            db_connected = True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")

    # Determine overall status
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
