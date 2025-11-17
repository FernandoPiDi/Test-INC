"""
Inference routes for model predictions
"""
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException

from services.inference import get_model_loader
from services.monitoring import get_monitoring_service
from types.inference import PredictionRequest, PredictionResponse, ModelInfo
from types.monitoring import ModelMetrics, PredictionLog
from utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["inference"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Endpoint para realizar predicciones de adherencia de pacientes

    Recibe los datos del paciente y retorna la predicción de adherencia a 12 meses
    junto con la probabilidad y metadatos del modelo.

    ### Parámetros de entrada:
    - **Demográficos**: sexo, edad, zona_residencia
    - **Clínicos**: tipo_cancer, estadio, aseguradora
    - **Agregados**: count_consultas, count_laboratorios, promedios de pruebas

    ### Respuesta:
    - **prediction**: 0 (No adherente) o 1 (Adherente)
    - **probability**: Probabilidad de adherencia (0.0 a 1.0)
    - **model_version**: Versión del modelo utilizado
    - **inference_time_ms**: Tiempo de inferencia en milisegundos

    ### Ejemplo de uso:
    ```json
    {
      "sexo": "Femenino",
      "edad": 55,
      "zona_residencia": "Urbana",
      "tipo_cancer": "Mama",
      "estadio": "Ii",
      "aseguradora": "Sura",
      "count_consultas": 12,
      "dias_desde_diagnostico": 365,
      "count_laboratorios": 8,
      "avg_resultado_numerico": 2.5,
      "avg_biopsia": 0.0,
      "avg_vpH": 0.0,
      "avg_marcador_ca125": 45.3,
      "avg_psa": 0.0,
      "avg_colonoscopia": 0.0
    }
    ```
    """
    request_id = str(uuid.uuid4())
    model_loader = get_model_loader()
    monitoring = get_monitoring_service()

    try:
        # Check if model is loaded
        if not model_loader.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please train a model first or check model files.",
            )

        # Make prediction
        response = model_loader.predict(request)

        # Log successful prediction
        monitoring.log_prediction(
            request_id=request_id,
            model_version=response.model_version,
            prediction=response.prediction,
            probability=response.probability,
            inference_time_ms=response.inference_time_ms,
            success=True,
            input_features=request.model_dump(),
        )

        logger.info(
            f"Prediction successful. Request ID: {request_id}, "
            f"Prediction: {response.prediction}, Probability: {response.probability:.4f}"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        # Log failed prediction
        model_info = model_loader.get_model_info()
        model_version = model_info.model_version if model_info else "unknown"

        monitoring.log_prediction(
            request_id=request_id,
            model_version=model_version,
            prediction=-1,
            probability=0.0,
            inference_time_ms=0.0,
            success=False,
            input_features=request.model_dump(),
            error_message=str(e),
        )

        logger.error(f"Prediction failed. Request ID: {request_id}, Error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}",
        )


@router.get("/model/info", response_model=ModelInfo)
async def get_model_info() -> ModelInfo:
    """
    Obtener información sobre el modelo cargado actualmente

    Retorna detalles del modelo incluyendo versión, features, y fecha de carga.
    """
    model_loader = get_model_loader()

    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=404,
            detail="No model loaded. Please train a model first.",
        )

    model_info = model_loader.get_model_info()
    if not model_info:
        raise HTTPException(
            status_code=500,
            detail="Model info not available.",
        )

    return model_info


@router.get("/model/metrics", response_model=ModelMetrics)
async def get_model_metrics(
    model_version: str | None = None, hours: int = 24
) -> ModelMetrics:
    """
    Obtener métricas de desempeño del modelo

    ### Parámetros:
    - **model_version**: Versión específica del modelo (opcional)
    - **hours**: Período de tiempo en horas para calcular métricas (default: 24)

    ### Métricas retornadas:
    - Total de predicciones
    - Distribución de predicciones (clase 0 vs clase 1)
    - Tiempos de inferencia (promedio, P95, P99)
    - Tasa de éxito
    - Conteo de errores
    """
    monitoring = get_monitoring_service()
    return monitoring.get_metrics(model_version=model_version, hours=hours)


@router.get("/model/predictions", response_model=list[PredictionLog])
async def get_recent_predictions(
    limit: int = 100, model_version: str | None = None
) -> list[PredictionLog]:
    """
    Obtener predicciones recientes

    ### Parámetros:
    - **limit**: Número máximo de predicciones a retornar (default: 100)
    - **model_version**: Filtrar por versión específica del modelo (opcional)
    """
    monitoring = get_monitoring_service()
    return monitoring.get_recent_predictions(limit=limit, model_version=model_version)


@router.post("/model/reload")
async def reload_model() -> dict:
    """
    Recargar el modelo más reciente desde el directorio de modelos

    Útil después de entrenar un nuevo modelo o actualizar versiones.
    """
    from services.inference import reload_model

    try:
        reload_model()
        logger.info("Model reloaded successfully via API")
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error reloading model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reloading model: {str(e)}",
        )

