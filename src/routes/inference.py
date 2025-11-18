"""
Rutas para inferencia y entrenamiento de modelos de Machine Learning

Este módulo contiene los endpoints para:
- Predicción de adherencia de pacientes
- Entrenamiento de modelos
"""

import uuid

from fastapi import APIRouter, HTTPException

from app_types.data import ResultadoEntrenamiento
from app_types.inference import (
    PredictionRequest,
    PredictionResponse,
    TrainingRequest,
)
from services.inference import predecir_con_modelo
from services.monitoring import get_monitoring_service
from utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/laboratorio", tags=["inferencia"])


@router.post("/predecir", response_model=PredictionResponse)
async def predecir(request: PredictionRequest) -> PredictionResponse:
    """
    Predecir adherencia de paciente

    Realiza predicción de adherencia a 12 meses usando el modelo especificado.
    Recibe tipo de modelo (xgboost o neural_network), datos demográficos,
    clínicos y agregados del paciente.

    Tipos de modelo disponibles:
    - "xgboost": Gradient Boosting (HistGradientBoostingClassifier)
    - "neural_network": Red Neuronal con TensorFlow

    Retorna la predicción (0 o 1), probabilidad y metadatos del modelo.
    """
    request_id = str(uuid.uuid4())
    monitoreo = get_monitoring_service()

    try:
        # Realizar predicción con el modelo especificado
        respuesta = predecir_con_modelo(request)

        # Registrar predicción exitosa
        monitoreo.log_prediction(
            request_id=request_id,
            model_version=respuesta.model_version,
            prediction=respuesta.prediction,
            probability=respuesta.probability,
            inference_time_ms=respuesta.inference_time_ms,
            success=True,
            input_features=request.model_dump(),
        )

        logger.info(
            f"Predicción exitosa con {request.tipo_modelo.value}. "
            f"Request ID: {request_id}, "
            f"Predicción: {respuesta.prediction}, "
            f"Probabilidad: {respuesta.probability:.4f}"
        )

        return respuesta

    except FileNotFoundError as e:
        logger.error(f"Modelo no encontrado: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        # Registrar predicción fallida
        monitoreo.log_prediction(
            request_id=request_id,
            model_version=f"{request.tipo_modelo.value}_fallida",
            prediction=-1,
            probability=0.0,
            inference_time_ms=0.0,
            success=False,
            input_features=request.model_dump(),
            error_message=str(e),
        )

        logger.error(
            f"Predicción fallida con {request.tipo_modelo.value}. "
            f"Request ID: {request_id}, Error: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Error durante la predicción: {str(e)}",
        )


@router.post("/modelado/entrenar", response_model=ResultadoEntrenamiento)
async def entrenar_modelo(
    request: TrainingRequest,
) -> ResultadoEntrenamiento:
    """
    Entrenar modelo de predicción de adherencia

    Entrena un modelo usando el dataset más reciente de ./data/dataset_modelado_*.csv
    generado por GET /laboratorio/dataset/modelado.

    Tipos de modelo disponibles:
    - "xgboost": Gradient Boosting (HistGradientBoostingClassifier)
    - "neural_network": Red Neuronal con TensorFlow

    El proceso incluye: carga del CSV, codificación de categóricas, split 80/20,
    entrenamiento y evaluación con métricas estándar (Accuracy, Precision, Recall, F1, AUC).

    El modelo entrenado se guarda automáticamente en ./models con timestamp.
    Requiere que exista al menos un archivo dataset en ./data.
    """
    from services.inference import entrenar_modelo_especifico

    try:
        logger.info(f"Iniciando entrenamiento de modelo: {request.tipo_modelo}")
        resultado = await entrenar_modelo_especifico(request.tipo_modelo)
        logger.info(
            f"Entrenamiento completado exitosamente. Modelo: {resultado.modelo}, "
            f"F1-Score: {resultado.metricas_test.f1_score:.4f}"
        )
        return resultado
    except FileNotFoundError as e:
        logger.error(f"Dataset no encontrado: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Error de validación en entrenamiento: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el entrenamiento: {str(e)}",
        )
