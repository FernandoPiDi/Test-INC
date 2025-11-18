"""
Pydantic types for inference endpoints
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TipoModelo(str, Enum):
    """
    Tipos de modelo disponibles para entrenamiento
    """

    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"


class PredictionRequest(BaseModel):
    """
    Request model for prediction endpoint
    """

    # Tipo de modelo a utilizar
    tipo_modelo: TipoModelo = Field(
        ...,
        description="Tipo de modelo a usar: 'xgboost' o 'neural_network'",
    )

    # Demographic variables
    sexo: str = Field(..., description="Sexo del paciente")
    edad: int = Field(..., ge=0, le=120, description="Edad del paciente")
    zona_residencia: str = Field(..., description="Zona de residencia")

    # Clinical variables
    tipo_cancer: str = Field(..., description="Tipo de cáncer")
    estadio: str = Field(..., description="Estadio del cáncer")
    aseguradora: str = Field(..., description="Aseguradora")

    # Aggregated features - consultations
    count_consultas: int = Field(..., ge=0, description="Número total de consultas")
    dias_desde_diagnostico: int = Field(
        ..., ge=0, description="Días desde el diagnóstico"
    )

    # Aggregated features - lab tests
    count_laboratorios: int = Field(
        ..., ge=0, description="Número total de pruebas de laboratorio"
    )
    avg_resultado_numerico: float = Field(
        ..., ge=0.0, description="Promedio general de resultados numéricos"
    )

    # Lab test averages by type
    avg_biopsia: float = Field(
        default=0.0, ge=0.0, description="Promedio de resultados de biopsias"
    )
    avg_vpH: float = Field(
        default=0.0, ge=0.0, description="Promedio de resultados de pruebas VPH"
    )
    avg_marcador_ca125: float = Field(
        default=0.0,
        ge=0.0,
        description="Promedio de marcador CA125",
    )
    avg_psa: float = Field(default=0.0, ge=0.0, description="Promedio de PSA")
    avg_colonoscopia: float = Field(
        default=0.0, ge=0.0, description="Promedio de colonoscopias"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "tipo_modelo": "xgboost",
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
                "avg_colonoscopia": 0.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    Response model for prediction endpoint
    """

    prediction: int = Field(
        ..., description="Predicción de adherencia (0 = No adherente, 1 = Adherente)"
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidad de adherencia (0.0 a 1.0)",
    )
    model_version: str = Field(..., description="Versión del modelo utilizado")
    model_name: str = Field(..., description="Nombre del modelo")
    inference_time_ms: float = Field(
        ..., description="Tiempo de inferencia en milisegundos"
    )


class TrainingRequest(BaseModel):
    """
    Request model for training endpoint
    """

    tipo_modelo: TipoModelo = Field(
        ...,
        description="Tipo de modelo a entrenar: 'xgboost' o 'neural_network'",
    )

    model_config = {"json_schema_extra": {"example": {"tipo_modelo": "xgboost"}}}


class ModelInfo(BaseModel):
    """
    Information about the loaded model
    """

    model_name: str
    model_version: str
    model_type: str
    features: list[str]
    loaded_at: str
    model_path: Optional[str] = None
