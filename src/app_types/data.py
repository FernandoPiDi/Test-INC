from datetime import date as date_type
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class PacienteLabAnalysis(BaseModel):
    """
    Pydantic model for patient lab analysis response
    """

    id_paciente: str
    sexo: str
    edad: int
    zona_residencia: Optional[str]
    fecha_dx: date_type
    tipo_cancer: str
    estadio: str
    aseguradora: str
    adherencia_12m: bool
    primera_fecha_lab: Optional[date_type]
    dias_entre_lab_y_dx: Optional[int]
    mensaje_consistencia: Optional[str] = None


class PacienteActividadReciente(BaseModel):
    """
    Pydantic model for patient recent activity response
    """

    id_paciente: str
    sexo: str
    edad: int
    zona_residencia: Optional[str]
    fecha_dx: date_type
    tipo_cancer: str
    estadio: str
    aseguradora: str
    adherencia_12m: bool
    ultima_consulta: Optional[date_type]
    dias_sin_consulta: Optional[int]
    total_consultas: int
    estado_actividad: str


class DashboardTipoCancer(BaseModel):
    """
    Pydantic model for cancer type dashboard response
    """

    tipo_cancer: str
    total_pacientes: int
    promedio_edad: float
    consultas_ultimos_6_meses: int
    total_laboratorios: int


class NormalizacionCampo(BaseModel):
    """
    Pydantic model for field normalization report
    """

    campo: str
    registros_modificados: int
    valores_unicos_antes: int
    valores_unicos_despues: int
    ejemplos_cambios: Dict[str, str]


class AnalisisValoresFaltantes(BaseModel):
    """
    Pydantic model for missing values analysis
    """

    campo: str
    total_registros: int
    valores_faltantes: int
    porcentaje_faltante: float
    recomendacion: str


class OutliersDetectados(BaseModel):
    """
    Pydantic model for outliers detection report
    """

    campo: str
    total_registros: int
    q1: float
    q3: float
    iqr: float
    limite_inferior: float
    limite_superior: float
    outliers_detectados: int
    porcentaje_outliers: float
    valores_outliers: List[float]


class ReporteProcesamiento(BaseModel):
    """
    Pydantic model for complete data processing report
    """

    normalizaciones: List[NormalizacionCampo]
    valores_faltantes: List[AnalisisValoresFaltantes]
    outliers: List[OutliersDetectados]
    resumen: Dict[str, Any]


class DatasetModelado(BaseModel):
    """
    Pydantic model for modeling-ready dataset record (one row per patient)
    """

    # Demographic variables
    sexo: str
    edad: int
    zona_residencia: str

    # Clinical variables
    tipo_cancer: str
    estadio: str
    aseguradora: str

    # Aggregated features - consultations
    count_consultas: int
    dias_desde_diagnostico: int

    # Aggregated features - lab tests
    count_laboratorios: int
    avg_resultado_numerico: float

    # Lab test averages by type (most common types)
    avg_biopsia: float
    avg_vpH: float
    avg_marcador_ca125: float
    avg_psa: float
    avg_colonoscopia: float

    # Target variable
    adherencia_12m: int


class MetricasModelo(BaseModel):
    """
    Pydantic model for model evaluation metrics
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float


class ResultadoEntrenamiento(BaseModel):
    """
    Pydantic model for training results
    """

    modelo: str
    total_registros: int
    registros_train: int
    registros_test: int
    features_utilizadas: List[str]
    metricas_train: MetricasModelo
    metricas_test: MetricasModelo
    tiempo_entrenamiento_segundos: float
    arquitectura: Dict[str, Any]
    resumen: str


class RutaDataset(BaseModel):
    """
    Pydantic model para respuesta con ruta de archivo de dataset
    """

    ruta_archivo: str
    total_registros: int
    descripcion: str
