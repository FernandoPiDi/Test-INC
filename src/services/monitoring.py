"""
Servicio de monitoreo de desempeño de modelos

Gestiona el registro y análisis de predicciones y métricas de modelos de ML.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from app_types.monitoring import PredictionLog
from utils.logging_config import get_logger

logger = get_logger(__name__)


class MonitoringService:
    """
    Servicio para monitoreo de desempeño y predicciones de modelos

    Registra todas las predicciones en disco y calcula métricas
    de desempeño en tiempo real.
    """

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        self.prediction_logs: list[PredictionLog] = []
        self._load_recent_logs()

    def _load_recent_logs(self) -> None:
        """Cargar logs de predicciones recientes desde disco"""
        try:
            log_file = self.logs_dir / "predictions.jsonl"
            if log_file.exists():
                with open(log_file, "r") as f:
                    for line in f:
                        if line.strip():
                            try:
                                log_data = json.loads(line)
                                # Convert timestamp string back to datetime
                                log_data["timestamp"] = datetime.fromisoformat(
                                    log_data["timestamp"]
                                )
                                self.prediction_logs.append(PredictionLog(**log_data))
                            except Exception as e:
                                logger.warning(f"Error loading log entry: {e}")
        except Exception as e:
            logger.warning(f"Error loading prediction logs: {e}")

    def log_prediction(
        self,
        request_id: str,
        model_version: str,
        prediction: int,
        probability: float,
        inference_time_ms: float,
        success: bool,
        input_features: dict,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Registrar una solicitud de predicción

        Args:
            request_id: ID único de la solicitud
            model_version: Versión del modelo utilizado
            prediction: Predicción realizada (0 o 1)
            probability: Probabilidad de la predicción
            inference_time_ms: Tiempo de inferencia en milisegundos
            success: Indica si la predicción fue exitosa
            input_features: Features de entrada utilizados
            error_message: Mensaje de error si la predicción falló
        """
        log_entry = PredictionLog(
            timestamp=datetime.now(),
            request_id=request_id,
            model_version=model_version,
            prediction=prediction,
            probability=probability,
            inference_time_ms=inference_time_ms,
            success=success,
            error_message=error_message,
            input_features=input_features,
        )

        self.prediction_logs.append(log_entry)

        # Persistir en disco (modo append)
        try:
            log_file = self.logs_dir / "predictions.jsonl"
            with open(log_file, "a") as f:
                f.write(log_entry.model_dump_json() + "\n")
        except Exception as e:
            logger.error(f"Error escribiendo log de predicción: {e}")

        # Mantener solo los últimos 1000 logs en memoria
        if len(self.prediction_logs) > 1000:
            self.prediction_logs = self.prediction_logs[-1000:]


# Instancia global del servicio de monitoreo
_monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    """
    Obtener o crear la instancia global del servicio de monitoreo

    Returns:
        Instancia singleton de MonitoringService
    """
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service
