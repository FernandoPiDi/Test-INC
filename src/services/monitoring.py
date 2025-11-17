"""
Monitoring service for model performance tracking
"""
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from types.monitoring import ModelMetrics, PredictionLog

from utils.logging_config import get_logger

logger = get_logger(__name__)


class MonitoringService:
    """
    Service for monitoring model performance and predictions
    """

    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        self.prediction_logs: list[PredictionLog] = []
        self._load_recent_logs()

    def _load_recent_logs(self) -> None:
        """Load recent prediction logs from disk"""
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
        Log a prediction request
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

        # Persist to disk (append mode)
        try:
            log_file = self.logs_dir / "predictions.jsonl"
            with open(log_file, "a") as f:
                f.write(log_entry.model_dump_json() + "\n")
        except Exception as e:
            logger.error(f"Error writing prediction log: {e}")

        # Keep only last 1000 logs in memory
        if len(self.prediction_logs) > 1000:
            self.prediction_logs = self.prediction_logs[-1000:]

    def get_metrics(
        self, model_version: Optional[str] = None, hours: int = 24
    ) -> ModelMetrics:
        """
        Calculate metrics for the specified time period
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter logs by time and version
        filtered_logs = [
            log
            for log in self.prediction_logs
            if log.timestamp >= cutoff_time
            and (model_version is None or log.model_version == model_version)
        ]

        if not filtered_logs:
            return ModelMetrics(
                timestamp=datetime.now(),
                model_version=model_version or "unknown",
                total_predictions=0,
                predictions_class_0=0,
                predictions_class_1=0,
                avg_inference_time_ms=0.0,
                p95_inference_time_ms=0.0,
                p99_inference_time_ms=0.0,
                error_count=0,
                success_rate=1.0,
            )

        # Calculate metrics
        total = len(filtered_logs)
        successful = sum(1 for log in filtered_logs if log.success)
        errors = total - successful

        predictions_0 = sum(1 for log in filtered_logs if log.prediction == 0)
        predictions_1 = sum(1 for log in filtered_logs if log.prediction == 1)

        inference_times = [
            log.inference_time_ms for log in filtered_logs if log.success
        ]

        if inference_times:
            sorted_times = sorted(inference_times)
            avg_time = sum(inference_times) / len(inference_times)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            p95_time = sorted_times[min(p95_idx, len(sorted_times) - 1)]
            p99_time = sorted_times[min(p99_idx, len(sorted_times) - 1)]
        else:
            avg_time = 0.0
            p95_time = 0.0
            p99_time = 0.0

        return ModelMetrics(
            timestamp=datetime.now(),
            model_version=model_version or "unknown",
            total_predictions=total,
            predictions_class_0=predictions_0,
            predictions_class_1=predictions_1,
            avg_inference_time_ms=round(avg_time, 2),
            p95_inference_time_ms=round(p95_time, 2),
            p99_inference_time_ms=round(p99_time, 2),
            error_count=errors,
            success_rate=round(successful / total if total > 0 else 0.0, 4),
        )

    def get_recent_predictions(
        self, limit: int = 100, model_version: Optional[str] = None
    ) -> list[PredictionLog]:
        """
        Get recent prediction logs
        """
        filtered = self.prediction_logs
        if model_version:
            filtered = [log for log in filtered if log.model_version == model_version]

        # Sort by timestamp descending
        filtered.sort(key=lambda x: x.timestamp, reverse=True)

        return filtered[:limit]


# Global monitoring service instance
_monitoring_service: Optional[MonitoringService] = None


def get_monitoring_service() -> MonitoringService:
    """
    Get or create the global monitoring service instance
    """
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service

