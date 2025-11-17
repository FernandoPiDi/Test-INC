"""
Pydantic types for monitoring and logging
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ModelMetrics(BaseModel):
    """
    Metrics for model performance monitoring
    """

    timestamp: datetime
    model_version: str
    total_predictions: int
    predictions_class_0: int
    predictions_class_1: int
    avg_inference_time_ms: float
    p95_inference_time_ms: float
    p99_inference_time_ms: float
    error_count: int
    success_rate: float


class PredictionLog(BaseModel):
    """
    Log entry for a prediction request
    """

    timestamp: datetime
    request_id: str
    model_version: str
    prediction: int
    probability: float
    inference_time_ms: float
    success: bool
    error_message: Optional[str] = None
    input_features: dict


class ModelVersion(BaseModel):
    """
    Information about a model version
    """

    version: str
    model_type: str
    created_at: datetime
    metrics: dict
    file_path: str
    is_active: bool
    description: Optional[str] = None


class HealthStatus(BaseModel):
    """
    Health check response
    """

    status: str = Field(..., description="Status: 'healthy', 'degraded', or 'unhealthy'")
    model_loaded: bool
    model_version: Optional[str] = None
    database_connected: bool
    timestamp: datetime
    uptime_seconds: float
    version: str

