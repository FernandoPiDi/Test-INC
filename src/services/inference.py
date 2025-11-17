"""
Inference service for model predictions
Handles model loading, versioning, and prediction logic
"""
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder

from types.inference import ModelInfo, PredictionRequest, PredictionResponse
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """
    Handles loading and managing ML models with versioning
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.current_model: Optional[object] = None
        self.current_model_info: Optional[ModelInfo] = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_names: list[str] = []
        self._load_latest_model()

    def _load_latest_model(self) -> None:
        """
        Load the latest model version from the models directory
        """
        try:
            # Look for model files in the models directory
            model_files = list(self.models_dir.glob("modelo_rf_*.pkl"))
            if not model_files:
                # Fallback to root directory for backward compatibility
                root_models = list(Path(".").glob("modelo_rf_*.pkl"))
                if root_models:
                    model_files = root_models
                else:
                    # Try the old format
                    old_model = Path("modelo_rf.pkl")
                    if old_model.exists():
                        model_files = [old_model]

            if not model_files:
                logger.warning("No model files found. Inference will not work.")
                return

            # Sort by modification time (newest first)
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

            # Extract version from filename or use timestamp
            version = self._extract_version(latest_model)

            logger.info(f"Loading model from {latest_model} (version: {version})")

            with open(latest_model, "rb") as f:
                self.current_model = pickle.load(f)

            # Load feature names
            feature_file = latest_model.parent / f"feature_names_{version}.pkl"
            if not feature_file.exists():
                # Try old format
                feature_file = Path("feature_names.pkl")

            if feature_file.exists():
                with open(feature_file, "rb") as f:
                    self.feature_names = pickle.load(f)
            else:
                logger.warning("Feature names file not found. Using default features.")
                self.feature_names = [
                    "edad",
                    "count_consultas",
                    "count_laboratorios",
                    "avg_resultado_numerico",
                    "avg_biopsia",
                    "avg_vpH",
                    "avg_marcador_ca125",
                    "avg_psa",
                    "avg_colonoscopia",
                    "zona_residencia_encoded",
                    "tipo_cancer_encoded",
                ]

            # Load label encoders if they exist
            encoder_file = latest_model.parent / f"label_encoders_{version}.pkl"
            if encoder_file.exists():
                with open(encoder_file, "rb") as f:
                    self.label_encoders = pickle.load(f)

            self.current_model_info = ModelInfo(
                model_name="Random Forest",
                model_version=version,
                model_type="RandomForestClassifier",
                features=self.feature_names,
                loaded_at=datetime.now().isoformat(),
                model_path=str(latest_model),
            )

            logger.info(
                f"Model loaded successfully. Version: {version}, Features: {len(self.feature_names)}"
            )

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            self.current_model = None

    def _extract_version(self, model_path: Path) -> str:
        """
        Extract version from model filename
        Format: modelo_rf_v1.0.0.pkl or modelo_rf_20240101_120000.pkl
        """
        filename = model_path.stem  # Remove .pkl extension
        if "_v" in filename:
            # Extract version like v1.0.0
            version = filename.split("_v")[1]
        elif filename.startswith("modelo_rf_"):
            # Extract timestamp version
            version = filename.replace("modelo_rf_", "")
        else:
            # Use modification time as version
            mtime = model_path.stat().st_mtime
            version = datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")

        return version

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.current_model is not None

    def get_model_info(self) -> Optional[ModelInfo]:
        """Get current model information"""
        return self.current_model_info

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Make a prediction using the loaded model
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Cannot make predictions.")

        start_time = time.time()

        try:
            # Prepare features in the correct order
            features = self._prepare_features(request)

            # Make prediction
            probability = self.current_model.predict_proba(features)[0, 1]  # type: ignore[union-attr]
            prediction = int(probability > 0.5)

            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            return PredictionResponse(
                prediction=prediction,
                probability=float(probability),
                model_version=self.current_model_info.model_version,  # type: ignore[union-attr]
                model_name=self.current_model_info.model_name,  # type: ignore[union-attr]
                inference_time_ms=round(inference_time, 2),
            )

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def _prepare_features(self, request: PredictionRequest) -> np.ndarray:
        """
        Prepare features from request in the correct format for the model
        """
        # Encode categorical variables if encoders are available
        zona_residencia_encoded = self._encode_categorical(
            "zona_residencia", request.zona_residencia
        )
        tipo_cancer_encoded = self._encode_categorical(
            "tipo_cancer", request.tipo_cancer
        )

        # Build feature array in the correct order
        features = np.array(
            [
                [
                    request.edad,
                    request.count_consultas,
                    request.count_laboratorios,
                    request.avg_resultado_numerico,
                    request.avg_biopsia,
                    request.avg_vpH,
                    request.avg_marcador_ca125,
                    request.avg_psa,
                    request.avg_colonoscopia,
                    zona_residencia_encoded,
                    tipo_cancer_encoded,
                ]
            ]
        )

        return features

    def _encode_categorical(self, feature_name: str, value: str) -> int:
        """
        Encode categorical value using label encoder if available
        Otherwise use a simple hash-based encoding
        """
        if feature_name in self.label_encoders:
            try:
                return self.label_encoders[feature_name].transform([value])[0]
            except ValueError:
                # Value not seen during training, use most common class
                logger.warning(
                    f"Unknown value '{value}' for {feature_name}. Using default encoding."
                )
                return 0
        else:
            # Simple hash-based encoding as fallback
            return hash(value) % 100


# Global model loader instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """
    Get or create the global model loader instance
    """
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


def reload_model() -> None:
    """
    Reload the latest model version
    """
    global _model_loader
    _model_loader = ModelLoader()
    logger.info("Model reloaded successfully")

