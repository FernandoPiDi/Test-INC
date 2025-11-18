"""
Servicio de inferencia para predicciones y entrenamiento de modelos

Gestiona carga de modelos, versionado, lógica de predicción y entrenamiento de modelos.
"""

import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app_types.data import (
    MetricasModelo,
    ResultadoEntrenamiento,
)
from app_types.inference import (
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
    TipoModelo,
)
from services.data import (
    crear_carpeta_modelos,
    generar_nombre_modelo,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


def obtener_dataset_mas_reciente() -> pd.DataFrame:
    """
    Obtiene el dataset más reciente de la carpeta ./data

    Returns:
        DataFrame con el dataset cargado desde el CSV más reciente

    Raises:
        FileNotFoundError: Si no se encuentra ningún archivo de dataset
    """
    # Obtener raíz del workspace
    workspace_root = Path(__file__).parent.parent.parent
    carpeta_data = workspace_root / "data"

    # Buscar archivos dataset_modelado_*.csv
    archivos_dataset = list(carpeta_data.glob("dataset_modelado_*.csv"))

    if not archivos_dataset:
        raise FileNotFoundError(
            "No se encontró ningún archivo dataset_modelado_*.csv en ./data. "
            "Por favor, ejecuta GET /laboratorio/dataset/modelado primero."
        )

    # Obtener el más reciente por timestamp en el nombre o por fecha de modificación
    archivo_mas_reciente = max(archivos_dataset, key=lambda p: p.stat().st_mtime)

    logger.info(f"Cargando dataset desde: {archivo_mas_reciente}")

    # Cargar CSV
    df = pd.read_csv(archivo_mas_reciente)

    logger.info(f"Dataset cargado: {len(df)} registros, {len(df.columns)} columnas")

    return df


class ModelLoader:
    """
    Cargador y gestor de modelos de ML con versionado

    Gestiona la carga, versionado y predicciones de modelos de Machine Learning.
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
        Cargar la versión más reciente del modelo desde el directorio de modelos
        """
        try:
            # Buscar archivos de modelo en el directorio de modelos
            model_files = list(self.models_dir.glob("modelo_rf_*.pkl"))
            if not model_files:
                # Fallback al directorio raíz para compatibilidad hacia atrás
                root_models = list(Path(".").glob("modelo_rf_*.pkl"))
                if root_models:
                    model_files = root_models
                else:
                    # Intentar formato antiguo
                    old_model = Path("modelo_rf.pkl")
                    if old_model.exists():
                        model_files = [old_model]

            if not model_files:
                logger.warning(
                    "No se encontraron archivos de modelo. La inferencia no funcionará."
                )
                return

            # Ordenar por tiempo de modificación (más reciente primero)
            latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

            # Extraer versión del nombre de archivo o usar timestamp
            version = self._extract_version(latest_model)

            logger.info(f"Loading model from {latest_model} (version: {version})")

            with open(latest_model, "rb") as f:
                self.current_model = pickle.load(f)

            # Cargar nombres de features
            feature_file = latest_model.parent / f"feature_names_{version}.pkl"
            if not feature_file.exists():
                # Intentar formato antiguo
                feature_file = Path("feature_names.pkl")

            if feature_file.exists():
                with open(feature_file, "rb") as f:
                    self.feature_names = pickle.load(f)
            else:
                logger.warning(
                    "Archivo de nombres de features no encontrado. Usando features por defecto."
                )
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

            # Cargar label encoders si existen
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
                f"Modelo cargado exitosamente. Versión: {version}, Features: {len(self.feature_names)}"
            )

        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}", exc_info=True)
            self.current_model = None

    def _extract_version(self, model_path: Path) -> str:
        """
        Extraer versión del nombre de archivo del modelo

        Formato: modelo_rf_v1.0.0.pkl o modelo_rf_20240101_120000.pkl
        """
        filename = model_path.stem  # Remover extensión .pkl
        if "_v" in filename:
            # Extraer versión como v1.0.0
            version = filename.split("_v")[1]
        elif filename.startswith("modelo_rf_"):
            # Extraer versión timestamp
            version = filename.replace("modelo_rf_", "")
        else:
            # Usar tiempo de modificación como versión
            mtime = model_path.stat().st_mtime
            version = datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")

        return version

    def is_loaded(self) -> bool:
        """Verificar si el modelo está cargado"""
        return self.current_model is not None

    def get_model_info(self) -> Optional[ModelInfo]:
        """Obtener información del modelo actual"""
        return self.current_model_info

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Realizar predicción usando el modelo cargado

        Args:
            request: Solicitud de predicción con features del paciente

        Returns:
            Respuesta con predicción, probabilidad y metadatos

        Raises:
            RuntimeError: Si el modelo no está cargado
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


def preparar_features_dict(request: PredictionRequest) -> dict:
    """
    Prepara diccionario de features desde el request de predicción

    Args:
        request: Solicitud de predicción con datos del paciente

    Returns:
        Diccionario con todas las features del paciente (solo las usadas en entrenamiento)
    """
    # Solo incluir las features que realmente se usan en el modelo
    # Features numéricas: edad, count_consultas, count_laboratorios, avg_*, etc.
    # Features categóricas: zona_residencia, tipo_cancer
    return {
        "edad": request.edad,
        "count_consultas": request.count_consultas,
        "count_laboratorios": request.count_laboratorios,
        "avg_resultado_numerico": request.avg_resultado_numerico,
        "avg_biopsia": request.avg_biopsia,
        "avg_vpH": request.avg_vpH,
        "avg_marcador_ca125": request.avg_marcador_ca125,
        "avg_psa": request.avg_psa,
        "avg_colonoscopia": request.avg_colonoscopia,
        "zona_residencia": request.zona_residencia,
        "tipo_cancer": request.tipo_cancer,
    }


def codificar_features_categoricas(features_dict: dict, encoders: dict) -> dict:
    """
    Codifica variables categóricas usando los encoders guardados

    Args:
        features_dict: Diccionario con features del paciente
        encoders: Diccionario con label encoders por columna

    Returns:
        Diccionario con features codificadas (renombradas a *_encoded)
    """
    for col, encoder in encoders.items():
        if col in features_dict:
            try:
                valor_encoded = encoder.transform([features_dict[col]])[0]
                # Renombrar columna a *_encoded
                features_dict[f"{col}_encoded"] = valor_encoded
                # Eliminar columna original
                del features_dict[col]
            except ValueError:
                logger.warning(
                    f"Valor desconocido '{features_dict[col]}' para {col}. "
                    f"Usando codificación por defecto."
                )
                features_dict[f"{col}_encoded"] = 0
                del features_dict[col]
    return features_dict


def predecir_con_xgboost(request: PredictionRequest) -> tuple[int, float, str]:
    """
    Realiza predicción usando modelo XGBoost

    Args:
        request: Solicitud de predicción con datos del paciente

    Returns:
        Tupla con (predicción, probabilidad, versión_modelo)

    Raises:
        FileNotFoundError: Si no se encuentra el modelo
    """
    carpeta_modelos = Path("models")
    archivos_modelo = list(carpeta_modelos.glob("xgboost_*.pkl"))

    if not archivos_modelo:
        raise FileNotFoundError(
            "No se encontró ningún modelo de tipo 'xgboost'. "
            "Por favor entrena un modelo primero usando POST /laboratorio/modelado/entrenar"
        )

    archivo_mas_reciente = max(archivos_modelo, key=lambda p: p.stat().st_mtime)

    # Cargar modelo y metadatos
    with open(archivo_mas_reciente, "rb") as f:
        datos_guardados = pickle.load(f)

    modelo = datos_guardados["modelo"]
    encoders = datos_guardados.get("label_encoders", {})
    feature_names = datos_guardados.get("features", [])

    # Preparar y codificar features
    features_dict = preparar_features_dict(request)
    features_dict = codificar_features_categoricas(features_dict, encoders)

    # Convertir a DataFrame usando el orden correcto de features
    df_features = pd.DataFrame([features_dict])
    # Asegurar que las columnas estén en el orden correcto
    df_features = df_features[feature_names]
    X = df_features.values

    # Predicción
    probabilidad = float(modelo.predict_proba(X)[0, 1])
    prediccion = int(probabilidad > 0.5)
    version_modelo = archivo_mas_reciente.stem

    return prediccion, probabilidad, version_modelo


def predecir_con_neural_network(request: PredictionRequest) -> tuple[int, float, str]:
    """
    Realiza predicción usando Red Neuronal (TensorFlow)

    Args:
        request: Solicitud de predicción con datos del paciente

    Returns:
        Tupla con (predicción, probabilidad, versión_modelo)

    Raises:
        FileNotFoundError: Si no se encuentra el modelo o scaler
    """
    carpeta_modelos = Path("models")
    archivos_modelo = list(carpeta_modelos.glob("neural_network_*.keras"))

    if not archivos_modelo:
        raise FileNotFoundError(
            "No se encontró ningún modelo de tipo 'neural_network'. "
            "Por favor entrena un modelo primero usando POST /laboratorio/modelado/entrenar"
        )

    archivo_mas_reciente = max(archivos_modelo, key=lambda p: p.stat().st_mtime)

    # Cargar modelo
    modelo = tf.keras.models.load_model(archivo_mas_reciente)  # type: ignore[attr-defined]

    # Buscar el scaler correspondiente
    timestamp = archivo_mas_reciente.stem.replace("neural_network_", "")
    archivo_scaler = carpeta_modelos / f"neural_network_scaler_{timestamp}.pkl"

    if not archivo_scaler.exists():
        raise FileNotFoundError(
            f"No se encontró el scaler para el modelo neural network: {archivo_scaler}"
        )

    with open(archivo_scaler, "rb") as f:
        datos_scaler = pickle.load(f)

    scaler = datos_scaler["scaler"]
    encoders = datos_scaler.get("label_encoders", {})
    feature_names = datos_scaler.get("features", [])

    # Preparar y codificar features
    features_dict = preparar_features_dict(request)
    features_dict = codificar_features_categoricas(features_dict, encoders)

    # Convertir a DataFrame usando el orden correcto de features
    df_features = pd.DataFrame([features_dict])
    # Asegurar que las columnas estén en el orden correcto
    df_features = df_features[feature_names]
    X = df_features.values
    X_scaled = scaler.transform(X)

    # Predicción
    probabilidad = float(modelo.predict(X_scaled, verbose=0)[0, 0])
    prediccion = int(probabilidad > 0.5)
    version_modelo = archivo_mas_reciente.stem

    return prediccion, probabilidad, version_modelo


def predecir_con_modelo(request: PredictionRequest) -> PredictionResponse:
    """
    Realiza una predicción usando el modelo especificado en el request

    Args:
        request: Solicitud de predicción con el tipo de modelo y features del paciente

    Returns:
        Respuesta con predicción, probabilidad y metadatos

    Raises:
        FileNotFoundError: Si no se encuentra el modelo especificado
        RuntimeError: Si hay error durante la predicción
    """
    start_time = time.time()
    tipo_modelo = request.tipo_modelo.value

    try:
        # Llamar a la función específica según el tipo de modelo
        if tipo_modelo == "xgboost":
            prediccion, probabilidad, version_modelo = predecir_con_xgboost(request)
        elif tipo_modelo == "neural_network":
            prediccion, probabilidad, version_modelo = predecir_con_neural_network(
                request
            )
        else:
            raise ValueError(f"Tipo de modelo no soportado: {tipo_modelo}")

        inference_time = (time.time() - start_time) * 1000  # Convertir a ms

        return PredictionResponse(
            prediction=prediccion,
            probability=probabilidad,
            model_version=version_modelo,
            model_name=tipo_modelo,
            inference_time_ms=round(inference_time, 2),
        )

    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(
            f"Error durante la predicción con {tipo_modelo}: {str(e)}", exc_info=True
        )
        raise RuntimeError(f"Error en predicción: {str(e)}")


async def entrenar_modelo_neural_network() -> ResultadoEntrenamiento:
    """
    Entrena una red neuronal usando TensorFlow para predicción de adherencia

    Carga el dataset más reciente desde ./data/dataset_modelado_*.csv

    Returns:
        ResultadoEntrenamiento con métricas y info del modelo
    """
    inicio = time.time()

    # ===========================================================================
    # 1. OBTENER DATASET DESDE CSV
    # ===========================================================================
    df = obtener_dataset_mas_reciente()

    if len(df) < 10:
        raise ValueError("Dataset muy pequeño para entrenar (mínimo 10 registros)")

    # ===========================================================================
    # 2. PREPARACIÓN DE FEATURES
    # ===========================================================================

    # Features seleccionadas según requerimientos
    features_numericas = [
        "edad",
        "count_consultas",
        "count_laboratorios",
        "avg_resultado_numerico",
        "avg_biopsia",
        "avg_vpH",
        "avg_marcador_ca125",
        "avg_psa",
        "avg_colonoscopia",
    ]

    features_categoricas = ["zona_residencia", "tipo_cancer"]

    # Codificar variables categóricas
    label_encoders = {}
    df_encoded = df.copy()

    # Imputar NaN en features numéricas antes del entrenamiento
    for col in features_numericas:
        if df_encoded[col].isna().any():
            logger.warning(
                f"Columna {col} tiene {df_encoded[col].isna().sum()} valores NaN. Imputando con 0."
            )
            df_encoded[col] = df_encoded[col].fillna(0)

    for col in features_categoricas:
        le = LabelEncoder()
        df_encoded[f"{col}_encoded"] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Lista final de features
    features_finales = features_numericas + [
        f"{col}_encoded" for col in features_categoricas
    ]

    # Preparar X e y
    X = df_encoded[features_finales].values
    y = df_encoded["adherencia_12m"].values

    # ===========================================================================
    # 3. DIVISIÓN TRAIN/TEST
    # ===========================================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ===========================================================================
    # 4. CONSTRUCCIÓN DE RED NEURONAL
    # ===========================================================================
    modelo_nn = tf.keras.Sequential(  # type: ignore[attr-defined]
        [
            tf.keras.layers.Input(shape=(len(features_finales),)),  # type: ignore[attr-defined]
            tf.keras.layers.Dense(64, activation="relu"),  # type: ignore[attr-defined]
            tf.keras.layers.Dropout(0.3),  # type: ignore[attr-defined]
            tf.keras.layers.Dense(32, activation="relu"),  # type: ignore[attr-defined]
            tf.keras.layers.Dropout(0.2),  # type: ignore[attr-defined]
            tf.keras.layers.Dense(16, activation="relu"),  # type: ignore[attr-defined]
            tf.keras.layers.Dense(1, activation="sigmoid"),  # type: ignore[attr-defined]
        ]
    )

    modelo_nn.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # type: ignore[attr-defined]
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # ===========================================================================
    # 5. ENTRENAMIENTO
    # ===========================================================================
    modelo_nn.fit(
        X_train_scaled,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
    )

    # ===========================================================================
    # 6. PREDICCIONES Y EVALUACIÓN
    # ===========================================================================

    def calcular_metricas(
        y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> MetricasModelo:
        """Calcular métricas de evaluación"""
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0.0)  # type: ignore[arg-type]
        rec = recall_score(y_true, y_pred, zero_division=0.0)  # type: ignore[arg-type]
        f1 = f1_score(y_true, y_pred, zero_division=0.0)  # type: ignore[arg-type]

        # Calcular AUC
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = auc(fpr, tpr)
        except Exception:
            auc_score = 0.0

        return MetricasModelo(
            accuracy=round(float(acc), 4),
            precision=round(float(prec), 4),
            recall=round(float(rec), 4),
            f1_score=round(float(f1), 4),
            auc=round(float(auc_score), 4),
        )

    # Predicciones
    y_train_pred_proba = modelo_nn.predict(X_train_scaled, verbose=0)
    y_test_pred_proba = modelo_nn.predict(X_test_scaled, verbose=0)

    # Calcular métricas
    metricas_train = calcular_metricas(np.array(y_train), np.array(y_train_pred_proba))
    metricas_test = calcular_metricas(np.array(y_test), np.array(y_test_pred_proba))

    # ===========================================================================
    # 7. RESUMEN Y RESULTADO
    # ===========================================================================
    tiempo_total = time.time() - inicio

    # Información de arquitectura
    arquitectura = {
        "tipo": "Sequential Neural Network",
        "capas": [
            {"tipo": "Dense", "neuronas": 64, "activacion": "relu"},
            {"tipo": "Dropout", "rate": 0.3},
            {"tipo": "Dense", "neuronas": 32, "activacion": "relu"},
            {"tipo": "Dropout", "rate": 0.2},
            {"tipo": "Dense", "neuronas": 16, "activacion": "relu"},
            {"tipo": "Dense", "neuronas": 1, "activacion": "sigmoid"},
        ],
        "optimizer": "Adam (lr=0.001)",
        "loss": "binary_crossentropy",
        "epochs": 50,
        "batch_size": 32,
    }

    # Generar resumen
    resumen = (
        f"Modelo de Red Neuronal entrenado exitosamente. "
        f"Accuracy en test: {metricas_test.accuracy:.2%}, "
        f"F1-Score en test: {metricas_test.f1_score:.4f}, "
        f"AUC en test: {metricas_test.auc:.4f}"
    )

    # ===========================================================================
    # 6. GUARDAR MODELO
    # ===========================================================================
    carpeta_modelos = crear_carpeta_modelos()
    nombre_archivo_modelo = generar_nombre_modelo("neural_network", "keras")
    nombre_archivo_scaler = generar_nombre_modelo("neural_network_scaler", "pkl")
    ruta_modelo = carpeta_modelos / nombre_archivo_modelo
    ruta_scaler = carpeta_modelos / nombre_archivo_scaler

    # Guardar modelo de Keras
    modelo_nn.save(str(ruta_modelo))
    logger.info(f"Modelo de Red Neuronal guardado en: {ruta_modelo}")

    # Guardar scaler y metadatos
    scaler_data = {
        "scaler": scaler,
        "features": features_finales,
        "label_encoders": label_encoders,
        "metricas_test": metricas_test.model_dump(),
        "fecha_entrenamiento": datetime.now().isoformat(),
    }

    with open(ruta_scaler, "wb") as f:
        pickle.dump(scaler_data, f)

    logger.info(f"Scaler y metadatos guardados en: {ruta_scaler}")

    return ResultadoEntrenamiento(
        modelo="Red Neuronal (TensorFlow)",
        total_registros=len(df),
        registros_train=len(X_train),
        registros_test=len(X_test),
        features_utilizadas=features_finales,
        metricas_train=metricas_train,
        metricas_test=metricas_test,
        tiempo_entrenamiento_segundos=round(tiempo_total, 2),
        arquitectura=arquitectura,
        resumen=resumen,
    )


async def entrenar_modelo_xgboost() -> ResultadoEntrenamiento:
    """
    Entrena un modelo Gradient Boosting (scikit-learn) para predicción de adherencia

    Carga el dataset más reciente desde ./data/dataset_modelado_*.csv

    Returns:
        ResultadoEntrenamiento: Resultados del entrenamiento del modelo Gradient Boosting
    """
    logger.info("Iniciando entrenamiento de modelo Gradient Boosting (scikit-learn)")

    # ===========================================================================
    # 1. OBTENER DATASET DESDE CSV
    # ===========================================================================
    df = obtener_dataset_mas_reciente()

    if len(df) < 10:
        raise ValueError("Dataset muy pequeño para entrenar (mínimo 10 registros)")

    logger.info(f"Dataset cargado: {len(df)} registros")

    # ===========================================================================
    # 2. PREPARACIÓN DE FEATURES
    # ===========================================================================
    features_numericas = [
        "edad",
        "count_consultas",
        "count_laboratorios",
        "avg_resultado_numerico",
        "avg_biopsia",
        "avg_vpH",
        "avg_marcador_ca125",
        "avg_psa",
        "avg_colonoscopia",
    ]

    features_categoricas = ["zona_residencia", "tipo_cancer"]

    # Codificar variables categóricas
    label_encoders = {}
    df_encoded = df.copy()

    # Imputar NaN en features numéricas antes del entrenamiento
    for col in features_numericas:
        if df_encoded[col].isna().any():
            logger.warning(
                f"Columna {col} tiene {df_encoded[col].isna().sum()} valores NaN. Imputando con 0."
            )
            df_encoded[col] = df_encoded[col].fillna(0)

    for col in features_categoricas:
        le = LabelEncoder()
        df_encoded[f"{col}_encoded"] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Lista final de features
    features_finales = features_numericas + [
        f"{col}_encoded" for col in features_categoricas
    ]

    # Preparar X e y
    X = df_encoded[features_finales].values
    y = df_encoded["adherencia_12m"].values

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Split: {len(X_train)} train, {len(X_test)} test")

    # ===========================================================================
    # 3. ENTRENAR MODELO GRADIENT BOOSTING
    # ===========================================================================
    inicio = time.time()

    # Crear modelo HistGradientBoostingClassifier (scikit-learn)
    modelo_gb = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=0,
    )

    # Entrenar
    logger.info("Entrenando modelo Gradient Boosting...")
    modelo_gb.fit(X_train, y_train)

    # ===========================================================================
    # 4. PREDICCIONES Y EVALUACIÓN
    # ===========================================================================

    # Predicciones
    y_train_pred_proba = modelo_gb.predict_proba(X_train)[:, 1]
    y_test_pred_proba = modelo_gb.predict_proba(X_test)[:, 1]

    def calcular_metricas_modelo(
        y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> MetricasModelo:
        """Calcular métricas de evaluación"""
        y_pred = (y_pred_proba > 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0.0)  # type: ignore[arg-type]
        rec = recall_score(y_true, y_pred, zero_division=0.0)  # type: ignore[arg-type]
        f1 = f1_score(y_true, y_pred, zero_division=0.0)  # type: ignore[arg-type]

        # Calcular AUC
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc_score = auc(fpr, tpr)
        except Exception:
            auc_score = 0.0

        return MetricasModelo(
            accuracy=round(float(acc), 4),
            precision=round(float(prec), 4),
            recall=round(float(rec), 4),
            f1_score=round(float(f1), 4),
            auc=round(float(auc_score), 4),
        )

    # Calcular métricas
    metricas_train = calcular_metricas_modelo(
        np.array(y_train), np.array(y_train_pred_proba)
    )
    metricas_test = calcular_metricas_modelo(
        np.array(y_test), np.array(y_test_pred_proba)
    )

    tiempo_total = time.time() - inicio

    logger.info(
        f"Entrenamiento completado en {tiempo_total:.2f}s. "
        f"Test Accuracy: {metricas_test.accuracy:.4f}, F1: {metricas_test.f1_score:.4f}"
    )

    # ===========================================================================
    # 5. PREPARAR ARQUITECTURA Y FEATURE IMPORTANCE
    # ===========================================================================

    # Importancia de features
    # Nota: HistGradientBoostingClassifier no expone feature_importances_ directamente
    # Se puede calcular con permutation importance si es necesario en el futuro
    feature_importance = None

    arquitectura = {
        "tipo": "HistGradientBoostingClassifier (scikit-learn)",
        "max_iter": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "algoritmo": "histogram-based gradient boosting",
    }

    if feature_importance:
        arquitectura["feature_importance"] = feature_importance

    # Generar resumen
    resumen = (
        f"Modelo Gradient Boosting (scikit-learn) entrenado exitosamente. "
        f"Accuracy en test: {metricas_test.accuracy:.2%}, "
        f"F1-Score en test: {metricas_test.f1_score:.4f}, "
        f"AUC en test: {metricas_test.auc:.4f}"
    )

    # ===========================================================================
    # 6. GUARDAR MODELO
    # ===========================================================================
    carpeta_modelos = crear_carpeta_modelos()
    nombre_archivo = generar_nombre_modelo("xgboost", "pkl")
    ruta_modelo = carpeta_modelos / nombre_archivo

    # Guardar modelo y metadatos
    modelo_data = {
        "modelo": modelo_gb,
        "features": features_finales,
        "label_encoders": label_encoders,
        "metricas_test": metricas_test.model_dump(),
        "fecha_entrenamiento": datetime.now().isoformat(),
    }

    with open(ruta_modelo, "wb") as f:
        pickle.dump(modelo_data, f)

    logger.info(f"Modelo guardado en: {ruta_modelo}")

    return ResultadoEntrenamiento(
        modelo="Gradient Boosting (scikit-learn)",
        total_registros=len(df),
        registros_train=len(X_train),
        registros_test=len(X_test),
        features_utilizadas=features_finales,
        metricas_train=metricas_train,
        metricas_test=metricas_test,
        tiempo_entrenamiento_segundos=round(tiempo_total, 2),
        arquitectura=arquitectura,
        resumen=resumen,
    )


async def entrenar_modelo_especifico(tipo_modelo: TipoModelo) -> ResultadoEntrenamiento:
    """
    Entrena un modelo específico según el tipo solicitado

    Carga automáticamente el dataset más reciente desde ./data/dataset_modelado_*.csv

    Args:
        tipo_modelo: Tipo de modelo a entrenar (xgboost o neural_network)

    Returns:
        ResultadoEntrenamiento: Resultados del entrenamiento del modelo seleccionado

    Raises:
        ValueError: Si el tipo de modelo no es válido o no hay datos suficientes
        FileNotFoundError: Si no se encuentra el archivo de dataset
    """
    logger.info(f"Solicitud de entrenamiento recibida para modelo: {tipo_modelo}")

    if tipo_modelo == TipoModelo.XGBOOST:
        return await entrenar_modelo_xgboost()
    elif tipo_modelo == TipoModelo.NEURAL_NETWORK:
        return await entrenar_modelo_neural_network()
    else:
        raise ValueError(
            f"Tipo de modelo no válido: {tipo_modelo}. "
            f"Opciones válidas: {', '.join([m.value for m in TipoModelo])}"
        )
