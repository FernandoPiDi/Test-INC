# Modelos Entrenados

Esta carpeta contiene los modelos de Machine Learning entrenados por la aplicación.

## Estructura de Archivos

### Gradient Boosting (scikit-learn)
- **Formato**: `xgboost_YYYYMMDD_HHMMSS.pkl`
- **Contenido**:
  - Modelo HistGradientBoostingClassifier (scikit-learn) entrenado
  - Lista de features utilizadas
  - Label encoders para variables categóricas
  - Métricas de evaluación
  - Fecha de entrenamiento

### Red Neuronal (TensorFlow)
- **Modelo**: `neural_network_YYYYMMDD_HHMMSS.keras`
  - Modelo de Keras/TensorFlow completo
  
- **Scaler y metadatos**: `neural_network_scaler_YYYYMMDD_HHMMSS.pkl`
  - StandardScaler para normalización
  - Lista de features utilizadas
  - Label encoders para variables categóricas
  - Métricas de evaluación
  - Fecha de entrenamiento

## Ejemplo de Nombres

```
xgboost_20250117_143025.pkl
neural_network_20250117_143127.keras
neural_network_scaler_20250117_143127.pkl
```

## Nota

Los archivos de modelos están excluidos del control de versiones (Git) debido a su tamaño.
Para entrenar nuevos modelos, utiliza el endpoint:

```bash
POST /laboratorio/modelado/entrenar
{
  "tipo_modelo": "xgboost"  # o "neural_network"
}
```

