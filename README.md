# API de Predicción de Adherencia de Pacientes Oncológicos

[![CI/CD Pipeline](https://github.com/USUARIO/REPO/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/USUARIO/REPO/actions/workflows/ci-cd.yml)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue)](https://github.com/USUARIO/REPO/pkgs/container/REPO)
[![Python](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)](https://fastapi.tiangolo.com/)

## 📋 Descripción del Proyecto

API desarrollada con FastAPI para predecir la adherencia a tratamientos de pacientes oncológicos utilizando Machine Learning. El sistema procesa datos de consultas médicas y resultados de laboratorio para generar predicciones mediante modelos de XGBoost y Redes Neuronales.

---

## 🎯 Solución a la Prueba Técnica

Este proyecto resuelve los siguientes requerimientos del examen técnico:

### **Parte 1: Ingeniería de Datos**

#### **a) Bases de datos: Consulta de información consolidada por paciente**

**Endpoint:** `GET /laboratorio/dataset`

```bash
curl -X GET "http://localhost:8000/laboratorio/dataset" \
     -H "accept: application/json"
```

**Descripción:** Retorna un dataset consolidado que incluye:

- Información del paciente (demografía y diagnóstico)
- Número total de consultas por paciente
- Número total de laboratorios por paciente
- Promedio de resultados numéricos por tipo de prueba (biopsia, VPH, CA125, PSA, colonoscopia)

**Respuesta:** Array de objetos con toda la información consolidada por paciente.

---

#### **b) Procesamiento de datos: Limpieza y normalización**

**Endpoint:** `PUT /laboratorio/procesamiento/limpieza`

```bash
curl -X PUT "http://localhost:8000/laboratorio/procesamiento/limpieza" \
     -H "accept: application/json"
```

**Descripción:** Ejecuta un pipeline de limpieza que incluye:

- ✅ Normalización de texto (minúsculas, eliminación de tildes)
- ✅ Estandarización de valores categóricos
- ✅ Corrección de outliers en resultados numéricos mediante Winsorización (IQR)
- ✅ **Imputación inteligente de valores faltantes:**
  - Columnas con < 5% faltantes: Valores por defecto o mediana/moda
  - Columnas con 5-20% faltantes: Mediana (numéricos) o moda (categóricos)
  - Columnas con > 20% faltantes: Imputación con 0 para columnas numéricas
- ✅ Validación de tipos de datos
- ✅ **Garantía de datos sin NaN** - Los valores nulos se imputan directamente en la base de datos

**Respuesta:** Reporte detallado con:

- Total de registros procesados por tabla
- Cambios realizados (normalizaciones, imputaciones, correcciones)
- Análisis detallado de valores faltantes con recomendaciones
- Outliers detectados y corregidos
- Tiempo de procesamiento

**⚠️ Importante:** Este endpoint debe ejecutarse **antes** de generar el dataset de modelado para garantizar que los datos estén completamente limpios.

---

#### **c) Dataset para modelado: Generación de dataset listo para ML**

**Endpoint:** `GET /laboratorio/dataset/modelado`

```bash
curl -X GET "http://localhost:8000/laboratorio/dataset/modelado" \
     -H "accept: application/json"
```

**Descripción:** Genera un dataset optimizado para Machine Learning:

- ✅ Una fila por paciente
- ✅ Todas las variables agregadas (conteos, promedios por tipo de prueba)
- ✅ **Sin valores nulos** - Datos completamente limpios desde la base de datos
- ✅ Tipos de datos correctos (numéricos como float, categóricos como string)
- ✅ Guardado como CSV con timestamp en `./data/dataset_modelado_YYYYMMDD_HHMMSS.csv`
- ✅ **Manejo robusto de NaN** - Valores `nan` de SQL se convierten automáticamente a 0.0

**Flujo recomendado:**

1. Cargar datos desde Excel: `POST /laboratorio/datos`
2. Ejecutar limpieza: `PUT /laboratorio/procesamiento/limpieza`
3. Generar dataset: `GET /laboratorio/dataset/modelado` ← Este endpoint

**Respuesta:**

```json
{
  "ruta_archivo": "./data/dataset_modelado_YYYYMMDD_HHMMSS.csv",
  "total_registros": 1000,
  "descripcion": "Dataset consolidado con 1000 pacientes. Incluye: datos demográficos, clínicos, conteos de consultas/laboratorios, y promedios de resultados por tipo de prueba. Listo para modelado de Machine Learning."
}
```

---

### **Parte 2: Machine Learning**

#### **a) Entrenamiento de modelos**

**Endpoint:** `POST /laboratorio/modelado/entrenar`

```bash
# Entrenar modelo XGBoost (Gradient Boosting)
curl -X POST "http://localhost:8000/laboratorio/modelado/entrenar" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"tipo_modelo": "xgboost"}'

# Entrenar modelo de Red Neuronal (TensorFlow)
curl -X POST "http://localhost:8000/laboratorio/modelado/entrenar" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{"tipo_modelo": "neural_network"}'
```

**Descripción:** Entrena modelos de predicción de adherencia:

- **XGBoost**: Modelo basado en HistGradientBoostingClassifier de scikit-learn
- **Neural Network**: Red neuronal profunda con TensorFlow/Keras

**Proceso automático:**

1. Carga del dataset más reciente de `./data/`
2. Codificación de variables categóricas (Label Encoding)
3. Split 80/20 (entrenamiento/test)
4. Entrenamiento del modelo
5. Evaluación con métricas estándar
6. Guardado automático en `./models/` con timestamp

**Respuesta:** Métricas completas del modelo

```json
{
  "modelo": "xgboost",
  "metricas_train": {
    "accuracy": 0.95,
    "precision": 0.94,
    "recall": 0.96,
    "f1_score": 0.95,
    "auc": 0.97
  },
  "metricas_test": {
    "accuracy": 0.92,
    "precision": 0.91,
    "recall": 0.93,
    "f1_score": 0.92,
    "auc": 0.94
  },
  "tiempo_entrenamiento": 45.23,
  "total_registros": 1000,
  "features_utilizados": 14,
  "fecha_entrenamiento": "2024-11-17T23:12:43"
}
```

---

#### **b) Predicción de adherencia**

**Endpoint:** `POST /laboratorio/predecir`

```bash
curl -X POST "http://localhost:8000/laboratorio/predecir" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
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
  "avg_colonoscopia": 0.0
}'
```

**Descripción:** Predice la adherencia a 12 meses de un paciente usando el modelo especificado.

**Parámetros:**

- `tipo_modelo`: `"xgboost"` o `"neural_network"`
- Datos demográficos: sexo, edad, zona_residencia
- Datos clínicos: tipo_cancer, estadio, aseguradora
- Variables agregadas: conteos de consultas/laboratorios, promedios de resultados

**Respuesta:**

```json
{
  "prediction": 1,
  "probability": 0.87,
  "model_version": "xgboost_20241117_231152",
  "model_name": "xgboost",
  "inference_time_ms": 15.43
}
```

- `prediction`: 0 = No adherente, 1 = Adherente
- `probability`: Probabilidad de adherencia (0.0 a 1.0)

---

#### **c) Evaluación con métricas técnicas**

Las métricas de **accuracy, precision, recall, F1-Score y AUC** se calculan automáticamente durante el entrenamiento y se retornan en la respuesta del endpoint `POST /laboratorio/modelado/entrenar`. Cada modelo incluye métricas tanto para el conjunto de entrenamiento (`metricas_train`) como para el conjunto de test (`metricas_test`).

**Ver sección (d) Comparación de Modelos** más abajo para los resultados completos.

---

#### **d) Comparación de Modelos**

**Resultados del entrenamiento (80 registros, datos limpios):**

| Métrica | XGBoost | Neural Network |
|---------|---------|----------------|
| Accuracy Test | **81.25%** | 68.75% |
| F1-Score Test | **0.88** | 0.78 |
| AUC Test | **0.60** | 0.58 |
| Tiempo Entrenamiento | ~0.2s | ~8s |
| Inference Time | **~5ms** | ~15ms |

**¿Cuál funciona mejor?**

**XGBoost** supera a la red neuronal en todas las métricas. Con 81.25% de accuracy vs 68.75%, XGBoost demuestra mejor capacidad para aprender patrones con datasets pequeños (< 1000 registros). Los modelos basados en árboles de decisión son más robustos con datos limitados, mientras que las redes neuronales requieren mayor cantidad de ejemplos para generalizar correctamente.

**Impacto de la limpieza de datos:**

- XGBoost mejoró de 68% a **81.25%** (+13%) gracias a la imputación de NaN
- Neural Network mejoró de 31% a **68.75%** (+37%) al eliminar valores faltantes

**¿Cuál es más fácil de desplegar?**

**XGBoost** es significativamente más sencillo:

- **Artifact único:** 1 archivo `.pkl` (~100KB) vs modelo `.keras` + scaler `.pkl`
- **Dependencias ligeras:** scikit-learn (~50MB) vs TensorFlow (~500MB)
- **Velocidad:** 3x más rápido en inferencia (5ms vs 15ms)
- **Recursos:** ~50MB RAM vs ~200MB RAM
- **Compatibilidad:** Cualquier servidor Python, sin necesidad de GPU

**Recomendación:** Para este caso de uso con datos limitados y requisitos de producción, **XGBoost es la mejor opción** tanto en rendimiento (81% accuracy) como en facilidad de deployment. La red neuronal solo se justificaría con >1000 registros de entrenamiento.

---

### **Parte 3: Análisis y Visualización**

#### **Dashboard por tipo de cáncer**

**Endpoint:** `GET /laboratorio/dashboard/{tipo_cancer}`

```bash
curl -X GET "http://localhost:8000/laboratorio/dashboard/Mama" \
     -H "accept: application/json"
```

**Descripción:** Retorna estadísticas agregadas para un tipo específico de cáncer:

- Total de pacientes
- Distribución por estadio
- Tasa de adherencia
- Promedios de edad
- Distribución por aseguradora
- Estadísticas de consultas y laboratorios

---

#### **Pacientes con actividad reciente**

**Endpoint:** `GET /laboratorio/pacientes/activos`

```bash
curl -X GET "http://localhost:8000/laboratorio/pacientes/activos?dias=30" \
     -H "accept: application/json"
```

**Descripción:** Lista pacientes con consultas o laboratorios en los últimos N días.

---

#### **Análisis de laboratorios por paciente**

**Endpoint:** `GET /laboratorio/analisis/paciente/{id_paciente}`

```bash
curl -X GET "http://localhost:8000/laboratorio/analisis/paciente/PAC001" \
     -H "accept: application/json"
```

**Descripción:** Análisis detallado de resultados de laboratorio de un paciente específico.

---

## 🔄 CI/CD Pipeline

Pipeline automatizado con GitHub Actions:

- ✅ **Ruff**: Validación de código
- ✅ **Pyright**: Type checking
- 🐳 **Docker Build**: Construcción automática
- 📦 **Registry**: Publicación en ghcr.io

### Pull de la imagen

```bash
docker pull ghcr.io/TU_USUARIO/test-inc:latest
```

---

## 🚀 Instalación y Configuración

### Opción 1: Docker (Recomendado)

#### 1. Crear archivo `.env`

```bash
DB_USERNAME=postgres
DB_PASSWORD=postgres123
DB_HOST=localhost
DB_PORT=5432
DB_NAME=adherencia_db
```

#### 2. Iniciar servicios

```bash
# Construir y levantar contenedores
docker-compose up --build

# En modo background
docker-compose up -d --build
```

**¿Qué hace Docker Compose?**

1. ✅ Inicia PostgreSQL con la base de datos
2. ✅ Espera a que PostgreSQL esté listo (health check)
3. ✅ Ejecuta migraciones de Alembic automáticamente
4. ✅ Inicia la API en el puerto 8000

La API estará disponible en: <http://localhost:8000>

---

### Opción 2: Instalación Local

#### 1. Instalar dependencias

```bash
# Usando uv (recomendado)
pip install uv
uv pip install -r pyproject.toml

# O usando pip directamente
pip install -e .
```

#### 2. Configurar base de datos

Editar `.env` con las credenciales de PostgreSQL local.

#### 3. Ejecutar migraciones

```bash
alembic upgrade head
```

#### 4. Iniciar la API

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📊 Flujo de Trabajo Completo

Para resolver la prueba técnica, sigue este flujo **en orden**:

### 0️⃣ **Cargar Datos Iniciales** (Si la BD está vacía)

```bash
# Subir archivo Excel con datos
curl -X POST "http://localhost:8000/laboratorio/datos" \
     -F "file=@./data/Dataset_prueba.xlsx"
```

**Resultado esperado:** ✓ 80 pacientes, 596 consultas, 430 laboratorios

### 1️⃣ **Procesar y Limpiar Datos** ⚠️ CRÍTICO

```bash
# Limpiar, normalizar e imputar valores faltantes
curl -X PUT "http://localhost:8000/laboratorio/procesamiento/limpieza"
```

**¿Por qué es crítico?**

- Imputa ~228 registros con NaN en `resultado_numerico` (53% de los datos)
- Normaliza variables categóricas
- Corrige outliers en datos numéricos
- **Garantiza dataset 100% limpio para ML**

**Resultado esperado:** ✓ Reporte con imputaciones realizadas

### 2️⃣ **Generar Dataset para Modelado**

```bash
# Crear dataset optimizado para ML (usa datos limpios de paso 1)
curl -X GET "http://localhost:8000/laboratorio/dataset/modelado"
```

**Resultado esperado:** ✓ CSV sin valores vacíos en `./data/dataset_modelado_YYYYMMDD_HHMMSS.csv`

### 3️⃣ **Entrenar Modelos**

```bash
# Entrenar XGBoost
curl -X POST "http://localhost:8000/laboratorio/modelado/entrenar" \
     -H "Content-Type: application/json" \
     -d '{"tipo_modelo": "xgboost"}'

# Entrenar Red Neuronal
curl -X POST "http://localhost:8000/laboratorio/modelado/entrenar" \
     -H "Content-Type: application/json" \
     -d '{"tipo_modelo": "neural_network"}'
```

### 4️⃣ **Realizar Predicciones**

```bash
# Predecir adherencia de un paciente con XGBoost
curl -X POST "http://localhost:8000/laboratorio/predecir" \
     -H "Content-Type: application/json" \
     -d '{
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
  "avg_colonoscopia": 0.0
}'
```

**Resultado esperado:**

```json
{
  "prediction": 1,
  "probability": 0.6094,
  "model_version": "xgboost_20251118_000450",
  "model_name": "xgboost",
  "inference_time_ms": 5.44
}
```

**Nota:** Cambia `"tipo_modelo"` a `"neural_network"` para usar el modelo de red neuronal.

---

## 📚 Documentación Interactiva

Una vez iniciada la API, accede a:

- **Swagger UI**: <http://localhost:8000/docs>
- **ReDoc**: <http://localhost:8000/redoc>
- **Health Check**: <http://localhost:8000/health>

La documentación interactiva permite:

- 🔍 Explorar todos los endpoints
- 📝 Ver esquemas de request/response
- ▶️ Probar endpoints directamente desde el navegador
- 📖 Leer descripciones detalladas de cada operación

---

## 🏗️ Arquitectura del Proyecto

```
Test-INC/
├── src/
│   ├── main.py                    # Aplicación principal FastAPI
│   ├── models/
│   │   └── tables.py              # Modelos SQLModel (ORM)
│   ├── routes/
│   │   ├── data.py                # Endpoints de datos y procesamiento
│   │   └── inference.py           # Endpoints de ML y predicción
│   ├── services/
│   │   ├── data.py                # Lógica de procesamiento de datos
│   │   ├── inference.py           # Lógica de ML y predicción
│   │   └── monitoring.py          # Monitoreo de predicciones
│   ├── app_types/
│   │   ├── data.py                # Tipos para datos
│   │   ├── inference.py           # Tipos para ML
│   │   └── monitoring.py          # Tipos para monitoreo
│   └── utils/
│       ├── settings.py            # Configuración y variables de entorno
│       └── logging_config.py      # Configuración de logging
├── alembic/                       # Migraciones de base de datos
├── models/                        # Modelos ML entrenados (*.pkl, *.keras)
├── data/                          # Datasets generados (*.csv)
├── logs/                          # Logs de la aplicación
├── docker-compose.yml             # Configuración Docker Compose
├── Dockerfile                     # Imagen Docker de la API
├── entrypoint.sh                  # Script de inicio (ejecuta migraciones)
└── pyproject.toml                 # Dependencias del proyecto
```

---

## 🔧 Stack Tecnológico

### Backend

- **FastAPI** 0.115+: Framework web moderno y rápido
- **Python** 3.13: Lenguaje de programación
- **SQLModel**: ORM para PostgreSQL
- **Pydantic**: Validación de datos
- **Alembic**: Migraciones de base de datos

### Machine Learning

- **scikit-learn**: Preprocesamiento y XGBoost
- **TensorFlow/Keras**: Redes Neuronales
- **pandas**: Manipulación de datos
- **numpy**: Computación numérica

### Base de Datos

- **PostgreSQL** 17: Base de datos relacional
- **psycopg3**: Driver de PostgreSQL

### DevOps

- **Docker**: Contenedorización
- **Docker Compose**: Orquestación de servicios
- **uv**: Gestor de paquetes de Python

---

## 📈 Modelos de Machine Learning

### XGBoost (HistGradientBoostingClassifier)

- **Algoritmo**: Gradient Boosting optimizado
- **Ventajas**: Rápido, maneja datos desbalanceados, robusto
- **Hiperparámetros**:
  - `max_depth`: 10
  - `learning_rate`: 0.1
  - `n_estimators`: 100

### Red Neuronal (TensorFlow)

- **Arquitectura**:
  - Input layer: 14 features
  - Hidden layers: [128, 64, 32] neuronas con ReLU
  - Dropout: 0.3 para regularización
  - Output layer: 1 neurona con sigmoid
- **Optimizador**: Adam
- **Loss**: Binary Crossentropy
- **Epochs**: 50 con early stopping

### Preprocesamiento

- **Label Encoding** para variables categóricas (`zona_residencia`, `tipo_cancer`)
- **StandardScaler** para normalización de features numéricas (solo Neural Network)
- **Imputación automática de NaN** durante entrenamiento (fallback de seguridad)
- **Corrección de outliers** mediante Winsorización en el endpoint de limpieza
- **Orden correcto de features** - Se preserva el orden de columnas usado durante entrenamiento

**Nota:** La imputación durante entrenamiento es un fallback. Los datos deberían estar limpios desde el endpoint `/limpieza`.

---

## 📊 Métricas de Evaluación

Los modelos se evalúan con:

- **Accuracy**: Precisión general
- **Precision**: Verdaderos positivos / (VP + FP)
- **Recall**: Verdaderos positivos / (VP + FN)
- **F1-Score**: Media armónica de precision y recall
- **AUC-ROC**: Área bajo la curva ROC

---

## 🐳 Docker: Detalles Técnicos

### Servicios

#### PostgreSQL

- **Imagen**: `postgres:17-alpine`
- **Puerto**: 5432
- **Volumen persistente**: `postgres_data`
- **Health check**: `pg_isready`

#### API FastAPI

- **Base**: `python:3.13-slim`
- **Puerto**: 8000
- **Volúmenes montados**:
  - `./models`: Modelos entrenados
  - `./data`: Datasets CSV
  - `./logs`: Logs de aplicación

### Flujo de Inicio

```
1. docker-compose up
   ↓
2. PostgreSQL inicia y pasa health check
   ↓
3. API espera a PostgreSQL (depends_on: service_healthy)
   ↓
4. entrypoint.sh ejecuta:
   - Espera conexión a PostgreSQL
   - Corre: alembic upgrade head
   - Inicia: uvicorn
   ↓
5. API lista en http://localhost:8000
```

---

## 🧪 Testing

### Test Manual con curl

Ver ejemplos de curl en cada sección de endpoints arriba.

### Test con Swagger UI

1. Abrir <http://localhost:8000/docs>
2. Expandir el endpoint deseado
3. Clic en "Try it out"
4. Llenar parámetros
5. Clic en "Execute"

### Test Automatizado

```bash
# Ejecutar tests (si están disponibles)
pytest test/
```

---

## 📝 Logging y Monitoreo

### Logs

Los logs se guardan en:

- **Consola**: Nivel INFO
- **Archivo**: `logs/app.log` con nivel DEBUG

### Monitoreo de Predicciones

Cada predicción se registra automáticamente con:

- Request ID único
- Versión del modelo
- Predicción y probabilidad
- Tiempo de inferencia
- Features de entrada
- Estado de éxito/error

---

## ⚠️ Troubleshooting

### Error: "No se encontró ningún modelo"

**Solución**: Entrenar un modelo primero usando el endpoint de entrenamiento.

### Error: "Dataset no encontrado"

**Solución**: Generar el dataset primero usando `GET /laboratorio/dataset/modelado`.

### Error de conexión a PostgreSQL

**Solución**:

1. Verificar que Docker Compose está corriendo
2. Revisar credenciales en `.env`
3. Verificar logs: `docker-compose logs postgres`

### Puerto 8000 ocupado

**Solución**:

```bash
# Ver qué proceso usa el puerto
sudo lsof -i :8000
# Cambiar puerto en docker-compose.yml o matar el proceso
```

---

## 🔧 Mejoras Técnicas Implementadas

### Manejo Robusto de Valores NaN

El sistema implementa un enfoque de múltiples capas para garantizar datos limpios:

#### 1. **Limpieza en la Fuente** (`/laboratorio/procesamiento/limpieza`)

- Imputa valores NaN directamente en la base de datos
- Estrategia adaptativa según porcentaje de valores faltantes
- Columnas numéricas con >20% faltantes: Imputación con 0 (antes se rechazaba)
- Ejemplo: `resultado_numerico` con 53% faltantes → 228 registros imputados con 0

#### 2. **Generación de Dataset** (`/laboratorio/dataset/modelado`)

- Detección de valores `nan` de tipo float retornados por SQL
- Conversión automática mediante función `safe_float()` que maneja:
  - `None` → 0.0
  - `float('nan')` → 0.0
  - Valores válidos → preservados
- Verificación final con `df.fillna(0)` como fallback

#### 3. **Entrenamiento de Modelos**

- Imputación adicional durante entrenamiento (fallback de seguridad)
- Preservación del orden de features entre entrenamiento y predicción
- Guardado de `label_encoders` y `feature_names` con cada modelo

#### 4. **Predicciones**

- Codificación correcta de variables categóricas a `*_encoded`
- Orden garantizado de columnas usando metadata del modelo
- Manejo de valores desconocidos en encoding (fallback a 0)

### Resultados de las Mejoras

**Antes:**

- Dataset con campos vacíos (`,,`)
- Warnings de imputación durante entrenamiento
- Accuracy ~68%

**Después:**

- Dataset 100% sin valores vacíos ✅
- Sin warnings de NaN ✅
- Accuracy ~81% ✅ (mejora del 13%)
- Pipeline completo sin errores ✅

### Acceso a Datos SQL

El sistema usa `text()` de SQLAlchemy para queries complejas. Se implementó:

- Acceso por índice (`row[0]`, `row[1]`) en lugar de atributos
- Compatible con resultados de tipo tupla
- Aplicado en endpoints:
  - `/laboratorio/analisis/dias-lab-diagnostico`
  - `/laboratorio/dataset/modelado`

---

## 👨‍💻 Desarrollo

### Agregar dependencias

```bash
# Agregar al pyproject.toml y ejecutar:
uv pip install -r pyproject.toml
```

### Crear nueva migración

```bash
# Después de modificar models/tables.py
alembic revision --autogenerate -m "descripción del cambio"
alembic upgrade head
```

### Validación de código

```bash
# Type checking
pyright src/

# Linting
ruff check src/
```

---

## 📄 Licencia

Este proyecto fue desarrollado como prueba técnica para el Laboratorio de Cocreación.

---

## 🤝 Contacto

Para dudas sobre la implementación o prueba técnica, contactar al equipo de desarrollo.

---

## 🎯 Checklist de Entrega

- ✅ Base de datos PostgreSQL con esquema definido
- ✅ Endpoints de consulta y agregación de datos
- ✅ Pipeline de limpieza y procesamiento de datos
- ✅ Generación de dataset para modelado
- ✅ Entrenamiento de modelo XGBoost
- ✅ Entrenamiento de Red Neuronal
- ✅ Endpoint de predicción con ambos modelos
- ✅ Documentación interactiva (Swagger)
- ✅ Docker Compose funcional
- ✅ Migraciones automáticas con Alembic
- ✅ Logging y monitoreo de predicciones
- ✅ Código validado (pyright + ruff)
- ✅ README completo con ejemplos

---

**¡API lista para demostración! 🚀**
