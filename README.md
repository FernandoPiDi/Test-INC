# API de Predicción de Adherencia de Pacientes Oncológicos

## Descripción del Proyecto

API desarrollada con FastAPI para predecir la adherencia a tratamientos de pacientes oncológicos utilizando Machine Learning. El sistema procesa los datos de consultas médicas y resultados de laboratorio para generar predicciones mediante modelos de XGBoost y Redes Neuronales.

---

## Instalación y Configuración

### Docker

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

1. Inicia PostgreSQL con la base de datos
2. Espera a que PostgreSQL esté listo (health check)
3. Ejecuta migraciones de Alembic automáticamente
4. Inicia la API en el puerto 8000

La API estará disponible en: <http://localhost:8000>

## Flujo de Trabajo Completo para la API

### **Cargar Datos Iniciales** (Si la BD está vacía)

```bash
# Subir archivo Excel con datos
curl -X POST "http://localhost:8000/laboratorio/datos" \
     -F "file=@./data/Dataset_prueba.xlsx"
```

**Resultado esperado:**  80 pacientes, 596 consultas, 430 laboratorios

### **Procesar y Limpiar Datos**

```bash
# Limpiar, normalizar e imputar valores faltantes
curl -X PUT "http://localhost:8000/laboratorio/procesamiento/limpieza"
```

¿Por qué es crítico?

- Imputa ~228 registros con NaN en `resultado_numerico` (53% de los datos)
- Normaliza variables categóricas
- Corrige outliers en datos numéricos
- Garantiza dataset 100% limpio para ML

Resultado esperado: Reporte con imputaciones realizadas

### **Generar Dataset para Modelado**

```bash
# Crear dataset optimizado para ML (usa datos limpios de paso 1)
curl -X GET "http://localhost:8000/laboratorio/dataset/modelado"
```

Resultado esperado: ✓ CSV sin valores vacíos en `./data/dataset_modelado_YYYYMMDD_HHMMSS.csv`

### **Entrenar Modelos**

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

### **Realizar Predicciones**

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

Resultado esperado:

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

## Solución a la Prueba Técnica

Este proyecto resuelve los siguientes requerimientos del examen técnico:

### **Parte 1: Ingeniería de Datos**

La implementación sigue un proceso ETL (Extract, Transform, Load) clásico aplicado a datos médicos:

- **Extract (Extraer)**: `POST /laboratorio/datos` - Subir y cargar archivos Excel con datos crudos
- **Transform (Transformar)**: `PUT /laboratorio/procesamiento/limpieza` - Limpiar, normalizar y estandarizar datos
- **Load (Cargar)**: `GET /laboratorio/dataset/modelado` - Generar datasets optimizados listos para análisis y modelos

Este flujo garantiza que los datos médicos sean confiables y estén preparados para generar insights precisos.

#### **a) Bases de datos:**

**Endpoint:** `GET /laboratorio/dataset`

```bash
curl -X GET "http://localhost:8000/laboratorio/dataset" \
     -H "accept: application/json"
```

**Query SQL - Extract (Carga de datos desde Excel):**

```sql
-- Los datos se cargan desde Excel usando pandas.read_excel()
-- y se insertan en las tablas usando SQLAlchemy ORM

INSERT INTO paciente (
    id_paciente, sexo, edad, zona_residencia, fecha_dx,
    tipo_cancer, estadio, aseguradora, adherencia_12m
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);

INSERT INTO consulta (
    id_consulta, id_paciente, fecha_consulta, motivo,
    prioridad, especialista
) VALUES (?, ?, ?, ?, ?, ?);

INSERT INTO laboratorio (
    id_lab, id_paciente, fecha_muestra, tipo_prueba,
    resultado, resultado_numerico, unidad
) VALUES (?, ?, ?, ?, ?, ?, ?);
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

- Normalización de texto (minúsculas, eliminación de tildes)
- Estandarización de valores categóricos
- Corrección de outliers en resultados numéricos mediante Winsorización (IQR)
- Imputación inteligente de valores faltantes:
  - Columnas con < 5% faltantes: Valores por defecto o mediana/moda
  - Columnas con 5-20% faltantes: Mediana (numéricos) o moda (categóricos)
  - Columnas con > 20% faltantes: Imputación con 0 para columnas numéricas
- Validación de tipos de datos
- Garantía de datos sin NaN - Los valores nulos se imputan directamente en la base de datos

Respuesta: Reporte detallado con:

- Total de registros procesados por tabla
- Cambios realizados (normalizaciones, imputaciones, correcciones)
- Análisis detallado de valores faltantes con recomendaciones
- Outliers detectados y corregidos
- Tiempo de procesamiento

**Importante:** Este endpoint debe ejecutarse **antes** de generar el dataset de modelado para garantizar que los datos estén completamente limpios.

**Query SQL - Transform (Carga de datos para limpieza):**

```sql
-- Datos se cargan desde BD a pandas para procesamiento
SELECT * FROM paciente;
SELECT * FROM consulta;
SELECT * FROM laboratorio;
```

---

#### **c) Dataset para modelado: Generación de dataset listo para ML**

**Endpoint:** `GET /laboratorio/dataset/modelado`

```bash
curl -X GET "http://localhost:8000/laboratorio/dataset/modelado" \
     -H "accept: application/json"
```

**Descripción:** Genera un dataset optimizado para Machine Learning:

- Una fila por paciente
- Todas las variables agregadas (conteos, promedios por tipo de prueba)
- **Sin valores nulos** - Datos completamente limpios desde la base de datos
- Guardado como CSV con timestamp en `./data/dataset_modelado_YYYYMMDD_HHMMSS.csv`
- **Manejo robusto de NaN** - Valores `nan` de SQL se convierten automáticamente a 0.0

**Flujo recomendado:**

1. Cargar datos desde Excel: `POST /laboratorio/datos`
2. Ejecutar limpieza: `PUT /laboratorio/procesamiento/limpieza`
3. Generar dataset: `GET /laboratorio/dataset/modelado`

**Query SQL - Load (Generación del dataset final):**

```sql
WITH paciente_consultas AS (
    SELECT
        p.id_paciente,
        COUNT(c.id_consulta) as count_consultas
    FROM paciente p
    LEFT JOIN consulta c ON p.id_paciente = c.id_paciente
    GROUP BY p.id_paciente
),
paciente_labs AS (
    SELECT
        p.id_paciente,
        COUNT(l.id_lab) as count_laboratorios,
        COALESCE(AVG(l.resultado_numerico), 0) as avg_resultado_numerico,
        COALESCE(AVG(CASE WHEN LOWER(l.tipo_prueba) LIKE '%biopsia%' THEN l.resultado_numerico END), 0) as avg_biopsia,
        COALESCE(AVG(CASE WHEN LOWER(l.tipo_prueba) LIKE '%vph%' THEN l.resultado_numerico END), 0) as avg_vph,
        COALESCE(AVG(CASE WHEN LOWER(l.tipo_prueba) LIKE '%ca125%' OR LOWER(l.tipo_prueba) LIKE '%marcador%' THEN l.resultado_numerico END), 0) as avg_marcador_ca125,
        COALESCE(AVG(CASE WHEN LOWER(l.tipo_prueba) LIKE '%psa%' THEN l.resultado_numerico END), 0) as avg_psa,
        COALESCE(AVG(CASE WHEN LOWER(l.tipo_prueba) LIKE '%colonoscopia%' THEN l.resultado_numerico END), 0) as avg_colonoscopia
    FROM paciente p
    LEFT JOIN laboratorio l ON p.id_paciente = l.id_paciente
    GROUP BY p.id_paciente
)
SELECT
    p.sexo,
    p.edad,
    COALESCE(p.zona_residencia, 'Desconocida') as zona_residencia,
    p.tipo_cancer,
    p.estadio,
    p.aseguradora,
    COALESCE(pc.count_consultas, 0) as count_consultas,
    CURRENT_DATE - p.fecha_dx as dias_desde_diagnostico,
    COALESCE(pl.count_laboratorios, 0) as count_laboratorios,
    COALESCE(pl.avg_resultado_numerico, 0) as avg_resultado_numerico,
    COALESCE(pl.avg_biopsia, 0) as avg_biopsia,
    COALESCE(pl.avg_vph, 0) as avg_vph,
    COALESCE(pl.avg_marcador_ca125, 0) as avg_marcador_ca125,
    COALESCE(pl.avg_psa, 0) as avg_psa,
    COALESCE(pl.avg_colonoscopia, 0) as avg_colonoscopia,
    CASE WHEN p.adherencia_12m THEN 1 ELSE 0 END as adherencia_12m
FROM paciente p
LEFT JOIN paciente_consultas pc ON p.id_paciente = pc.id_paciente
LEFT JOIN paciente_labs pl ON p.id_paciente = pl.id_paciente
WHERE p.sexo IS NOT NULL
  AND p.tipo_cancer IS NOT NULL
  AND p.estadio IS NOT NULL
  AND p.aseguradora IS NOT NULL
ORDER BY p.id_paciente;
```

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

---

#### **d) Comparación de Modelos**

**Resultados del entrenamiento (80 registros: 64 train, 16 test, datos limpios):**

| Métrica | XGBoost | Neural Network |
|---------|---------|----------------|
| **Accuracy** | **81.25%** | 68.75% |
| **Precision** | **0.7857** | 0.75 |
| **Recall** | **1.0** | 0.8182 |
| **F1-Score** | **0.88** | 0.7826 |
| **AUC-ROC** | **0.6** | 0.6545 |
| Tiempo Entrenamiento | **0.12s** | 2.6s |
| Inference Time | **~5ms** | ~15ms |

**¿Cuál funciona mejor?**

**XGBoost** este modelo destaca en recall perfecto (100%) y un accuracy sobre 80%, identificando correctamente todos los pacientes con riesgo de abandono. Sin embargo, su precision moderada (78.57%) genera algunos falsos positivos. La red neuronal muestra mejor AUC (0.6545 vs 0.6), sugiriendo mayor capacidad discriminativa, pero sacrifica accuracy (68.75% vs 81.25%).

**Limitaciones detectadas:**

- **Métrica faltante crítica:** No se calculó **Balanced Accuracy**, esencial para datasets desbalanceados como este (adherencia médica).
- **Dataset pequeño:** Solo 80 registros limitan la capacidad de generalización de ambos modelos.
- **AUC moderado:** XGBoost (0.6) vs Red Neuronal (0.6545) sugieren necesidad de más features clínicas.

**Impacto de la limpieza de datos:**

- XGBoost mejoró de 68% a **81.25%** (+13%) gracias a la imputación de NaN
- Neural Network mejoró de 31% a **68.75%** (+37%) al eliminar valores faltantes

**¿Cuál es más fácil de desplegar?**

**XGBoost** es la opción clara para producción médica:

- **Simplicidad:** 1 archivo `.pkl` (~100KB) vs modelo `.keras` + scaler `.pkl`
- **Rendimiento:** 3x más rápido en inferencia crítica (5ms vs 15ms)
- **Recursos:** ~50MB RAM vs ~200MB RAM
- **Fiabilidad:** Sin dependencias complejas de GPU/TPU

**Arquitecturas utilizadas:**

**XGBoost (HistGradientBoostingClassifier):**

- **Algoritmo:** Histogram-based gradient boosting
- **Parámetros:** max_iter=100, max_depth=6, learning_rate=0.1
- **Features:** 11 (edad, consultas, laboratorios, promedios marcadores, zona_residencia_encoded, tipo_cancer_encoded)

**Red Neuronal (TensorFlow Sequential):**

- **Arquitectura:** 64 → 32 → 16 neuronas con Dropout (0.3, 0.2)
- **Optimizador:** Adam (lr=0.001)
- **Entrenamiento:** 50 epochs, batch_size=32
- **Features:** 11 (mismas que XGBoost)

**Recomendación:** **XGBoost** es la opción recomendada para producción médica inmediata. Su recall perfecto (100%) asegura que ningún paciente con riesgo de abandono pase desapercibido, compensando su menor precisión. La velocidad de entrenamiento (0.12s vs 2.6s) lo hace ideal para re-entrenamiento frecuente con nuevos datos médicos.

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

### **Parte 4: Despliegue y Arquitectura**

#### **a) Arquitectura de la API de predicción**

La arquitectura implementada sigue una estructura modular con separación clara de responsabilidades:

**i. Almacenamiento:**
**PostgreSQL 17**: Base de datos relacional con esquema definido por SQLModel
**Volúmenes persistentes**: Datos de PostgreSQL y modelos ML guardados en contenedores
**Estructura de tablas**: `paciente`, `consulta`, `laboratorio` con relaciones normalizadas

**ii. Pipeline ETL:**
**Carga**: Endpoint `POST /laboratorio/datos` para subir archivos Excel
**Transformación**: Endpoint `PUT /laboratorio/procesamiento/limpieza` ejecuta limpieza automática
**Validación**: Tipos de datos, normalización categórica, imputación de valores faltantes
**Salida**: Dataset limpio generado automáticamente en `./data/`

**iii. Entrenamiento:**
**Endpoint**: `POST /laboratorio/modelado/entrenar` con parámetro `tipo_modelo`
**Modelos**: XGBoost (recomendado) y Red Neuronal TensorFlow
**Guardado**: Modelos serializados en `./models/` con timestamp y metadata
**Métricas**: Accuracy, Precision, Recall, F1-Score, AUC calculadas automáticamente

**iv. Endpoint para inferencia:**
**Endpoint**: `POST /laboratorio/predecir` con features del paciente
**Procesamiento**: Codificación automática de variables categóricas
**Respuesta**: Predicción binaria (0/1), probabilidad y tiempo de inferencia
**Versionado**: Cada modelo incluye timestamp y versión en metadata

**v. Monitoreo:**
**Health checks**: Endpoint `/health` con verificación de conectividad
**Métricas de predicción**: Tiempo de respuesta, estado éxito/error
**Logs estructurados**: Archivo `logs/app.log` con niveles DEBUG/INFO
**Docker health checks**: Verificación automática de servicios

**vi. Logs:**
**Consola**: Nivel INFO para operaciones normales
**Archivo**: Nivel DEBUG con rotación automática
**Estructura**: JSON con campos timestamp, level, message, request_id
**Monitoreo de predicciones**: Cada inferencia se registra automáticamente

#### **b) Dockerfile para empaquetar el modelo**

```dockerfile
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PYTHONHASHSEED=random

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq5 postgresql-client curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/models /app/logs /app/data && \
    chown -R appuser:appuser /app

WORKDIR /app

COPY --chown=appuser:appuser pyproject.toml ./
RUN pip install --no-cache-dir uv
RUN uv pip install --system --no-cache -r pyproject.toml

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser alembic/ ./alembic/
COPY --chown=appuser:appuser alembic.ini ./
COPY --chown=appuser:appuser entrypoint.sh ./

RUN chmod +x /app/entrypoint.sh
RUN mkdir -p /app/models /app/logs /app/data && \
    chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
```

#### **c) Supervisión, actualización y CI/CD**

**Supervisión del desempeño del modelo:**
**Métricas automáticas**: Accuracy, F1-Score calculadas en cada predicción vs conjunto de test
**Drift detection**: Monitoreo de distribución de features de entrada
**Alertas**: Umbrales configurables para degradación de performance
**Logging de predicciones**: Cada inferencia se registra con resultado y confianza

**Actualización del modelo:**
**Pipeline automatizado**: Workflow GitHub Actions ejecuta re-entrenamiento semanal
**Validación A/B**: Comparación de versiones nuevas vs producción antes de deploy
**Rollback automático**: Si nueva versión tiene performance < 95% de la actual
**Notificación**: Alertas Slack/email cuando se actualiza modelo en producción

**Incorporación del CI/CD:**
**GitHub Actions**: Pipeline completo con linting (ruff), type checking (pyright), tests
**Build automático**: Docker image generada y publicada en ghcr.io
**Deploy**: Actualización automática de contenedores en staging/production
**Versionado semántico**: Tags v1.2.3 para releases estables

**Kubeflow para MLOps avanzado:**
**Plataforma ideal** para este caso médico por su integración nativa con Kubernetes
**Pipelines automatizados** para re-entrenamiento médico con datos sensibles
**Experiment tracking** y model versioning nativos para compliance regulatorio
**Multi-tenancy** perfecta para equipos clínicos y de desarrollo separados

---

### **Parte 5: Visualización y analítica**

#### **Dashboard ejecutivo de analítica oncológica**

La API incluye endpoints para generar dashboards comprehensivos que permiten visualizar los KPIs clave del programa oncológico:

##### Dashboard principal - Panorama general

![Dashboard Principal](img/Captura%20de%20pantalla%202025-11-18%20a%20las%200.21.07%20(2).png)

*Vista general con métricas principales

##### Dashboard de adherencia y seguimiento

![Dashboard Adherencia](img/Captura%20de%20pantalla%202025-11-18%20a%20las%200.21.17%20(2).png)

*Análisis detallado de adherencia al tratamiento:*

- **Tasa de adherencia** por grupo demográfico (barras agrupadas)
- **Promedio de resultados** por tipo de laboratorio (líneas)
- **Alertas de pacientes** sin seguimiento reciente (lista)
- **Distribución por aseguradora** (barras apiladas)

#### **KPIs para equipo directivo**

**Indicadores estratégicos principales:**

- **Tasa de adherencia global**: Meta >80%, indicador de efectividad del programa
- **Cobertura de diagnóstico temprano**: % pacientes detectados en estadio I-II
- **Tiempo diagnóstico-tratamiento**: <30 días promedio
- **Reducción de costos**: Comparación costos evitados vs invertidos

**Indicadores operativos clave:**

- **Utilización de servicios**: Consultas por especialidad vs capacidad instalada
- **Tasa de abandono**: Pacientes que dejan el tratamiento antes de 6 meses
- **Efectividad de laboratorios**: % resultados críticos identificados oportunamente
- **Satisfacción del paciente**: Encuestas de experiencia (meta por implementar)

#### **Tipos de visualización y justificación**

**Indicadores clave (KPIs Cards):**

- **Métricas principales**: Números grandes y visibles con indicadores de tendencia
- **Por qué**: Llaman la atención inmediata, facilitan toma de decisiones rápida

**Barras (Bar Charts):**

- **Total pacientes por tipo de cáncer**: Comparación clara entre categorías
- **Adherencia por grupo**: Facilita identificación de segmentos de alto riesgo
- **Por qué**: Fácil interpretación, comparación directa, estándar en reportes ejecutivos

**Líneas (Line Charts):**

- **Consultas por mes**: Muestra tendencias temporales y estacionalidad
- **Promedio resultados laboratorio**: Evolución de indicadores de salud
- **Por qué**: Excelente para detectar patrones, cambios y tendencias a lo largo del tiempo

**Tablas (Data Tables):**

- **Listado de pacientes críticos**: Detalles específicos con filtros
- **Resumen de KPIs**: Valores exactos con comparaciones
- **Por qué**: Precisión numérica, capacidad de drill-down, exportación de datos

**Justificación general:**

- **Simplicidad**: Mantener visualizaciones claras para ejecutivos no técnicos
- **Accionabilidad**: Cada gráfico responde preguntas específicas de negocio
- **Consistencia**: Etiquetas claras

---

### **Sistema de analítica para identificar patrones de uso de servicios entre pacientes oncológicos**

**i. ¿Qué datos usaría?**

Usaría los tres datasets proporcionados:

- **Pacientes.csv**: Para segmentar por características demográficas (edad, tipo_cancer, estadio, aseguradora)
- **Consultas.csv**: Es el dato clave de "uso". Se analizarían los motivos (Quimioterapia, Radioterapia, Cirugía, etc.)
- **Laboratorios.csv**: Como un tipo de servicio adicional (Biopsias, Marcadores tumorales, etc.)

**ii. ¿Cómo los limpiaría?**

- **Nulos**: Rellenar zona_residencia y aseguradora (ej. con "Desconocido" o la moda)
- **Formato**: Convertir todas las columnas de fechas (fecha_dx, fecha_consulta, fecha_muestra) a formato datetime
- **Coherencia**: Estandarizar valores categóricos (ej. "Pulmón" vs "pulmon", "M" vs "Masculino")

**iii. ¿Cómo estructuraría un modelo o análisis?**

- **Feature Engineering**: Agregar datos a nivel de paciente (id_paciente). Crear variables como total_consultas, n_quimioterapias, n_radioterapias, n_biopsias, tiempo_desde_diagnostico, etc.
- **Modelo (Clustering)**: Aplicar algoritmo no supervisado como K-Means sobre los datos agregados
- **Análisis**: El modelo agruparía pacientes en "clústeres". Cada clúster representaría un patrón (ej. "Patrón 1: Alto uso de Quimioterapia y Laboratorios", "Patrón 2: Enfoque Quirúrgico y Control")

**iv. ¿Qué producto final entregaría?**

Un **Dashboard Interactivo** (ej. en Power BI o Tableau) que permita a la dirección:

- Visualizar los patrones de uso encontrados (ej. gráfico de pastel con los clústeres)
- Filtrar estos patrones por tipo de cáncer, estadio o aseguradora
- Entender las características de cada patrón (ej. qué servicios consume cada clúster)
- Generar reportes automáticos de insights operativos

**v. ¿Qué riesgos técnicos anticiparía?**

- **Calidad de Datos**: El principal riesgo. Si los datos de origen (ej. motivo de consulta) se registran mal, el análisis será incorrecto ("Garbage In, Garbage Out")
- **Privacidad**: Manejo de datos sensibles de pacientes (Habeas Data), requiriendo anonimización y controles de acceso estrictos
- **Escalabilidad**: El análisis de clustering puede volverse lento y costoso si los datos crecen de miles a millones de registros

---

## Documentación Interactiva

Una vez iniciada la API, accede a:

- **Swagger UI**: <http://localhost:8000/docs>
- **Health Check**: <http://localhost:8000/health>

---

## Arquitectura del Proyecto

```bash
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
