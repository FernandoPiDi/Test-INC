from io import BytesIO
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlmodel import Session

from services.data import (
    DashboardTipoCancer,
    DatasetModelado,
    PacienteActividadReciente,
    PacienteLabAnalysis,
    ReporteProcesamiento,
    ResultadoCompletoModelos,
    entrenar_ambos_modelos,
    generar_dataset_modelado,
    get_dashboard_por_tipo_cancer,
    get_dias_entre_lab_y_diagnostico,
    get_pacientes_sin_actividad_reciente,
    procesar_limpieza_datos,
    process_excel_upload,
)
from utils.settings import get_session

router = APIRouter(prefix="/laboratorio", tags=["laboratorio"])


@router.post("/datos")
async def upload_datos(
    file: UploadFile = File(
        ..., description="Archivo Excel con hojas: Pacientes, Consultas, Laboratorios"
    ),
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Endpoint para subir datos desde un archivo Excel

    El archivo debe contener tres hojas:
    - Pacientes: Datos de pacientes
    - Consultas: Datos de consultas médicas
    - Laboratorios: Datos de pruebas de laboratorio

    Returns:
        Estadísticas de registros insertados y actualizados
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No se proporcionó un archivo")

    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=400, detail="El archivo debe ser un Excel (.xlsx o .xls)"
        )

    try:
        # Read file content
        contents = await file.read()

        # Convert bytes to BytesIO for pandas
        file_buffer = BytesIO(contents)

        # Process the Excel file
        stats = await process_excel_upload(file_buffer, session)

        return {"mensaje": "Datos procesados exitosamente", "estadisticas": stats}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error procesando archivo: {str(e)}"
        )
    finally:
        await file.close()


@router.post("/procesamiento/limpieza", response_model=ReporteProcesamiento)
async def ejecutar_limpieza_datos(
    session: Session = Depends(get_session),
) -> ReporteProcesamiento:
    """
    Endpoint para ejecutar el proceso de limpieza y estandarización de datos.

    **IMPORTANTE**: Este endpoint modifica los datos en la base de datos.
    Se recomienda ejecutarlo después de cargar datos nuevos.

    ### Procesos ejecutados:

    #### 1. Normalización de Variables Categóricas
    Estandariza valores categóricos aplicando las siguientes transformaciones:
    - Capitalización estándar (primera letra mayúscula)
    - Eliminación de espacios en blanco extra
    - Normalización de variantes (ej: "control", "Control", "CONTROL" → "Control")

    **Campos normalizados:**
    - Pacientes: sexo, zona_residencia, tipo_cancer, estadio, aseguradora
    - Consultas: motivo, prioridad, especialista
    - Laboratorios: tipo_prueba, resultado, unidad

    #### 2. Análisis de Valores Faltantes
    Identifica y analiza campos con valores NULL:
    - Calcula porcentaje de valores faltantes
    - Proporciona recomendaciones basadas en el porcentaje:
      * < 5%: Mantener como NULL
      * 5-20%: Considerar imputación con moda/mediana
      * > 20%: Requiere análisis especial

    **Campos analizados:**
    - paciente.zona_residencia
    - laboratorio.resultado, resultado_numerico, unidad

    #### 3. Detección de Outliers (Método IQR)
    Detecta valores atípicos en variables numéricas usando el método del rango intercuartílico:
    - Calcula Q1 (percentil 25), Q3 (percentil 75) e IQR
    - Define límites: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
    - Identifica valores fuera de estos límites
    - Proporciona estadísticas y listado de outliers (máximo 20 por campo)

    **Campos analizados:**
    - paciente.edad
    - laboratorio.resultado_numerico

    ### Returns:
    Reporte completo con:
    - **normalizaciones**: Lista de campos normalizados con ejemplos de cambios
    - **valores_faltantes**: Análisis de campos con valores NULL
    - **outliers**: Detección de valores atípicos con estadísticas IQR
    - **resumen**: Resumen ejecutivo del procesamiento

    ### Ejemplo de uso:
    1. Cargar datos usando POST /laboratorio/datos
    2. Ejecutar limpieza usando POST /laboratorio/procesamiento/limpieza
    3. Revisar el reporte para validar cambios

    ### Nota:
    - La normalización modifica los datos en la BD (UPDATE)
    - El análisis de valores faltantes y outliers es solo informativo
    - Se recomienda revisar el reporte antes de continuar con análisis posteriores
    """
    try:
        reporte = await procesar_limpieza_datos(session)
        return reporte
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error ejecutando proceso de limpieza: {str(e)}",
        )


@router.get("/analisis/dias-lab-diagnostico", response_model=List[PacienteLabAnalysis])
async def get_analisis_dias_lab_diagnostico(
    session: Session = Depends(get_session),
) -> List[PacienteLabAnalysis]:
    """
    Endpoint que ejecuta una consulta SQL para calcular los días entre
    la primera prueba de laboratorio y la fecha de diagnóstico de cada paciente.

    Returns:
        Lista de pacientes con sus datos y el cálculo de días entre la primera
        prueba de laboratorio y el diagnóstico. Si un paciente no tiene pruebas
        de laboratorio, los campos primera_fecha_lab y dias_entre_lab_y_dx serán None.

        El campo mensaje_consistencia solo aparecerá cuando haya problemas:
        - "DATOS INCONSISTENTES: ...": cuando la fecha de diagnóstico es anterior a la primera prueba
        - "Sin datos de laboratorio": cuando el paciente no tiene pruebas de laboratorio
        - null: cuando los datos son consistentes (fecha de diagnóstico posterior a la primera prueba)
    """
    try:
        resultado = await get_dias_entre_lab_y_diagnostico(session)
        return resultado
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error ejecutando consulta SQL: {str(e)}"
        )


@router.get(
    "/analisis/actividad-reciente",
    response_model=List[PacienteActividadReciente],
)
async def get_pacientes_inactivos(
    session: Session = Depends(get_session),
) -> List[PacienteActividadReciente]:
    """
    Endpoint que ejecuta una consulta SQL para identificar pacientes sin actividad reciente.

    Identifica pacientes que:
    - No han tenido consultas en los últimos 90 días, O
    - Nunca han tenido consultas registradas

    La consulta utiliza SQL puro con CURRENT_DATE para calcular los días desde
    la última consulta.

    Returns:
        Lista de pacientes sin actividad reciente, ordenados por días sin consulta
        (descendente). Los pacientes sin consultas aparecen primero.

        Cada registro incluye:
        - Datos completos del paciente
        - ultima_consulta: fecha de la última consulta (null si nunca tuvo)
        - dias_sin_consulta: días desde la última consulta (null si nunca tuvo)
        - total_consultas: número total de consultas del paciente
        - estado_actividad: descripción del estado ("Sin consultas registradas" o "Inactivo por X días")
    """
    try:
        resultado = await get_pacientes_sin_actividad_reciente(session)
        return resultado
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error ejecutando consulta SQL: {str(e)}"
        )


@router.get(
    "/dashboard/tipo-cancer",
    response_model=List[DashboardTipoCancer],
)
async def get_dashboard_cancer(
    session: Session = Depends(get_session),
) -> List[DashboardTipoCancer]:
    """
    Endpoint que genera un dashboard básico con métricas agregadas por tipo de cáncer.

    Utiliza una consulta SQL pura para generar estadísticas agregadas por cada tipo de cáncer:

    Métricas incluidas:
    - **total_pacientes**: Número total de pacientes con ese tipo de cáncer
    - **promedio_edad**: Edad promedio de los pacientes (redondeado a 2 decimales)
    - **consultas_ultimos_6_meses**: Número de consultas realizadas en los últimos 6 meses
    - **total_laboratorios**: Número total de pruebas de laboratorio registradas

    La consulta utiliza:
    - LEFT JOIN con tablas consulta y laboratorio para incluir todos los pacientes
    - INTERVAL '6 months' para filtrar consultas recientes
    - COUNT(DISTINCT) para evitar duplicados
    - GROUP BY tipo_cancer para agrupar métricas

    Returns:
        Lista de métricas por tipo de cáncer, ordenada por número de pacientes (descendente)
    """
    try:
        resultado = await get_dashboard_por_tipo_cancer(session)
        return resultado
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error ejecutando consulta SQL: {str(e)}"
        )


@router.get("/dataset/modelado", response_model=List[DatasetModelado])
async def obtener_dataset_modelado(
    session: Session = Depends(get_session),
) -> List[DatasetModelado]:
    """
    Endpoint que genera un dataset listo para modelado de Machine Learning.

    ### Características del Dataset:

    #### Estructura:
    - **Una fila por paciente** - Datos agregados a nivel de paciente
    - **Sin valores nulos** - Todos los NULL son manejados apropiadamente
    - **Tipos de datos correctos** - Listos para usar en ML

    #### Variables Incluidas:

    **Variables Demográficas:**
    - sexo: Sexo del paciente (string)
    - edad: Edad en años (integer)
    - zona_residencia: Zona de residencia (string, "Desconocida" si es NULL)

    **Variables Clínicas:**
    - tipo_cancer: Tipo de cáncer diagnosticado (string)
    - estadio: Estadio del cáncer (string)
    - aseguradora: Compañía aseguradora (string)

    **Features Agregadas - Consultas:**
    - count_consultas: Número total de consultas (integer, 0 si no tiene)
    - dias_desde_diagnostico: Días desde la fecha de diagnóstico hasta hoy (integer)

    **Features Agregadas - Laboratorios:**
    - count_laboratorios: Número total de pruebas de laboratorio (integer, 0 si no tiene)
    - avg_resultado_numerico: Promedio general de resultados numéricos (float, 0 si no tiene)

    **Promedios por Tipo de Prueba:**
    - avg_biopsia: Promedio de resultados de biopsias (float, 0 si no tiene)
    - avg_vpH: Promedio de resultados de pruebas VPH (float, 0 si no tiene)
    - avg_marcador_ca125: Promedio de marcador CA125 (float, 0 si no tiene)
    - avg_psa: Promedio de PSA (float, 0 si no tiene)
    - avg_colonoscopia: Promedio de colonoscopias (float, 0 si no tiene)

    **Variable Objetivo:**
    - adherencia_12m: Adherencia a 12 meses (integer: 0 o 1)

    ### Manejo de Valores Nulos:
    - Zona de residencia NULL → "Desconocida"
    - Conteos sin datos → 0
    - Promedios sin datos → 0.0
    - Pacientes con variables categóricas NULL son excluidos del dataset

    ### Características SQL:
    - Utiliza CTEs (Common Table Expressions) para agregaciones eficientes
    - LEFT JOIN para incluir pacientes sin consultas/laboratorios
    - COALESCE para manejo de NULL
    - CASE WHEN para promedios condicionales por tipo de prueba

    ### Uso Típico:
    1. Cargar datos: POST /laboratorio/datos
    2. Limpiar datos: POST /laboratorio/procesamiento/limpieza
    3. Obtener dataset: GET /laboratorio/dataset/modelado
    4. Usar dataset para entrenar modelos ML

    ### Returns:
    Lista de registros tipo DatasetModelado - un registro por paciente con todas
    las variables agregadas, sin nulos y con tipos correctos para modelado.

    ### Ejemplo de Registro:
    ```json
    {
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
      "adherencia_12m": 1
    }
    ```
    """
    try:
        dataset = await generar_dataset_modelado(session)
        return dataset
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando dataset para modelado: {str(e)}",
        )


@router.post("/modelado/entrenar", response_model=ResultadoCompletoModelos)
async def entrenar_modelos_comparar(
    session: Session = Depends(get_session),
) -> ResultadoCompletoModelos:
    """
    Endpoint para entrenar AMBOS modelos (Red Neuronal y Random Forest) y compararlos.

    ## 🎯 Objetivo

    Entrenar dos modelos predictivos para **adherencia_12m** (0 = No adherente, 1 = Adherente):
    - **a) Red Neuronal** (TensorFlow)
    - **b) Random Forest** (scikit-learn)

    Y responder la pregunta **d)**: ¿Cuál funciona mejor y por qué? ¿Cuál es más fácil de desplegar?

    ## 📊 Features Utilizadas (Punto b)

    ### Variables Numéricas:
    - edad, count_consultas, count_laboratorios
    - avg_resultado_numerico
    - avg_biopsia, avg_vpH, avg_marcador_ca125, avg_psa, avg_colonoscopia

    ### Variables Categóricas (Label Encoded):
    - zona_residencia, tipo_cancer

    ## 🔧 Arquitecturas

    ### Red Neuronal (TensorFlow):
    ```
    Input(11) → Dense(64,ReLU) → Dropout(0.3) → Dense(32,ReLU) →
    Dropout(0.2) → Dense(16,ReLU) → Output(1,Sigmoid)
    ```
    - Optimizer: Adam (lr=0.001)
    - Loss: Binary Crossentropy
    - Epochs: 50, Batch: 32
    - **Requiere**: Normalización StandardScaler

    ### Random Forest (scikit-learn):
    ```
    100 árboles, max_depth=10, min_samples_split=5
    ```
    - Criterion: Gini
    - **No requiere**: Normalización
    - **Bonus**: Feature importance automático

    ## 📈 Métricas Evaluadas (Punto c)

    Para ambos modelos (train y test):
    - ✅ **Accuracy**: Proporción de predicciones correctas
    - ✅ **Precision**: TP / (TP + FP)
    - ✅ **Recall**: TP / (TP + FN)
    - ✅ **F1-Score**: Media armónica de Precision y Recall
    - ✅ **AUC**: Área bajo la curva ROC

    ## 🏆 Comparación de Modelos (Punto d)

    El endpoint responde automáticamente:

    1. **¿Cuál funciona mejor?**
       - Compara F1-Score, Accuracy y AUC
       - Identifica el modelo con mejor desempeño
       - Explica las razones técnicas

    2. **¿Cuál es más fácil de desplegar?**
       - Analiza complejidad de infraestructura
       - Evalúa tiempo de inferencia
       - Considera interpretabilidad
       - Compara recursos necesarios

    3. **Diferencias clave**
       - Tiempo de entrenamiento
       - Métricas comparadas lado a lado
       - Requisitos técnicos
       - Interpretabilidad

    4. **Recomendación final**
       - Basada en desempeño Y facilidad de despliegue
       - Considera el contexto de producción

    ## 🔄 Proceso de Entrenamiento

    1. Obtención del dataset agregado
    2. Codificación de variables categóricas
    3. División train/test (80/20, estratificado)
    4. **Red Neuronal**: Normalización + Entrenamiento
    5. **Random Forest**: Entrenamiento directo (sin normalización)
    6. Evaluación con 5 métricas estándar
    7. **Comparación automática** de resultados

    ## 📤 Respuesta

    Retorna un objeto ResultadoCompletoModelos con:

    ```json
    {
      "red_neuronal": {
        "modelo": "Red Neuronal (TensorFlow)",
        "metricas_train": {...},
        "metricas_test": {...},
        "tiempo_entrenamiento_segundos": 12.5,
        "arquitectura": {...}
      },
      "random_forest": {
        "modelo": "Random Forest (scikit-learn)",
        "metricas_train": {...},
        "metricas_test": {...},
        "tiempo_entrenamiento_segundos": 0.8,
        "arquitectura": {
          "feature_importance": {...}
        }
      },
      "comparacion": {
        "mejor_modelo": "Random Forest",
        "razon_mejor_desempeno": "...",
        "modelo_mas_facil_desplegar": "Random Forest",
        "razon_facilidad_despliegue": "...",
        "diferencias_clave": [...],
        "recomendacion_final": "..."
      }
    }
    ```

    ## 💡 Ejemplo de Uso

    ```bash
    # 1. Cargar datos
    POST /laboratorio/datos

    # 2. Limpiar datos
    POST /laboratorio/procesamiento/limpieza

    # 3. Entrenar AMBOS modelos y comparar
    POST /laboratorio/modelado/entrenar
    ```

    ## 📝 Notas

    - Entrena AMBOS modelos en una sola llamada
    - Comparación automática incluida
    - Tiempo total: ~10-20 segundos
    - NO guarda modelos en disco (demostración)
    - Mínimo 10 registros requeridos
    - Responde completamente los puntos a, b, c y d del requerimiento
    """
    try:
        resultado = await entrenar_ambos_modelos(session)
        return resultado
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error entrenando modelos: {str(e)}",
        )
