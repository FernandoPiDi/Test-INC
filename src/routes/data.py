"""
Rutas para gestión de datos de laboratorio

Este módulo contiene los endpoints para:
- Carga de datos desde archivos Excel
- Limpieza y estandarización de datos
- Análisis y consultas de datos
- Generación de datasets para modelado
"""

from io import BytesIO
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlmodel import Session

from app_types.data import (
    DashboardTipoCancer,
    PacienteActividadReciente,
    PacienteLabAnalysis,
    ReporteProcesamiento,
    RutaDataset,
)
from services.data import (
    generar_y_guardar_dataset_modelado,
    get_dashboard_por_tipo_cancer,
    get_dias_entre_lab_y_diagnostico,
    get_pacientes_sin_actividad_reciente,
    procesar_limpieza_datos,
    process_excel_upload,
)
from utils.settings import get_session

router = APIRouter(prefix="/laboratorio", tags=["datos"])


@router.post("/datos")
async def upload_datos(
    file: UploadFile = File(
        ..., description="Archivo Excel con hojas: Pacientes, Consultas, Laboratorios"
    ),
    session: Session = Depends(get_session),
) -> Dict[str, Any]:
    """
    Cargar datos desde archivo Excel

    Sube datos desde un archivo Excel con tres hojas: Pacientes, Consultas y Laboratorios.
    Retorna estadísticas de los registros insertados o actualizados.
    """
    # Validar tipo de archivo
    if not file.filename:
        raise HTTPException(status_code=400, detail="No se proporcionó un archivo")

    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=400, detail="El archivo debe ser un Excel (.xlsx o .xls)"
        )

    try:
        # Leer contenido del archivo
        contents = await file.read()

        # Convertir bytes a BytesIO para pandas
        file_buffer = BytesIO(contents)

        # Procesar el archivo Excel
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


@router.put("/procesamiento/limpieza", response_model=ReporteProcesamiento)
async def ejecutar_limpieza_datos(
    session: Session = Depends(get_session),
) -> ReporteProcesamiento:
    """
    Ejecutar proceso de limpieza y estandarización de datos

    Aplica tres transformaciones principales:
    1. Normalización de variables categóricas (capitalización y estandarización)
    2. Imputación de valores faltantes (según porcentaje de nulls)
    3. Corrección de outliers mediante Winsorization (método IQR)

    IMPORTANTE: Este proceso modifica permanentemente los datos en la base de datos.
    Se recomienda ejecutarlo una sola vez después de cargar los datos.

    Retorna un reporte detallado con las transformaciones aplicadas.
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
    Analizar días entre laboratorio y diagnóstico

    Calcula los días entre la primera prueba de laboratorio y la fecha de diagnóstico
    para cada paciente. Retorna información de consistencia cuando detecta problemas.
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
    Identificar pacientes sin actividad reciente

    Identifica pacientes sin actividad reciente (sin consultas en los últimos 90 días
    o que nunca han tenido consultas). Retorna lista ordenada por días de inactividad.
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
    Generar dashboard de métricas por tipo de cáncer

    Genera métricas agregadas por tipo de cáncer: total de pacientes, edad promedio,
    consultas de los últimos 6 meses y total de laboratorios.
    """
    try:
        resultado = await get_dashboard_por_tipo_cancer(session)
        return resultado
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error ejecutando consulta SQL: {str(e)}"
        )


@router.get("/dataset/modelado", response_model=RutaDataset)
async def obtener_dataset_modelado(
    session: Session = Depends(get_session),
) -> RutaDataset:
    """
    Generar dataset consolidado para modelado de ML

    Genera un dataset consolidado y lo guarda en ./data como CSV.
    Consolida información de pacientes, consultas y laboratorios en una fila por paciente,
    incluyendo variables demográficas, clínicas, conteos agregados y promedios de resultados
    por tipo de prueba (biopsia, VPH, CA125, PSA, colonoscopia).

    El dataset generado está listo para ML: sin nulos, tipos de datos correctos y todas
    las variables agregadas. Se recomienda ejecutar el endpoint de limpieza antes.

    Retorna la ruta del archivo CSV generado con timestamp.
    """
    try:
        resultado = await generar_y_guardar_dataset_modelado(session)
        return resultado
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generando dataset para modelado: {str(e)}",
        )
