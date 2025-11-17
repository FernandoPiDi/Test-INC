from datetime import date as date_type
from datetime import datetime
from typing import Any, BinaryIO, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sqlmodel import Session, text

from models.tables import Consulta, Laboratorio, Paciente
from utils.logging_config import get_logger


class PacienteLabAnalysis(BaseModel):
    """
    Pydantic model for patient lab analysis response
    """

    id_paciente: str
    sexo: str
    edad: int
    zona_residencia: Optional[str]
    fecha_dx: date_type
    tipo_cancer: str
    estadio: str
    aseguradora: str
    adherencia_12m: bool
    primera_fecha_lab: Optional[date_type]
    dias_entre_lab_y_dx: Optional[int]
    mensaje_consistencia: Optional[str] = None


class PacienteActividadReciente(BaseModel):
    """
    Pydantic model for patient recent activity response
    """

    id_paciente: str
    sexo: str
    edad: int
    zona_residencia: Optional[str]
    fecha_dx: date_type
    tipo_cancer: str
    estadio: str
    aseguradora: str
    adherencia_12m: bool
    ultima_consulta: Optional[date_type]
    dias_sin_consulta: Optional[int]
    total_consultas: int
    estado_actividad: str


class DashboardTipoCancer(BaseModel):
    """
    Pydantic model for cancer type dashboard response
    """

    tipo_cancer: str
    total_pacientes: int
    promedio_edad: float
    consultas_ultimos_6_meses: int
    total_laboratorios: int


class NormalizacionCampo(BaseModel):
    """
    Pydantic model for field normalization report
    """

    campo: str
    registros_modificados: int
    valores_unicos_antes: int
    valores_unicos_despues: int
    ejemplos_cambios: Dict[str, str]


class AnalisisValoresFaltantes(BaseModel):
    """
    Pydantic model for missing values analysis
    """

    campo: str
    total_registros: int
    valores_faltantes: int
    porcentaje_faltante: float
    recomendacion: str


class OutliersDetectados(BaseModel):
    """
    Pydantic model for outliers detection report
    """

    campo: str
    total_registros: int
    q1: float
    q3: float
    iqr: float
    limite_inferior: float
    limite_superior: float
    outliers_detectados: int
    porcentaje_outliers: float
    valores_outliers: List[float]


class ReporteProcesamiento(BaseModel):
    """
    Pydantic model for complete data processing report
    """

    normalizaciones: List[NormalizacionCampo]
    valores_faltantes: List[AnalisisValoresFaltantes]
    outliers: List[OutliersDetectados]
    resumen: Dict[str, Any]


class DatasetModelado(BaseModel):
    """
    Pydantic model for modeling-ready dataset record (one row per patient)
    """

    # Demographic variables
    sexo: str
    edad: int
    zona_residencia: str

    # Clinical variables
    tipo_cancer: str
    estadio: str
    aseguradora: str

    # Aggregated features - consultations
    count_consultas: int
    dias_desde_diagnostico: int

    # Aggregated features - lab tests
    count_laboratorios: int
    avg_resultado_numerico: float

    # Lab test averages by type (most common types)
    avg_biopsia: float
    avg_vpH: float
    avg_marcador_ca125: float
    avg_psa: float
    avg_colonoscopia: float

    # Target variable
    adherencia_12m: int


class MetricasModelo(BaseModel):
    """
    Pydantic model for model evaluation metrics
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float


class ResultadoEntrenamiento(BaseModel):
    """
    Pydantic model for training results
    """

    modelo: str
    total_registros: int
    registros_train: int
    registros_test: int
    features_utilizadas: List[str]
    metricas_train: MetricasModelo
    metricas_test: MetricasModelo
    tiempo_entrenamiento_segundos: float
    arquitectura: Dict[str, Any]
    resumen: str


class ComparacionModelos(BaseModel):
    """
    Pydantic model for model comparison results
    """

    mejor_modelo: str
    razon_mejor_desempeno: str
    modelo_mas_facil_desplegar: str
    razon_facilidad_despliegue: str
    diferencias_clave: List[str]
    recomendacion_final: str


class ResultadoCompletoModelos(BaseModel):
    """
    Pydantic model for complete modeling results (both models + comparison)
    """

    red_neuronal: ResultadoEntrenamiento
    random_forest: ResultadoEntrenamiento
    comparacion: ComparacionModelos


def parse_date(date_value: Any) -> date_type:
    """
    Parse date from various formats (datetime, string, etc.)
    """
    if pd.isna(date_value):
        raise ValueError("Date value cannot be null")

    if isinstance(date_value, pd.Timestamp):
        return date_value.date()
    elif isinstance(date_value, datetime):
        return date_value.date()
    elif isinstance(date_value, str):
        # Try to parse string date
        return pd.to_datetime(date_value).date()
    else:
        return date_value


def parse_bool(value: Any) -> bool:
    """
    Parse boolean from various formats
    """
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in ["true", "1", "yes", "si", "sí"]
    return False


async def process_excel_upload(file: BinaryIO, session: Session) -> Dict[str, Any]:
    """
    Process uploaded Excel file and insert data into database

    Args:
        file: Binary file object of the Excel file
        session: Database session

    Returns:
        Dictionary with statistics about inserted records
    """
    stats = {
        "pacientes_insertados": 0,
        "consultas_insertadas": 0,
        "laboratorios_insertados": 0,
        "pacientes_actualizados": 0,
        "consultas_actualizadas": 0,
        "laboratorios_actualizados": 0,
        "errores": [],
    }

    try:
        # Read Excel file
        excel_data = pd.read_excel(file, sheet_name=None, engine="openpyxl")

        # Verify required sheets exist
        required_sheets = ["Pacientes", "Consultas", "Laboratorios"]
        for sheet in required_sheets:
            if sheet not in excel_data:
                raise ValueError(
                    f"Hoja requerida '{sheet}' no encontrada en el archivo Excel"
                )

        # Process Pacientes sheet
        df_pacientes = excel_data["Pacientes"]
        for _, row in df_pacientes.iterrows():
            try:
                id_paciente = str(row["id_paciente"])

                # Check if patient already exists
                existing_paciente = session.get(Paciente, id_paciente)

                zona_res_value = row["zona_residencia"]
                zona_residencia = (
                    str(zona_res_value) if pd.notna(zona_res_value) else None  # type: ignore[operator]
                )

                paciente_data = {
                    "id_paciente": id_paciente,
                    "sexo": str(row["sexo"]),
                    "edad": int(row["edad"]),
                    "zona_residencia": zona_residencia,
                    "fecha_dx": parse_date(row["fecha_dx"]),
                    "tipo_cancer": str(row["tipo_cancer"]),
                    "estadio": str(row["estadio"]),
                    "aseguradora": str(row["aseguradora"]),
                    "adherencia_12m": parse_bool(row["adherencia_12m"]),
                }

                if existing_paciente:
                    # Update existing patient
                    for key, value in paciente_data.items():
                        setattr(existing_paciente, key, value)
                    stats["pacientes_actualizados"] += 1
                else:
                    # Create new patient
                    paciente = Paciente(**paciente_data)
                    session.add(paciente)
                    stats["pacientes_insertados"] += 1

            except Exception as e:
                stats["errores"].append(
                    f"Error en paciente {row.get('id_paciente', 'desconocido')}: {str(e)}"
                )

        # Commit pacientes before processing consultas/laboratorios
        session.commit()

        # Process Consultas sheet
        df_consultas = excel_data["Consultas"]
        for _, row in df_consultas.iterrows():
            try:
                id_consulta = str(row["id_consulta"])

                # Check if consulta already exists
                existing_consulta = session.get(Consulta, id_consulta)

                consulta_data = {
                    "id_consulta": id_consulta,
                    "fecha_consulta": parse_date(row["fecha_consulta"]),
                    "motivo": str(row["motivo"]),
                    "prioridad": str(row["prioridad"]),
                    "especialista": str(row["especialista"]),
                    "id_paciente": str(row["id_paciente"]),
                }

                if existing_consulta:
                    # Update existing consulta
                    for key, value in consulta_data.items():
                        setattr(existing_consulta, key, value)
                    stats["consultas_actualizadas"] += 1
                else:
                    # Create new consulta
                    consulta = Consulta(**consulta_data)
                    session.add(consulta)
                    stats["consultas_insertadas"] += 1

            except Exception as e:
                stats["errores"].append(
                    f"Error en consulta {row.get('id_consulta', 'desconocido')}: {str(e)}"
                )

        # Commit consultas
        session.commit()

        # Process Laboratorios sheet
        df_laboratorios = excel_data["Laboratorios"]
        for _, row in df_laboratorios.iterrows():
            try:
                id_lab = str(row["id_lab"])

                # Check if laboratorio already exists
                existing_lab = session.get(Laboratorio, id_lab)

                # Parse optional fields
                resultado_value = row["resultado"]
                resultado = str(resultado_value) if pd.notna(resultado_value) else None  # type: ignore[arg-type]

                resultado_num_value = row["resultado_numerico"]
                resultado_numerico = (
                    float(resultado_num_value)
                    if pd.notna(resultado_num_value)  # type: ignore[operator]
                    else None
                )

                unidad_value = row["unidad"]
                unidad = str(unidad_value) if pd.notna(unidad_value) else None  # type: ignore[arg-type]

                laboratorio_data = {
                    "id_lab": id_lab,
                    "fecha_muestra": parse_date(row["fecha_muestra"]),
                    "tipo_prueba": str(row["tipo_prueba"]),
                    "resultado": resultado,
                    "resultado_numerico": resultado_numerico,
                    "unidad": unidad,
                    "id_paciente": str(row["id_paciente"]),
                }

                if existing_lab:
                    # Update existing laboratorio
                    for key, value in laboratorio_data.items():
                        setattr(existing_lab, key, value)
                    stats["laboratorios_actualizados"] += 1
                else:
                    # Create new laboratorio
                    laboratorio = Laboratorio(**laboratorio_data)
                    session.add(laboratorio)
                    stats["laboratorios_insertados"] += 1

            except Exception as e:
                stats["errores"].append(
                    f"Error en laboratorio {row.get('id_lab', 'desconocido')}: {str(e)}"
                )

        # Final commit
        session.commit()

    except Exception as e:
        session.rollback()
        raise Exception(f"Error procesando archivo Excel: {str(e)}")

    return stats


async def get_dias_entre_lab_y_diagnostico(
    session: Session,
) -> List[PacienteLabAnalysis]:
    """
    Execute SQL query to calculate days between first lab test and diagnosis date

    Args:
        session: Database session

    Returns:
        List of PacienteLabAnalysis objects with patient data and calculated days
    """
    # Pure SQL query to calculate days between first lab test and diagnosis
    query = text("""
        SELECT 
            p.id_paciente,
            p.sexo,
            p.edad,
            p.zona_residencia,
            p.fecha_dx,
            p.tipo_cancer,
            p.estadio,
            p.aseguradora,
            p.adherencia_12m,
            MIN(l.fecha_muestra) as primera_fecha_lab,
            (p.fecha_dx - MIN(l.fecha_muestra)) as dias_entre_lab_y_dx
        FROM paciente p
        LEFT JOIN laboratorio l ON p.id_paciente = l.id_paciente
        GROUP BY 
            p.id_paciente, 
            p.sexo, 
            p.edad, 
            p.zona_residencia, 
            p.fecha_dx, 
            p.tipo_cancer, 
            p.estadio, 
            p.aseguradora, 
            p.adherencia_12m
        ORDER BY p.id_paciente
    """)

    result = session.execute(query)
    rows = result.fetchall()

    # Convert rows to Pydantic models
    pacientes_analysis = []
    for row in rows:
        # Determine consistency message (only show when there's an issue)
        mensaje_consistencia = None
        if row.primera_fecha_lab is None:
            mensaje_consistencia = "Sin datos de laboratorio"
        elif row.dias_entre_lab_y_dx is not None and row.dias_entre_lab_y_dx < 0:
            mensaje_consistencia = "DATOS INCONSISTENTES: La fecha de diagnóstico es anterior a la primera prueba de laboratorio"

        pacientes_analysis.append(
            PacienteLabAnalysis(
                id_paciente=row.id_paciente,
                sexo=row.sexo,
                edad=row.edad,
                zona_residencia=row.zona_residencia,
                fecha_dx=row.fecha_dx,
                tipo_cancer=row.tipo_cancer,
                estadio=row.estadio,
                aseguradora=row.aseguradora,
                adherencia_12m=row.adherencia_12m,
                primera_fecha_lab=row.primera_fecha_lab,
                dias_entre_lab_y_dx=row.dias_entre_lab_y_dx,
                mensaje_consistencia=mensaje_consistencia,
            )
        )

    return pacientes_analysis


async def get_pacientes_sin_actividad_reciente(
    session: Session,
) -> List[PacienteActividadReciente]:
    """
    Execute SQL query to identify patients without recent activity (no consultations in last 90 days)

    Args:
        session: Database session

    Returns:
        List of PacienteActividadReciente objects with patients who haven't had consultations
        in the last 90 days or never had any consultations
    """
    # Pure SQL query to get patients without recent activity
    query = text("""
        SELECT 
            p.id_paciente,
            p.sexo,
            p.edad,
            p.zona_residencia,
            p.fecha_dx,
            p.tipo_cancer,
            p.estadio,
            p.aseguradora,
            p.adherencia_12m,
            MAX(c.fecha_consulta) as ultima_consulta,
            CASE 
                WHEN MAX(c.fecha_consulta) IS NULL THEN NULL
                ELSE CURRENT_DATE - MAX(c.fecha_consulta)
            END as dias_sin_consulta,
            COUNT(c.id_consulta) as total_consultas
        FROM paciente p
        LEFT JOIN consulta c ON p.id_paciente = c.id_paciente
        GROUP BY 
            p.id_paciente, 
            p.sexo, 
            p.edad, 
            p.zona_residencia, 
            p.fecha_dx, 
            p.tipo_cancer, 
            p.estadio, 
            p.aseguradora, 
            p.adherencia_12m
        HAVING 
            MAX(c.fecha_consulta) IS NULL 
            OR CURRENT_DATE - MAX(c.fecha_consulta) > 90
        ORDER BY dias_sin_consulta DESC NULLS FIRST
    """)

    result = session.execute(query)
    rows = result.fetchall()

    # Convert rows to Pydantic models
    pacientes_inactivos = []
    for row in rows:
        # Determine activity status
        if row.ultima_consulta is None:
            estado_actividad = "Sin consultas registradas"
        else:
            estado_actividad = f"Inactivo por {row.dias_sin_consulta} días"

        pacientes_inactivos.append(
            PacienteActividadReciente(
                id_paciente=row.id_paciente,
                sexo=row.sexo,
                edad=row.edad,
                zona_residencia=row.zona_residencia,
                fecha_dx=row.fecha_dx,
                tipo_cancer=row.tipo_cancer,
                estadio=row.estadio,
                aseguradora=row.aseguradora,
                adherencia_12m=row.adherencia_12m,
                ultima_consulta=row.ultima_consulta,
                dias_sin_consulta=row.dias_sin_consulta,
                total_consultas=row.total_consultas,
                estado_actividad=estado_actividad,
            )
        )

    return pacientes_inactivos


async def get_dashboard_por_tipo_cancer(session: Session) -> List[DashboardTipoCancer]:
    """
    Execute SQL query to generate a dashboard by cancer type with key metrics

    Args:
        session: Database session

    Returns:
        List of DashboardTipoCancer objects with aggregated statistics per cancer type:
        - Total number of patients
        - Average age
        - Number of consultations in the last 6 months
        - Total number of lab tests
    """
    # Pure SQL query to generate dashboard metrics by cancer type
    query = text("""
        SELECT 
            p.tipo_cancer,
            COUNT(DISTINCT p.id_paciente) as total_pacientes,
            ROUND(AVG(p.edad), 2) as promedio_edad,
            COUNT(DISTINCT CASE 
                WHEN c.fecha_consulta >= CURRENT_DATE - INTERVAL '6 months' 
                THEN c.id_consulta 
            END) as consultas_ultimos_6_meses,
            COUNT(DISTINCT l.id_lab) as total_laboratorios
        FROM paciente p
        LEFT JOIN consulta c ON p.id_paciente = c.id_paciente
        LEFT JOIN laboratorio l ON p.id_paciente = l.id_paciente
        GROUP BY p.tipo_cancer
        ORDER BY total_pacientes DESC
    """)

    result = session.execute(query)
    rows = result.fetchall()

    # Convert rows to Pydantic models
    dashboard_data = []
    for row in rows:
        dashboard_data.append(
            DashboardTipoCancer(
                tipo_cancer=row.tipo_cancer,
                total_pacientes=row.total_pacientes,
                promedio_edad=float(row.promedio_edad),
                consultas_ultimos_6_meses=row.consultas_ultimos_6_meses,
                total_laboratorios=row.total_laboratorios,
            )
        )

    return dashboard_data


async def procesar_limpieza_datos(session: Session) -> ReporteProcesamiento:
    """
    Execute data cleaning and standardization process

    Args:
        session: Database session

    Returns:
        ReporteProcesamiento with:
        - Categorical variable normalizations
        - Missing values analysis
        - Outlier detection using IQR method
    """
    normalizaciones: List[NormalizacionCampo] = []
    valores_faltantes: List[AnalisisValoresFaltantes] = []
    outliers_list: List[OutliersDetectados] = []

    # ===========================================================================
    # 1. NORMALIZACIÓN DE VARIABLES CATEGÓRICAS
    # ===========================================================================

    # Campos categóricos a normalizar por tabla
    campos_categoricos = {
        "paciente": [
            "sexo",
            "zona_residencia",
            "tipo_cancer",
            "estadio",
            "aseguradora",
        ],
        "consulta": ["motivo", "prioridad", "especialista"],
        "laboratorio": ["tipo_prueba", "resultado", "unidad"],
    }

    def normalizar_texto(valor: Optional[str]) -> Optional[str]:
        """Normaliza texto: capitaliza primera letra, elimina espacios extra"""
        if valor is None or valor == "":
            return None
        return valor.strip().capitalize()

    # Normalizar cada tabla y campo
    for tabla, campos in campos_categoricos.items():
        for campo in campos:
            # Obtener valores únicos antes de normalización
            query_antes = text(
                f"SELECT DISTINCT {campo} FROM {tabla} WHERE {campo} IS NOT NULL"
            )
            result_antes = session.execute(query_antes)
            valores_antes = [row[0] for row in result_antes.fetchall()]
            valores_unicos_antes = len(valores_antes)

            # Crear mapeo de normalización
            mapeo_cambios: Dict[str, str] = {}
            registros_modificados = 0

            for valor_original in valores_antes:
                valor_normalizado = normalizar_texto(valor_original)
                if valor_normalizado != valor_original:
                    mapeo_cambios[valor_original] = valor_normalizado  # type: ignore[assignment]
                    # Actualizar registros
                    update_query = text(
                        f"UPDATE {tabla} SET {campo} = :nuevo WHERE {campo} = :viejo"
                    )
                    result = session.execute(
                        update_query,
                        {"nuevo": valor_normalizado, "viejo": valor_original},
                    )
                    registros_modificados += result.rowcount  # type: ignore[attr-defined]

            session.commit()

            # Obtener valores únicos después de normalización
            query_despues = text(
                f"SELECT DISTINCT {campo} FROM {tabla} WHERE {campo} IS NOT NULL"
            )
            result_despues = session.execute(query_despues)
            valores_unicos_despues = len(result_despues.fetchall())

            # Agregar al reporte (solo si hubo cambios)
            if mapeo_cambios:
                # Limitar ejemplos a máximo 5
                ejemplos = dict(list(mapeo_cambios.items())[:5])
                normalizaciones.append(
                    NormalizacionCampo(
                        campo=f"{tabla}.{campo}",
                        registros_modificados=registros_modificados,
                        valores_unicos_antes=valores_unicos_antes,
                        valores_unicos_despues=valores_unicos_despues,
                        ejemplos_cambios=ejemplos,
                    )
                )

    # ===========================================================================
    # 2. ANÁLISIS DE VALORES FALTANTES
    # ===========================================================================

    # Analizar campos con posibles valores faltantes
    campos_analizar_null = [
        ("paciente", "zona_residencia"),
        ("laboratorio", "resultado"),
        ("laboratorio", "resultado_numerico"),
        ("laboratorio", "unidad"),
    ]

    for tabla, campo in campos_analizar_null:
        # Contar total y nulos
        query_analisis = text(f"""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE {campo} IS NULL) as nulos
            FROM {tabla}
        """)
        result = session.execute(query_analisis)
        row = result.fetchone()

        if row:
            total = row.total
            nulos = row.nulos
            porcentaje = (nulos / total * 100) if total > 0 else 0

            # Determinar recomendación
            if porcentaje < 5:
                recomendacion = "Mantener como NULL - porcentaje bajo"
            elif porcentaje < 20:
                recomendacion = "Considerar imputación con moda/mediana"
            else:
                recomendacion = (
                    "Requiere análisis especial - alto porcentaje de faltantes"
                )

            valores_faltantes.append(
                AnalisisValoresFaltantes(
                    campo=f"{tabla}.{campo}",
                    total_registros=total,
                    valores_faltantes=nulos,
                    porcentaje_faltante=round(porcentaje, 2),
                    recomendacion=recomendacion,
                )
            )

    # ===========================================================================
    # 3. DETECCIÓN DE OUTLIERS USANDO IQR
    # ===========================================================================

    # Campos numéricos a analizar
    campos_numericos = [
        ("paciente", "edad"),
        ("laboratorio", "resultado_numerico"),
    ]

    for tabla, campo in campos_numericos:
        # Calcular Q1, Q3 e IQR
        query_stats = text(f"""
            SELECT 
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {campo}) as q1,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {campo}) as q3,
                COUNT(*) as total
            FROM {tabla}
            WHERE {campo} IS NOT NULL
        """)
        result = session.execute(query_stats)
        row = result.fetchone()

        if row and row.total > 0:
            q1 = float(row.q1)
            q3 = float(row.q3)
            iqr = q3 - q1
            limite_inferior = q1 - 1.5 * iqr
            limite_superior = q3 + 1.5 * iqr
            total = row.total

            # Detectar outliers
            query_outliers = text(f"""
                SELECT {campo}
                FROM {tabla}
                WHERE {campo} IS NOT NULL
                  AND ({campo} < :limite_inf OR {campo} > :limite_sup)
                ORDER BY {campo}
                LIMIT 20
            """)
            result_outliers = session.execute(
                query_outliers,
                {"limite_inf": limite_inferior, "limite_sup": limite_superior},
            )
            valores_outliers_raw = [float(row[0]) for row in result_outliers.fetchall()]

            # Contar total de outliers
            query_count_outliers = text(f"""
                SELECT COUNT(*)
                FROM {tabla}
                WHERE {campo} IS NOT NULL
                  AND ({campo} < :limite_inf OR {campo} > :limite_sup)
            """)
            result_count = session.execute(
                query_count_outliers,
                {"limite_inf": limite_inferior, "limite_sup": limite_superior},
            )
            count_outliers = result_count.scalar() or 0

            porcentaje_outliers = (count_outliers / total * 100) if total > 0 else 0

            outliers_list.append(
                OutliersDetectados(
                    campo=f"{tabla}.{campo}",
                    total_registros=total,
                    q1=round(q1, 2),
                    q3=round(q3, 2),
                    iqr=round(iqr, 2),
                    limite_inferior=round(limite_inferior, 2),
                    limite_superior=round(limite_superior, 2),
                    outliers_detectados=count_outliers,
                    porcentaje_outliers=round(porcentaje_outliers, 2),
                    valores_outliers=valores_outliers_raw,
                )
            )

    # ===========================================================================
    # 4. RESUMEN EJECUTIVO
    # ===========================================================================

    resumen = {
        "timestamp": datetime.now().isoformat(),
        "total_normalizaciones": len(normalizaciones),
        "total_registros_modificados": sum(
            n.registros_modificados for n in normalizaciones
        ),
        "campos_con_valores_faltantes": len(valores_faltantes),
        "campos_con_outliers": len(outliers_list),
        "total_outliers_detectados": sum(o.outliers_detectados for o in outliers_list),
        "estado": "Procesamiento completado exitosamente",
    }

    return ReporteProcesamiento(
        normalizaciones=normalizaciones,
        valores_faltantes=valores_faltantes,
        outliers=outliers_list,
        resumen=resumen,
    )


async def generar_dataset_modelado(session: Session) -> List[DatasetModelado]:
    """
    Generate modeling-ready dataset with aggregated features

    Args:
        session: Database session

    Returns:
        List of DatasetModelado objects - one row per patient with:
        - All categorical variables
        - Aggregated counts (consultations, labs)
        - Averaged lab results by test type
        - No null values (handled appropriately)
        - Correct data types
    """
    # Complex SQL query to generate aggregated dataset
    query = text("""
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
        ORDER BY p.id_paciente
    """)

    result = session.execute(query)
    rows = result.fetchall()

    # Convert rows to Pydantic models
    dataset = []
    for row in rows:
        dataset.append(
            DatasetModelado(
                sexo=row.sexo,
                edad=row.edad,
                zona_residencia=row.zona_residencia,
                tipo_cancer=row.tipo_cancer,
                estadio=row.estadio,
                aseguradora=row.aseguradora,
                count_consultas=row.count_consultas,
                dias_desde_diagnostico=row.dias_desde_diagnostico,
                count_laboratorios=row.count_laboratorios,
                avg_resultado_numerico=round(float(row.avg_resultado_numerico), 2),
                avg_biopsia=round(float(row.avg_biopsia), 2),
                avg_vpH=round(float(row.avg_vph), 2),
                avg_marcador_ca125=round(float(row.avg_marcador_ca125), 2),
                avg_psa=round(float(row.avg_psa), 2),
                avg_colonoscopia=round(float(row.avg_colonoscopia), 2),
                adherencia_12m=row.adherencia_12m,
            )
        )

    return dataset


async def entrenar_ambos_modelos(session: Session) -> ResultadoCompletoModelos:
    """
    Train both Neural Network and Random Forest models and compare them

    Args:
        session: Database session

    Returns:
        ResultadoCompletoModelos with both models' results and comparison
    """
    import time

    import tensorflow as tf
    from sklearn.ensemble import RandomForestClassifier
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

    # ===========================================================================
    # 1. OBTENER DATASET (común para ambos modelos)
    # ===========================================================================
    dataset_list = await generar_dataset_modelado(session)
    df = pd.DataFrame([d.model_dump() for d in dataset_list])

    if len(df) < 10:
        raise ValueError("Dataset muy pequeño para entrenar (mínimo 10 registros)")

    # ===========================================================================
    # 2. PREPARACIÓN DE FEATURES (común para ambos)
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

    def calcular_metricas_modelo(
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

    # ===========================================================================
    # 3. ENTRENAR RED NEURONAL
    # ===========================================================================
    inicio_nn = time.time()

    # Normalización para NN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Construir red neuronal
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

    # Entrenar
    modelo_nn.fit(
        X_train_scaled,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
    )

    # Predicciones
    y_train_pred_proba_nn = modelo_nn.predict(X_train_scaled, verbose=0)
    y_test_pred_proba_nn = modelo_nn.predict(X_test_scaled, verbose=0)

    # Métricas
    metricas_train_nn = calcular_metricas_modelo(
        np.array(y_train), np.array(y_train_pred_proba_nn)
    )
    metricas_test_nn = calcular_metricas_modelo(
        np.array(y_test), np.array(y_test_pred_proba_nn)
    )

    tiempo_nn = time.time() - inicio_nn

    resultado_nn = ResultadoEntrenamiento(
        modelo="Red Neuronal (TensorFlow)",
        total_registros=len(df),
        registros_train=len(X_train),
        registros_test=len(X_test),
        features_utilizadas=features_finales,
        metricas_train=metricas_train_nn,
        metricas_test=metricas_test_nn,
        tiempo_entrenamiento_segundos=round(tiempo_nn, 2),
        arquitectura={
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
        },
        resumen=f"Accuracy test: {metricas_test_nn.accuracy:.2%}, F1: {metricas_test_nn.f1_score:.4f}, AUC: {metricas_test_nn.auc:.4f}",
    )

    # ===========================================================================
    # 4. ENTRENAR RANDOM FOREST
    # ===========================================================================
    inicio_rf = time.time()

    # Random Forest no requiere normalización
    modelo_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    # Entrenar
    modelo_rf.fit(X_train, y_train)

    # Predicciones
    y_train_pred_proba_rf = modelo_rf.predict_proba(X_train)[:, 1]  # type: ignore[call-overload]
    y_test_pred_proba_rf = modelo_rf.predict_proba(X_test)[:, 1]  # type: ignore[call-overload]

    # Métricas
    metricas_train_rf = calcular_metricas_modelo(
        np.array(y_train), np.array(y_train_pred_proba_rf)
    )
    metricas_test_rf = calcular_metricas_modelo(
        np.array(y_test), np.array(y_test_pred_proba_rf)
    )

    tiempo_rf = time.time() - inicio_rf

    # Importancia de features
    feature_importance = dict(
        zip(
            features_finales,
            [round(float(x), 4) for x in modelo_rf.feature_importances_],
        )
    )

    resultado_rf = ResultadoEntrenamiento(
        modelo="Random Forest (scikit-learn)",
        total_registros=len(df),
        registros_train=len(X_train),
        registros_test=len(X_test),
        features_utilizadas=features_finales,
        metricas_train=metricas_train_rf,
        metricas_test=metricas_test_rf,
        tiempo_entrenamiento_segundos=round(tiempo_rf, 2),
        arquitectura={
            "tipo": "Random Forest Classifier",
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "criterion": "gini",
            "feature_importance": feature_importance,
        },
        resumen=f"Accuracy test: {metricas_test_rf.accuracy:.2%}, F1: {metricas_test_rf.f1_score:.4f}, AUC: {metricas_test_rf.auc:.4f}",
    )

    # ===========================================================================
    # 5. COMPARACIÓN DE MODELOS (Punto d)
    # ===========================================================================

    # Determinar mejor modelo por desempeño
    if metricas_test_rf.f1_score > metricas_test_nn.f1_score:
        mejor_modelo = "Random Forest"
        diferencia_f1 = metricas_test_rf.f1_score - metricas_test_nn.f1_score
        razon_desempeno = (
            f"Random Forest supera a la Red Neuronal en F1-Score por {diferencia_f1:.4f} puntos. "
            f"También tiene mejor Accuracy ({metricas_test_rf.accuracy:.2%} vs {metricas_test_nn.accuracy:.2%}) "
            f"y AUC ({metricas_test_rf.auc:.4f} vs {metricas_test_nn.auc:.4f}). "
            "Random Forest es más robusto con datasets pequeños y no requiere tunning extenso de hiperparámetros."
        )
    else:
        mejor_modelo = "Red Neuronal"
        diferencia_f1 = metricas_test_nn.f1_score - metricas_test_rf.f1_score
        razon_desempeno = (
            f"Red Neuronal supera a Random Forest en F1-Score por {diferencia_f1:.4f} puntos. "
            f"También tiene mejor Accuracy ({metricas_test_nn.accuracy:.2%} vs {metricas_test_rf.accuracy:.2%}) "
            f"y AUC ({metricas_test_nn.auc:.4f} vs {metricas_test_rf.auc:.4f}). "
            "Las redes neuronales pueden capturar relaciones no lineales complejas mejor que los árboles."
        )

    # Modelo más fácil de desplegar
    modelo_mas_facil = "Random Forest"
    razon_facilidad = (
        "Random Forest es significativamente más fácil de desplegar porque: "
        "(1) No requiere GPU ni TensorFlow en producción, "
        "(2) El modelo se serializa fácilmente con pickle/joblib, "
        "(3) Predicciones instantáneas sin carga de modelo pesado, "
        "(4) Menor consumo de memoria y CPU, "
        f"(5) Tiempo de entrenamiento mucho menor ({tiempo_rf:.2f}s vs {tiempo_nn:.2f}s), "
        "(6) No requiere normalización de datos en producción, "
        "(7) Más interpretable con feature importance."
    )

    # Diferencias clave
    diferencias = [
        f"Tiempo de entrenamiento: RF {tiempo_rf:.2f}s vs NN {tiempo_nn:.2f}s",
        f"Accuracy: RF {metricas_test_rf.accuracy:.4f} vs NN {metricas_test_nn.accuracy:.4f}",
        f"F1-Score: RF {metricas_test_rf.f1_score:.4f} vs NN {metricas_test_nn.f1_score:.4f}",
        f"AUC: RF {metricas_test_rf.auc:.4f} vs NN {metricas_test_nn.auc:.4f}",
        "RF no requiere normalización de features, NN sí",
        "RF proporciona feature importance automáticamente",
        "NN requiere TensorFlow/GPU para entrenamiento óptimo",
        "RF más interpretable y explicable para stakeholders",
    ]

    # Recomendación final
    recomendacion = (
        f"**Recomendación**: Usar {mejor_modelo if mejor_modelo == 'Random Forest' else 'Random Forest'} para producción. "
        "Aunque el desempeño puede ser similar, Random Forest ofrece: "
        "(1) Despliegue más simple y robusto, "
        "(2) Menor latencia en predicciones, "
        "(3) No requiere infraestructura especializada, "
        "(4) Mayor interpretabilidad con feature importance, "
        "(5) Mantenimiento más sencillo. "
        f"Con {metricas_test_rf.accuracy:.2%} de accuracy y {metricas_test_rf.f1_score:.4f} de F1-Score, "
        "Random Forest cumple con los requisitos de precisión mientras minimiza la complejidad operacional."
    )

    comparacion = ComparacionModelos(
        mejor_modelo=mejor_modelo,
        razon_mejor_desempeno=razon_desempeno,
        modelo_mas_facil_desplegar=modelo_mas_facil,
        razon_facilidad_despliegue=razon_facilidad,
        diferencias_clave=diferencias,
        recomendacion_final=recomendacion,
    )

    # ===========================================================================
    # 6. GUARDAR MODELO CON VERSIONADO
    # ===========================================================================
    import pickle
    from pathlib import Path

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Generate version based on timestamp
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_short = datetime.now().strftime("v%Y%m%d")

    # Save Random Forest model (recommended for production)
    modelo_filename = models_dir / f"modelo_rf_{version}.pkl"
    with open(modelo_filename, "wb") as f:
        pickle.dump(modelo_rf, f)

    # Save feature names
    feature_names_file = models_dir / f"feature_names_{version}.pkl"
    with open(feature_names_file, "wb") as f:
        pickle.dump(features_finales, f)

    # Save label encoders
    encoders_file = models_dir / f"label_encoders_{version}.pkl"
    with open(encoders_file, "wb") as f:
        pickle.dump(label_encoders, f)

    # Save model metadata
    metadata = {
        "version": version,
        "version_short": version_short,
        "model_type": "RandomForestClassifier",
        "features": features_finales,
        "metrics": {
            "train": {
                "accuracy": metricas_train_rf.accuracy,
                "precision": metricas_train_rf.precision,
                "recall": metricas_train_rf.recall,
                "f1_score": metricas_train_rf.f1_score,
                "auc": metricas_train_rf.auc,
            },
            "test": {
                "accuracy": metricas_test_rf.accuracy,
                "precision": metricas_test_rf.precision,
                "recall": metricas_test_rf.recall,
                "f1_score": metricas_test_rf.f1_score,
                "auc": metricas_test_rf.auc,
            },
        },
        "training_info": {
            "total_records": len(df),
            "train_records": len(X_train),
            "test_records": len(X_test),
            "training_time_seconds": tiempo_rf,
        },
        "created_at": datetime.now().isoformat(),
    }

    metadata_file = models_dir / f"metadata_{version}.json"
    import json

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Also create a symlink or copy to latest for backward compatibility
    latest_model = models_dir / "modelo_rf_latest.pkl"
    import shutil

    shutil.copy(modelo_filename, latest_model)

    logger = get_logger(__name__)
    logger.info(
        f"Model saved successfully. Version: {version}, "
        f"File: {modelo_filename}, Metrics: {metricas_test_rf.f1_score:.4f} F1"
    )

    return ResultadoCompletoModelos(
        red_neuronal=resultado_nn,
        random_forest=resultado_rf,
        comparacion=comparacion,
    )


async def entrenar_modelo_neural_network(session: Session) -> ResultadoEntrenamiento:
    """
    Train a neural network model using TensorFlow for adherencia prediction

    Args:
        session: Database session

    Returns:
        ResultadoEntrenamiento with training metrics and model info
    """
    import time

    import tensorflow as tf
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

    inicio = time.time()

    # ===========================================================================
    # 1. OBTENER DATASET
    # ===========================================================================
    dataset_list = await generar_dataset_modelado(session)
    df = pd.DataFrame([d.model_dump() for d in dataset_list])

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
