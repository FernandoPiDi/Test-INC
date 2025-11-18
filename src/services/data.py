from datetime import date as date_type
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sqlmodel import Session, text

from app_types.data import (
    AnalisisValoresFaltantes,
    DashboardTipoCancer,
    DatasetModelado,
    NormalizacionCampo,
    OutliersDetectados,
    PacienteActividadReciente,
    PacienteLabAnalysis,
    ReporteProcesamiento,
    RutaDataset,
)
from models.tables import Consulta, Laboratorio, Paciente
from utils.logging_config import get_logger

logger = get_logger(__name__)


def crear_carpeta_modelos() -> Path:
    """
    Crear carpeta de modelos

    Crea la carpeta ./models en la raíz del workspace si no existe

    Returns:
        Ruta a la carpeta de modelos
    """
    # Obtener raíz del workspace (2 niveles arriba de services)
    workspace_root = Path(__file__).parent.parent.parent
    carpeta_modelos = workspace_root / "models"
    carpeta_modelos.mkdir(exist_ok=True)
    return carpeta_modelos


def generar_nombre_modelo(nombre_modelo: str, extension: str) -> str:
    """
    Generar nombre de archivo para modelo con timestamp

    Args:
        nombre_modelo: Nombre base del modelo (ej: "xgboost", "neural_network")
        extension: Extensión del archivo (ej: "pkl", "keras")

    Returns:
        Nombre del archivo con formato modelname_timestamp.extension
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{nombre_modelo}_{timestamp}.{extension}"


def parse_date(date_value: Any) -> date_type:
    """
    Parsear fecha desde varios formatos (datetime, string, etc.)
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
    Parsear booleano desde varios formatos
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
    Procesar archivo Excel cargado e insertar datos en base de datos

    Args:
        file: Objeto de archivo binario del Excel
        session: Sesión de base de datos

    Returns:
        Diccionario con estadísticas de registros insertados
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
        # Leer archivo Excel
        excel_data = pd.read_excel(file, sheet_name=None, engine="openpyxl")

        # Verificar que existan las hojas requeridas
        required_sheets = ["Pacientes", "Consultas", "Laboratorios"]
        for sheet in required_sheets:
            if sheet not in excel_data:
                raise ValueError(
                    f"Hoja requerida '{sheet}' no encontrada en el archivo Excel"
                )

        # Procesar hoja de Pacientes
        df_pacientes = excel_data["Pacientes"]
        for _, row in df_pacientes.iterrows():
            try:
                id_paciente = str(row["id_paciente"])

                # Verificar si el paciente ya existe
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
                    # Actualizar paciente existente
                    for key, value in paciente_data.items():
                        setattr(existing_paciente, key, value)
                    stats["pacientes_actualizados"] += 1
                else:
                    # Crear nuevo paciente
                    paciente = Paciente(**paciente_data)
                    session.add(paciente)
                    stats["pacientes_insertados"] += 1

            except Exception as e:
                stats["errores"].append(
                    f"Error en paciente {row.get('id_paciente', 'desconocido')}: {str(e)}"
                )

        # Commit de pacientes antes de procesar consultas/laboratorios
        session.commit()

        # Procesar hoja de Consultas
        df_consultas = excel_data["Consultas"]
        for _, row in df_consultas.iterrows():
            try:
                id_consulta = str(row["id_consulta"])

                # Verificar si la consulta ya existe
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
                    # Actualizar consulta existente
                    for key, value in consulta_data.items():
                        setattr(existing_consulta, key, value)
                    stats["consultas_actualizadas"] += 1
                else:
                    # Crear nueva consulta
                    consulta = Consulta(**consulta_data)
                    session.add(consulta)
                    stats["consultas_insertadas"] += 1

            except Exception as e:
                stats["errores"].append(
                    f"Error en consulta {row.get('id_consulta', 'desconocido')}: {str(e)}"
                )

        # Commit de consultas
        session.commit()

        # Procesar hoja de Laboratorios
        df_laboratorios = excel_data["Laboratorios"]
        for _, row in df_laboratorios.iterrows():
            try:
                id_lab = str(row["id_lab"])

                # Verificar si el laboratorio ya existe
                existing_lab = session.get(Laboratorio, id_lab)

                # Parsear campos opcionales
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
                    # Actualizar laboratorio existente
                    for key, value in laboratorio_data.items():
                        setattr(existing_lab, key, value)
                    stats["laboratorios_actualizados"] += 1
                else:
                    # Crear nuevo laboratorio
                    laboratorio = Laboratorio(**laboratorio_data)
                    session.add(laboratorio)
                    stats["laboratorios_insertados"] += 1

            except Exception as e:
                stats["errores"].append(
                    f"Error en laboratorio {row.get('id_lab', 'desconocido')}: {str(e)}"
                )

        # Commit final
        session.commit()

    except Exception as e:
        session.rollback()
        raise Exception(f"Error procesando archivo Excel: {str(e)}")

    return stats


async def get_dias_entre_lab_y_diagnostico(
    session: Session,
) -> List[PacienteLabAnalysis]:
    """
    Calcular días entre prueba de laboratorio y diagnóstico

    Ejecuta consulta SQL para calcular días entre la primera prueba de
    laboratorio y la fecha de diagnóstico.

    Args:
        session: Sesión de base de datos

    Returns:
        Lista de objetos PacienteLabAnalysis con datos del paciente y días calculados
    """
    # Consulta SQL para calcular días entre primera prueba de laboratorio y diagnóstico
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

    # Convertir filas a modelos Pydantic
    pacientes_analysis = []
    for row in rows:
        # Determinar mensaje de consistencia (solo mostrar cuando hay un problema)
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
    Identificar pacientes sin actividad reciente

    Ejecuta consulta SQL para identificar pacientes sin consultas en los últimos 90 días.

    Args:
        session: Sesión de base de datos

    Returns:
        Lista de objetos PacienteActividadReciente con pacientes que no han tenido
        consultas en los últimos 90 días o nunca han tenido consultas
    """
    # Consulta SQL para obtener pacientes sin actividad reciente
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

    # Convertir filas a modelos Pydantic
    pacientes_inactivos = []
    for row in rows:
        # Determinar estado de actividad
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
    Generar dashboard por tipo de cáncer

    Ejecuta consulta SQL para generar un dashboard por tipo de cáncer con métricas clave.

    Args:
        session: Sesión de base de datos

    Returns:
        Lista de objetos DashboardTipoCancer con estadísticas agregadas por tipo de cáncer:
        - Total de pacientes
        - Edad promedio
        - Número de consultas en los últimos 6 meses
        - Total de pruebas de laboratorio
    """
    # Consulta SQL para generar métricas de dashboard por tipo de cáncer
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

    # Convertir filas a modelos Pydantic
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


class NormalizadorCategorico:
    """
    Transformador compatible con sklearn para normalización de variables categóricas

    Normaliza texto capitalizando la primera letra y eliminando espacios extra.
    Rastrea cambios para propósitos de reporte.
    """

    def __init__(self) -> None:
        self.normalizaciones: List[NormalizacionCampo] = []

    def normalizar_texto(self, valor: Any) -> Optional[str]:
        """Normalizar texto: capitalizar primera letra, eliminar espacios extra"""
        if pd.isna(valor) or valor == "":
            return None
        return str(valor).strip().capitalize()

    def fit(self, X: pd.DataFrame, y: Any = None) -> "NormalizadorCategorico":
        """Método fit (requerido por sklearn, pero no usado en este transformador)"""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transformar columnas categóricas normalizando el texto

        Args:
            X: DataFrame con columnas categóricas

        Returns:
            DataFrame transformado
        """
        X_copia = X.copy()
        self.normalizaciones = []

        columnas_categoricas = X_copia.select_dtypes(include=["object"]).columns

        for col in columnas_categoricas:
            if col in X_copia.columns:
                # Obtener valores únicos antes de normalización
                valores_unicos_antes = int(X_copia[col].nunique())
                valores_antes = X_copia[col].dropna().unique()

                # Aplicar normalización
                X_copia[col] = X_copia[col].apply(self.normalizar_texto)

                # Obtener valores únicos después de normalización
                valores_unicos_despues = int(X_copia[col].nunique())

                # Rastrear cambios
                mapeo_cambios: Dict[str, str] = {}
                registros_modificados = 0

                for valor_original in valores_antes:
                    valor_normalizado = self.normalizar_texto(valor_original)
                    if valor_normalizado != valor_original:
                        mapeo_cambios[str(valor_original)] = str(valor_normalizado)
                        registros_modificados += int((X[col] == valor_original).sum())

                if mapeo_cambios:
                    ejemplos = dict(list(mapeo_cambios.items())[:5])
                    self.normalizaciones.append(
                        NormalizacionCampo(
                            campo=col,
                            registros_modificados=registros_modificados,
                            valores_unicos_antes=valores_unicos_antes,
                            valores_unicos_despues=valores_unicos_despues,
                            ejemplos_cambios=ejemplos,
                        )
                    )

        return X_copia


class ImputadorAdaptativoValoresFaltantes:
    """
    Transformador compatible con sklearn para imputación adaptativa de valores faltantes

    Estrategia de imputación basada en porcentaje de valores faltantes:
    - < 5%: Rellenar con valores por defecto
    - 5-20%: Imputar con moda (categóricos) o mediana (numéricos)
    - > 20%: No imputar (demasiado arriesgado)
    """

    def __init__(self) -> None:
        self.valores_faltantes: List[AnalisisValoresFaltantes] = []
        self.valores_imputacion: Dict[str, Any] = {}
        self.valores_defecto = {
            "zona_residencia": "Desconocida",
            "resultado": "Sin resultado",
            "unidad": "N/a",
            "resultado_numerico": 0.0,  # Resultados de laboratorio sin valor = 0
        }

    def fit(
        self, X: pd.DataFrame, y: Any = None
    ) -> "ImputadorAdaptativoValoresFaltantes":
        """
        Ajustar imputador calculando estadísticas para cada columna

        Args:
            X: DataFrame a ajustar
            y: Ignorado (para compatibilidad con sklearn)

        Returns:
            Instancia ajustada del imputador
        """
        self.valores_imputacion = {}

        for col in X.columns:
            porcentaje_faltante = (X[col].isna().sum() / len(X)) * 100

            if porcentaje_faltante > 0:
                # Verificar si hay valor por defecto definido
                if col in self.valores_defecto:
                    self.valores_imputacion[col] = self.valores_defecto[col]
                elif porcentaje_faltante < 5:
                    # Para pocos valores faltantes, usar moda/mediana
                    if X[col].dtype in ["object", "category"]:
                        valor_moda = X[col].mode()
                        self.valores_imputacion[col] = (
                            valor_moda[0] if len(valor_moda) > 0 else "Desconocido"
                        )
                    else:
                        self.valores_imputacion[col] = X[col].median()
                elif porcentaje_faltante < 20:
                    # Usar moda para categóricos, mediana para numéricos
                    if X[col].dtype in ["object", "category"]:
                        valor_moda = X[col].mode()
                        self.valores_imputacion[col] = (
                            valor_moda[0] if len(valor_moda) > 0 else None
                        )
                    else:
                        self.valores_imputacion[col] = X[col].median()
                else:
                    # Para > 20% de valores faltantes en columnas numéricas, imputar con 0
                    # Esto es crítico para columnas como resultado_numerico en laboratorios
                    if X[col].dtype in ["float64", "float32", "int64", "int32"]:
                        self.valores_imputacion[col] = 0.0
                    # Para categóricos con muchos faltantes, no imputar (demasiado arriesgado)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transformar DataFrame imputando valores faltantes

        Args:
            X: DataFrame a transformar

        Returns:
            DataFrame transformado con valores imputados
        """
        X_copia = X.copy()
        self.valores_faltantes = []

        for col in X.columns:
            total = len(X_copia)
            nulos_antes = X_copia[col].isna().sum()
            porcentaje = (nulos_antes / total * 100) if total > 0 else 0

            if (
                col in self.valores_imputacion
                and self.valores_imputacion[col] is not None
            ):
                X_copia[col] = X_copia[col].fillna(self.valores_imputacion[col])
                nulos_despues = X_copia[col].isna().sum()
                registros_imputados = nulos_antes - nulos_despues

                if X_copia[col].dtype in ["object", "category"]:
                    accion = f"Imputados con '{self.valores_imputacion[col]}' ({registros_imputados} registros)"
                else:
                    valor = self.valores_imputacion[col]
                    if porcentaje >= 20:
                        accion = f"Imputados con 0 ({registros_imputados} registros) - alta tasa de faltantes ({porcentaje:.1f}%)"
                    else:
                        accion = f"Imputados con mediana ({valor:.2f}) - {registros_imputados} registros"
            elif nulos_antes > 0 and porcentaje >= 20:
                nulos_despues = nulos_antes
                accion = f"No imputado (categórico) - porcentaje alto ({porcentaje:.2f}%). Requiere análisis manual"
            else:
                nulos_despues = nulos_antes
                accion = "No requiere imputación - sin valores faltantes"

            if nulos_antes > 0 or porcentaje > 0:
                self.valores_faltantes.append(
                    AnalisisValoresFaltantes(
                        campo=col,
                        total_registros=total,
                        valores_faltantes=nulos_despues,
                        porcentaje_faltante=round(
                            (nulos_despues / total * 100) if total > 0 else 0, 2
                        ),
                        recomendacion=accion,
                    )
                )

        return X_copia


class WinsorizadorOutliersIQR:
    """
    Transformador compatible con sklearn para corrección de outliers usando método IQR

    Aplica Winsorization: limita valores extremos a límites basados en IQR
    """

    def __init__(self, multiplicador_iqr: float = 1.5) -> None:
        self.multiplicador_iqr = multiplicador_iqr
        self.outliers_list: List[OutliersDetectados] = []
        self.limites: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "WinsorizadorOutliersIQR":
        """
        Ajustar winsorizador calculando límites IQR para columnas numéricas

        Args:
            X: DataFrame a ajustar
            y: Ignorado (para compatibilidad con sklearn)

        Returns:
            Instancia ajustada del winsorizador
        """
        self.limites = {}
        columnas_numericas = X.select_dtypes(include=[np.number]).columns

        for col in columnas_numericas:
            if X[col].notna().sum() > 0:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                limite_inferior = q1 - self.multiplicador_iqr * iqr
                limite_superior = q3 + self.multiplicador_iqr * iqr

                self.limites[col] = {
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "inferior": limite_inferior,
                    "superior": limite_superior,
                }

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transformar DataFrame winsorizando outliers

        Args:
            X: DataFrame a transformar

        Returns:
            DataFrame transformado con valores winsorizados
        """
        X_copia = X.copy()
        self.outliers_list = []

        for col, limites in self.limites.items():
            if col in X_copia.columns:
                # Detectar outliers antes de corrección
                mascara_outliers = (X_copia[col] < limites["inferior"]) | (
                    X_copia[col] > limites["superior"]
                )
                conteo_outliers_antes = mascara_outliers.sum()
                valores_outliers_raw = (
                    X_copia.loc[mascara_outliers, col].sort_values().head(20).tolist()
                )

                # Aplicar winsorization
                X_copia[col] = X_copia[col].clip(
                    lower=limites["inferior"], upper=limites["superior"]
                )

                total = X_copia[col].notna().sum()
                porcentaje_outliers = (
                    (conteo_outliers_antes / total * 100) if total > 0 else 0
                )

                self.outliers_list.append(
                    OutliersDetectados(
                        campo=col,
                        total_registros=int(total),
                        q1=round(float(limites["q1"]), 2),
                        q3=round(float(limites["q3"]), 2),
                        iqr=round(float(limites["iqr"]), 2),
                        limite_inferior=round(float(limites["inferior"]), 2),
                        limite_superior=round(float(limites["superior"]), 2),
                        outliers_detectados=int(conteo_outliers_antes),
                        porcentaje_outliers=round(float(porcentaje_outliers), 2),
                        valores_outliers=[float(v) for v in valores_outliers_raw],
                    )
                )

        return X_copia


def generar_resumen_procesamiento(
    normalizaciones: List[NormalizacionCampo],
    valores_faltantes: List[AnalisisValoresFaltantes],
    outliers_list: List[OutliersDetectados],
) -> Dict[str, Any]:
    """
    Generate executive summary of data processing

    Args:
        normalizaciones: List of normalization results
        valores_faltantes: List of missing values imputation results
        outliers_list: List of outlier correction results

    Returns:
        Dictionary with summary statistics
    """
    total_registros_normalizados = sum(n.registros_modificados for n in normalizaciones)
    total_valores_imputados = sum(
        v.total_registros - v.valores_faltantes for v in valores_faltantes
    )
    total_outliers_corregidos = sum(o.outliers_detectados for o in outliers_list)

    return {
        "timestamp": datetime.now().isoformat(),
        "transformaciones_aplicadas": {
            "normalizacion_categoricas": {
                "campos_procesados": len(normalizaciones),
                "registros_modificados": total_registros_normalizados,
                "descripcion": "Capitalización y estandarización de texto",
            },
            "imputacion_valores_faltantes": {
                "campos_procesados": len(valores_faltantes),
                "registros_imputados": total_valores_imputados,
                "descripcion": "Relleno de valores NULL con moda/mediana o valores por defecto",
            },
            "correccion_outliers": {
                "campos_procesados": len(outliers_list),
                "outliers_corregidos": total_outliers_corregidos,
                "descripcion": "Winsorization - valores extremos limitados a límites IQR",
            },
        },
        "total_registros_modificados": total_registros_normalizados
        + total_valores_imputados
        + total_outliers_corregidos,
        "estado": "✓ Procesamiento y transformaciones completados exitosamente",
        "advertencia": "Los datos en la base de datos han sido modificados permanentemente",
    }


async def cargar_datos_a_dataframes(session: Session) -> Dict[str, pd.DataFrame]:
    """
    Cargar todos los datos de la base de datos a DataFrames de pandas

    Args:
        session: Sesión de base de datos

    Returns:
        Diccionario con nombres de tablas como claves y DataFrames como valores
    """
    tablas = {
        "paciente": "SELECT * FROM paciente",
        "consulta": "SELECT * FROM consulta",
        "laboratorio": "SELECT * FROM laboratorio",
    }

    dataframes = {}
    for nombre_tabla, consulta in tablas.items():
        df = pd.read_sql(consulta, session.bind)
        dataframes[nombre_tabla] = df

    return dataframes


async def guardar_dataframes_a_db(
    session: Session, dataframes: Dict[str, pd.DataFrame]
) -> None:
    """
    Guardar DataFrames limpios de vuelta a la base de datos

    Args:
        session: Sesión de base de datos
        dataframes: Diccionario con nombres de tablas y DataFrames limpios
    """
    # Orden correcto para eliminar respetando restricciones de clave foránea
    # Primero tablas hijas (con FK), luego tabla padre
    orden_eliminacion = ["laboratorio", "consulta", "paciente"]

    # Eliminar registros existentes en el orden correcto
    for nombre_tabla in orden_eliminacion:
        if nombre_tabla in dataframes:
            session.execute(text(f"DELETE FROM {nombre_tabla}"))

    # Mapeo de columnas de fecha por tabla
    columnas_fecha = {
        "paciente": ["fecha_dx"],
        "consulta": ["fecha_consulta"],
        "laboratorio": ["fecha_muestra"],
    }

    # Insertar datos limpios usando SQLModel para mantener tipos correctos
    for nombre_tabla, df in dataframes.items():
        # Convertir columnas de fecha a objetos date de Python
        if nombre_tabla in columnas_fecha:
            for col_fecha in columnas_fecha[nombre_tabla]:
                if col_fecha in df.columns:
                    # Convertir a datetime y extraer solo la fecha
                    df[col_fecha] = pd.to_datetime(df[col_fecha]).dt.date

        # Convertir DataFrame a lista de diccionarios e insertar fila por fila
        # usando los modelos SQLModel para mantener los tipos correctos
        if nombre_tabla == "paciente":
            for _, row in df.iterrows():
                paciente = Paciente(**row.to_dict())
                session.add(paciente)
        elif nombre_tabla == "consulta":
            for _, row in df.iterrows():
                consulta = Consulta(**row.to_dict())
                session.add(consulta)
        elif nombre_tabla == "laboratorio":
            for _, row in df.iterrows():
                laboratorio = Laboratorio(**row.to_dict())
                session.add(laboratorio)

    session.commit()


async def procesar_limpieza_datos(session: Session) -> ReporteProcesamiento:
    """
    Ejecutar pipeline de limpieza y estandarización de datos usando sklearn y pandas

    Esta función implementa un Pipeline de scikit-learn con tres transformadores:
    1. NormalizadorCategorico: Normalizar variables categóricas
    2. ImputadorAdaptativoValoresFaltantes: Imputar valores faltantes adaptativamente
    3. WinsorizadorOutliersIQR: Corregir outliers usando método IQR

    El pipeline usa pandas para manipulación eficiente de datos en lugar de iteraciones SQL.
    Todas las transformaciones modifican los datos en la base de datos permanentemente.

    Args:
        session: Sesión de base de datos

    Returns:
        ReporteProcesamiento con reporte completo de transformaciones incluyendo:
        - Normalizaciones de variables categóricas
        - Imputación de valores faltantes
        - Correcciones de outliers
        - Resumen ejecutivo
    """
    # Cargar datos de la base de datos a DataFrames de pandas
    dataframes = await cargar_datos_a_dataframes(session)

    # Inicializar colecciones de resultados
    todas_normalizaciones: List[NormalizacionCampo] = []
    todos_valores_faltantes: List[AnalisisValoresFaltantes] = []
    todos_outliers: List[OutliersDetectados] = []
    dataframes_limpios: Dict[str, pd.DataFrame] = {}

    # Procesar cada tabla por separado con su propio pipeline de sklearn
    for nombre_tabla, df in dataframes.items():
        # Inicializar transformadores
        normalizador_cat = NormalizadorCategorico()
        imputador_faltantes = ImputadorAdaptativoValoresFaltantes()
        winsorizador_outliers = WinsorizadorOutliersIQR()

        # Crear pipeline de sklearn con tres pasos de transformación
        pipeline = Pipeline(
            steps=[
                ("normalizar", normalizador_cat),
                ("imputar", imputador_faltantes),
                ("winsorizar", winsorizador_outliers),
            ]
        )

        # Ajustar y transformar datos usando el pipeline
        df_limpio = pipeline.fit_transform(df)

        # Almacenar dataframe limpio
        dataframes_limpios[nombre_tabla] = df_limpio

        # Recolectar reportes de cada transformador con prefijos de tabla
        for norm in normalizador_cat.normalizaciones:
            norm.campo = f"{nombre_tabla}.{norm.campo}"
            todas_normalizaciones.append(norm)

        for val_falt in imputador_faltantes.valores_faltantes:
            val_falt.campo = f"{nombre_tabla}.{val_falt.campo}"
            todos_valores_faltantes.append(val_falt)

        for outlier in winsorizador_outliers.outliers_list:
            outlier.campo = f"{nombre_tabla}.{outlier.campo}"
            todos_outliers.append(outlier)

    # Guardar datos limpios de vuelta a la base de datos
    await guardar_dataframes_a_db(session, dataframes_limpios)

    # Generar resumen ejecutivo
    resumen = generar_resumen_procesamiento(
        todas_normalizaciones, todos_valores_faltantes, todos_outliers
    )

    return ReporteProcesamiento(
        normalizaciones=todas_normalizaciones,
        valores_faltantes=todos_valores_faltantes,
        outliers=todos_outliers,
        resumen=resumen,
    )


async def generar_dataset_modelado(session: Session) -> List[DatasetModelado]:
    """
    Generar dataset listo para modelado con features agregados

    Args:
        session: Sesión de base de datos

    Returns:
        Lista de objetos DatasetModelado - una fila por paciente con:
        - Todas las variables categóricas
        - Conteos agregados (consultas, laboratorios)
        - Promedios de resultados de laboratorio por tipo de prueba
        - Sin valores nulos (manejados apropiadamente)
        - Tipos de datos correctos
    """
    # Consulta SQL compleja para generar dataset agregado
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

    # Convertir filas a modelos Pydantic
    # Acceder a los valores por índice ya que text() retorna tuplas
    dataset = []
    for row in rows:
        # Helper function para manejar None y NaN
        def safe_float(val):
            """Convertir valor a float, manejando None y NaN"""
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return 0.0
            return float(val)

        dataset.append(
            DatasetModelado(
                sexo=row[0],
                edad=row[1],
                zona_residencia=row[2],
                tipo_cancer=row[3],
                estadio=row[4],
                aseguradora=row[5],
                count_consultas=row[6] if row[6] is not None else 0,
                dias_desde_diagnostico=row[7],
                count_laboratorios=row[8] if row[8] is not None else 0,
                avg_resultado_numerico=round(safe_float(row[9]), 2),
                avg_biopsia=round(safe_float(row[10]), 2),
                avg_vpH=round(safe_float(row[11]), 2),
                avg_marcador_ca125=round(safe_float(row[12]), 2),
                avg_psa=round(safe_float(row[13]), 2),
                avg_colonoscopia=round(safe_float(row[14]), 2),
                adherencia_12m=row[15],
            )
        )

    return dataset


async def generar_y_guardar_dataset_modelado(session: Session) -> RutaDataset:
    """
    Genera un dataset consolidado listo para modelado y lo guarda en la carpeta ./data

    Construye un único DataFrame que consolida:
    - Información del paciente
    - Número total de consultas
    - Número total de laboratorios
    - Promedio de resultados numéricos por tipo de prueba

    El dataset generado incluye:
    - Una fila por paciente
    - Todas las variables agregadas
    - Sin nulos (manejados apropiadamente)
    - Con tipos de datos correctos

    Args:
        session: Sesión de base de datos

    Returns:
        RutaDataset con la ruta del archivo guardado y metadatos
    """
    # Obtener dataset usando la función existente
    dataset_list = await generar_dataset_modelado(session)

    # Convertir a DataFrame
    df = pd.DataFrame([d.model_dump() for d in dataset_list])

    # Verificar que no haya NaN (deberían estar manejados en generar_dataset_modelado)
    if df.isna().any().any():
        logger.warning(
            "Se encontraron valores NaN inesperados. Reemplazando con 0 como fallback."
        )
        df = df.fillna(0)

    # Crear carpeta data si no existe
    workspace_root = Path(__file__).parent.parent.parent
    carpeta_data = workspace_root / "data"
    carpeta_data.mkdir(exist_ok=True)

    # Generar nombre de archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"dataset_modelado_{timestamp}.csv"
    ruta_archivo = carpeta_data / nombre_archivo

    # Guardar DataFrame como CSV
    df.to_csv(ruta_archivo, index=False, encoding="utf-8")

    logger.info(
        f"Dataset consolidado guardado exitosamente: {ruta_archivo} "
        f"({len(df)} registros, {len(df.columns)} columnas)"
    )

    # Retornar ruta relativa desde la raíz del workspace
    ruta_relativa = f"./data/{nombre_archivo}"

    return RutaDataset(
        ruta_archivo=ruta_relativa,
        total_registros=len(df),
        descripcion=(
            f"Dataset consolidado con {len(df)} pacientes. "
            f"Incluye: datos demográficos, clínicos, conteos de consultas/laboratorios, "
            f"y promedios de resultados por tipo de prueba. "
            f"Listo para modelado de Machine Learning."
        ),
    )
