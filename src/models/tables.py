"""
Modelos de base de datos para pacientes, consultas y laboratorios

Define las tablas principales de la base de datos:
- Paciente: Información demográfica y clínica de pacientes
- Consulta: Registro de consultas médicas
- Laboratorio: Resultados de pruebas de laboratorio
"""

from datetime import date
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


# ============================================================================
# MODELO PACIENTE
# ============================================================================
class Paciente(SQLModel, table=True):
    """
    Modelo de la tabla Pacientes

    Representa la información demográfica y clínica de un paciente oncológico.
    Tabla central que se vincula con Consultas y Laboratorios.
    """

    # Clave primaria
    id_paciente: str = Field(primary_key=True)

    # Datos demográficos
    sexo: str
    edad: int
    zona_residencia: Optional[str] = Field(default=None)
    fecha_dx: date

    # Información clínica
    tipo_cancer: str
    estadio: str
    aseguradora: str
    adherencia_12m: bool

    # Relaciones uno-a-muchos
    consultas: List["Consulta"] = Relationship(back_populates="paciente")
    laboratorios: List["Laboratorio"] = Relationship(back_populates="paciente")


# ============================================================================
# MODELO CONSULTA
# ============================================================================
class Consulta(SQLModel, table=True):
    """
    Modelo de la tabla Consultas

    Representa una consulta médica de un paciente.
    Cada registro está vinculado a un único paciente.
    """

    # Clave primaria
    id_consulta: str = Field(primary_key=True)

    # Información de la consulta
    fecha_consulta: date
    motivo: str
    prioridad: str
    especialista: str

    # Clave foránea y relación
    id_paciente: str = Field(foreign_key="paciente.id_paciente")
    paciente: Paciente = Relationship(back_populates="consultas")


# ============================================================================
# MODELO LABORATORIO
# ============================================================================
class Laboratorio(SQLModel, table=True):
    """
    Modelo de la tabla Laboratorios

    Representa un resultado de prueba de laboratorio de un paciente.
    Cada registro está vinculado a un único paciente.
    Los resultados pueden ser categóricos, numéricos, o ambos.
    """

    # Clave primaria
    id_lab: str = Field(primary_key=True)

    # Información de la prueba
    fecha_muestra: date
    tipo_prueba: str

    # Resultados (pueden ser nulos)
    resultado: Optional[str] = Field(default=None)
    resultado_numerico: Optional[float] = Field(default=None)
    unidad: Optional[str] = Field(default=None)

    # Clave foránea y relación
    id_paciente: str = Field(foreign_key="paciente.id_paciente")
    paciente: Paciente = Relationship(back_populates="laboratorios")
