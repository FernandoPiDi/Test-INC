from datetime import date
from typing import Optional, List
from sqlmodel import Field, Relationship, SQLModel


# ------------------------------------------------------------------------------
# Paciente Model
# ------------------------------------------------------------------------------
# This class represents the 'Pacientes' table.
# It serves as the central table, linked to by Consultas and Laboratorios.
# ------------------------------------------------------------------------------
class Paciente(SQLModel, table=True):
    # Primary Key
    id_paciente: str = Field(primary_key=True)

    # Patient demographics and info
    sexo: str
    edad: int
    zona_residencia: Optional[str] = Field(default=None)  # Visto un blank in CSV
    fecha_dx: date

    # Clinical details
    tipo_cancer: str
    estadio: str
    aseguradora: str
    adherencia_12m: bool

    # --- Relationships ---
    # One-to-Many relationship with Consulta
    # One patient can have many consultations
    consultas: List["Consulta"] = Relationship(back_populates="paciente")

    # One-to-Many relationship with Laboratorio
    # One patient can have many lab results
    laboratorios: List["Laboratorio"] = Relationship(back_populates="paciente")


# ------------------------------------------------------------------------------
# Consulta Model
# ------------------------------------------------------------------------------
# This class represents the 'Consultas' table.
# Each record is linked to a single Paciente.
# ------------------------------------------------------------------------------
class Consulta(SQLModel, table=True):
    # Primary Key
    id_consulta: str = Field(primary_key=True)

    # Consultation details
    fecha_consulta: date
    motivo: str
    prioridad: str
    especialista: str

    # --- Foreign Key & Relationship ---
    # Foreign key linking this consultation to a patient
    id_paciente: str = Field(foreign_key="paciente.id_paciente")

    # Many-to-One relationship with Paciente
    # Allows easy access like `consulta_instance.paciente`
    paciente: Paciente = Relationship(back_populates="consultas")


# ------------------------------------------------------------------------------
# Laboratorio Model
# ------------------------------------------------------------------------------
# This class represents the 'Laboratorios' table.
# Each record is linked to a single Paciente.
# ------------------------------------------------------------------------------
class Laboratorio(SQLModel, table=True):
    # Primary Key
    id_lab: str = Field(primary_key=True)

    # Lab details
    fecha_muestra: date
    tipo_prueba: str

    # Results can be categorical (Maligna), numeric, or both.
    # Based on the CSV, these fields are often empty.
    resultado: Optional[str] = Field(default=None)
    resultado_numerico: Optional[float] = Field(
        default=None
    )  # float is safer for lab values
    unidad: Optional[str] = Field(default=None)

    # --- Foreign Key & Relationship ---
    # Foreign key linking this lab result to a patient
    id_paciente: str = Field(foreign_key="paciente.id_paciente")

    # Many-to-One relationship with Paciente
    # Allows easy access like `lab_instance.paciente`
    paciente: Paciente = Relationship(back_populates="laboratorios")
