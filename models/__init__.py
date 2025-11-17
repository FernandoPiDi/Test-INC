"""
Models package.
Import all models here so they are automatically registered with SQLModel.metadata
"""

from models.tables import Consulta, Laboratorio, Paciente

__all__ = ["Paciente", "Consulta", "Laboratorio"]
