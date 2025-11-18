"""
Configuración del sistema de logging de la aplicación

Gestiona el logging a consola y archivo con formato estandarizado.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Crear directorio de logs
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configurar sistema de logging para toda la aplicación

    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Ruta opcional para logging a archivo
        format_string: Formato opcional personalizado
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )

    # Crear formateador
    formatter = logging.Formatter(format_string)

    # Logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Eliminar handlers existentes
    root_logger.handlers.clear()

    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Handler para archivo (si se especifica)
    if log_file:
        file_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # También registrar en log general de la aplicación
    app_log_file = LOGS_DIR / "app.log"
    app_file_handler = logging.FileHandler(app_log_file)
    app_file_handler.setLevel(logging.DEBUG)
    app_file_handler.setFormatter(formatter)
    root_logger.addHandler(app_file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Obtener instancia de logger para un módulo

    Args:
        name: Nombre del logger (típicamente __name__)

    Returns:
        Instancia de logger configurada
    """
    return logging.getLogger(name)


# Inicializar logging al importar
setup_logging()
