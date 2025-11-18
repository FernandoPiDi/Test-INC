"""
Configuración de la aplicación y base de datos

Gestiona variables de entorno y conexión a la base de datos PostgreSQL.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlmodel import Session, create_engine


class Settings(BaseSettings):
    """Configuración de la aplicación cargada desde variables de entorno"""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    db_username: str
    db_password: str
    db_host: str
    db_port: int
    db_name: str

    @property
    def database_url(self) -> str:
        """Construir URL de conexión a PostgreSQL"""
        return f"postgresql+psycopg://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


# Instancia global de configuración
settings = Settings()  # type: ignore[call-arg]

# Motor de base de datos
engine = create_engine(settings.database_url, echo=False)


def get_session():
    """
    Función de dependencia para obtener sesión de base de datos

    Yields:
        Session: Sesión de SQLModel para consultas a la base de datos
    """
    with Session(engine) as session:
        yield session
