from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlmodel import Session, create_engine


class Settings(BaseSettings):
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
        return f"postgresql+psycopg://{self.db_username}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"


settings = Settings()  # type: ignore[call-arg]

# Create database engine
engine = create_engine(settings.database_url, echo=False)


def get_session():
    """
    Dependency function to get database session
    """
    with Session(engine) as session:
        yield session
