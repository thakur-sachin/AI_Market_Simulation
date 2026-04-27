from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = ""
    sarvam_api_key: str = ""
    sarvam_base_url: str = "https://api.sarvam.ai/v1"

    # DB
    postgres_url: str = "postgresql://localhost:5432/launchlens"
    redis_url: str = "redis://localhost:6379/0"

    # Data dirs
    census_data_dir: Path = Path("./data/raw/census")
    nfhs_data_dir: Path = Path("./data/raw/nfhs")
    district_profiles_dir: Path = Path("./data/processed/districts")

    # Simulation defaults
    default_agent_count: int = 1000
    llm_batch_size: int = 200
    llm_max_concurrent: int = 50


@lru_cache
def get_settings() -> Settings:
    return Settings()
