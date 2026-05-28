from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM (hosted)
    anthropic_api_key: str = ""
    sarvam_api_key: str = ""
    sarvam_base_url: str = "https://api.sarvam.ai/v1"

    # LLM (local — Ollama / llama.cpp / LM Studio, all OpenAI-compatible)
    local_llm_enabled: bool = False
    local_llm_url: str = "http://localhost:11434/v1"
    local_llm_model: str = "qwen2.5:0.5b"
    local_llm_api_key: str = "ollama"   # placeholder; most local servers ignore this

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
    local_llm_max_concurrent: int = 4   # tiny models choke on high parallelism


@lru_cache
def get_settings() -> Settings:
    return Settings()
