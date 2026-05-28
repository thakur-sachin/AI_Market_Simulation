from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Remote LLM providers (populated when keys arrive) ──
    anthropic_api_key: str | None = None
    sarvam_api_key: str | None = None
    sarvam_base_url: str = "https://api.sarvam.ai/v1"

    # ── Local LLM via Ollama (default on 8 GB VRAM dev hardware) ──
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_default_model: str = "qwen2.5:3b-instruct-q4_K_M"
    ollama_indic_model: str = "qwen2.5:3b-instruct-q4_K_M"
    ollama_fast_model: str = "gemma2:2b-instruct-q4_K_M"

    # Engine selection: "auto" | "mock" | "local" | "sarvam" | "claude"
    launchlens_engine: str = "auto"

    # ── Pricing (USD per 1M tokens) ──
    sarvam_input_price_per_mtok: float = 0.50
    sarvam_output_price_per_mtok: float = 1.50
    claude_input_price_per_mtok: float = 3.00
    claude_output_price_per_mtok: float = 15.00

    # ── DB (optional) ──
    postgres_url: str | None = None
    redis_url: str | None = None

    # ── Data dirs ──
    census_data_dir: Path = Path("./data/raw/census")
    nfhs_data_dir: Path = Path("./data/raw/nfhs")
    trai_data_dir: Path = Path("./data/raw/trai")
    nsso_data_dir: Path = Path("./data/raw/nsso")
    district_profiles_dir: Path = Path("./data/processed/districts")
    calibration_dir: Path = Path("./data/calibration")

    # ── Simulation defaults ──
    default_agent_count: int = 100
    llm_batch_size: int = 16
    llm_max_concurrent_local: int = 4
    llm_max_concurrent_remote: int = 8
    max_prompt_tokens: int = 3500

    # ── Response cache ──
    cache_dir: Path = Path("~/.launchlens/cache").expanduser()
    cache_size_limit_gb: float = 2.0

    # ── Cost guardrail ──
    cost_confirm_threshold_usd: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """Clear the lru_cache and re-read env. Use after mutating os.environ at runtime."""
    get_settings.cache_clear()
    return get_settings()
