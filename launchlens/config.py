from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Remote LLM providers (placeholders — populated when keys arrive) ──
    anthropic_api_key: str | None = None
    sarvam_api_key: str | None = None
    sarvam_base_url: str = "https://api.sarvam.ai/v1"

    # ── Local LLM via Ollama (default on this 8 GB VRAM dev hardware) ──
    ollama_base_url: str = "http://localhost:11434/v1"
    # Default models (Q4 GGUF tags pulled by scripts/setup_local_models.sh).
    ollama_default_model: str = "qwen2.5:3b-instruct-q4_K_M"     # English + multilingual
    ollama_indic_model: str = "qwen2.5:3b-instruct-q4_K_M"       # Hindi/regional; swap to sarvam-1 when available
    ollama_fast_model: str = "gemma2:2b-instruct-q4_K_M"         # smaller/faster

    # Engine selection: "auto" | "mock" | "local" | "sarvam" | "claude"
    launchlens_engine: str = "auto"

    # ── Pricing (USD per 1M tokens). Used by LLMUsageTracker. ──
    # Local Ollama runs are always $0; only remote providers consume cost.
    sarvam_input_price_per_mtok: float = 0.50
    sarvam_output_price_per_mtok: float = 1.50
    claude_input_price_per_mtok: float = 3.00     # Sonnet 4.6 list price
    claude_output_price_per_mtok: float = 15.00

    # ── DB (optional, off by default) ──
    postgres_url: str | None = None
    redis_url: str | None = None

    # ── Data dirs ──
    census_data_dir: Path = Path("./data/raw/census")
    nfhs_data_dir: Path = Path("./data/raw/nfhs")
    trai_data_dir: Path = Path("./data/raw/trai")
    nsso_data_dir: Path = Path("./data/raw/nsso")
    district_profiles_dir: Path = Path("./data/processed/districts")
    calibration_dir: Path = Path("./data/calibration")

    # ── Simulation defaults (tuned for 8 GB VRAM laptop) ──
    default_agent_count: int = 100
    llm_batch_size: int = 16
    llm_max_concurrent_local: int = 4
    llm_max_concurrent_remote: int = 8
    max_prompt_tokens: int = 3500

    # Response cache
    cache_dir: Path = Path("~/.launchlens/cache").expanduser()
    cache_size_limit_gb: float = 2.0

    # Cost guardrail: require explicit --confirm-cost when projected spend exceeds
    cost_confirm_threshold_usd: float = 10.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
