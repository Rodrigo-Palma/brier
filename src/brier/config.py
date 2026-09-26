"""Settings, read from the environment. Nothing about the machine it runs on is
written into the code."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every value has a default that works on a laptop with Ollama running."""

    model_config = SettingsConfigDict(env_prefix="BRIER_", env_file=".env", extra="ignore")

    ollama_url: str = "http://localhost:11434"
    encoder_model: str = "nomic-embed-text"
    weights_path: Path = Path("models/brier.npz")
    cache_path: Path = Path(".cache/encoder")
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
