"""Configuration for the weather assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


@dataclass(frozen=True)
class Settings:
    provider: str = "ollama"
    model: str = "mistral"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 600
    timeout: int = 300
    # Weather backend
    geocoding_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    default_unit: str = "celsius"
    forecast_days: int = 3
    http_timeout: int = 15
    max_retries: int = 1
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        default_model = "mistral" if provider == "ollama" else "gpt-4o-mini"
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL", default_model),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "600")),
            timeout=int(os.getenv("LLM_TIMEOUT", "300")),
            geocoding_url=os.getenv(
                "GEOCODING_URL", "https://geocoding-api.open-meteo.com/v1/search"
            ),
            forecast_url=os.getenv(
                "FORECAST_URL", "https://api.open-meteo.com/v1/forecast"
            ),
            default_unit=os.getenv("WEATHER_UNIT", "celsius"),
            forecast_days=int(os.getenv("FORECAST_DAYS", "3")),
            http_timeout=int(os.getenv("HTTP_TIMEOUT", "15")),
            max_retries=int(os.getenv("MAX_WEATHER_RETRIES", "1")),
            api_key=os.getenv("OPENAI_API_KEY") if provider == "openai" else None,
        )


UNIT_SYMBOLS = {"celsius": "°C", "fahrenheit": "°F"}
