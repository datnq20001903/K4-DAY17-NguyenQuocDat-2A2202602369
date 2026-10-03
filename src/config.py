from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from model_provider import ProviderConfig, normalize_provider


DEFAULT_COMPACT_THRESHOLD_TOKENS = 1000
DEFAULT_COMPACT_KEEP_MESSAGES = 6
_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "local-model",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-20250514",
    "ollama": "llama3.2",
    "openrouter": "openai/gpt-4o-mini",
}
_API_KEY_VARIABLES = {
    "openai": "OPENAI_API_KEY", "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
_BASE_URL_VARIABLES = {
    "openai": "OPENAI_BASE_URL", "custom": "CUSTOM_BASE_URL",
    "gemini": "GEMINI_BASE_URL", "anthropic": "ANTHROPIC_BASE_URL",
    "ollama": "OLLAMA_BASE_URL", "openrouter": "OPENROUTER_BASE_URL",
}
_DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434",
    "openrouter": "https://openrouter.ai/api/v1",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_model() -> ProviderConfig:
    return ProviderConfig("openai", _DEFAULT_MODELS["openai"], 0.0)


@dataclass
class LabConfig:
    """Shared paths, compact settings, and independent main/judge configurations."""

    base_dir: Path = field(default_factory=_repo_root)
    data_dir: Path = field(default_factory=lambda: _repo_root() / "data")
    state_dir: Path = field(default_factory=lambda: _repo_root() / "state")
    compact_threshold_tokens: int = DEFAULT_COMPACT_THRESHOLD_TOKENS
    compact_keep_messages: int = DEFAULT_COMPACT_KEEP_MESSAGES
    model: ProviderConfig = field(default_factory=_default_model)
    judge_model: ProviderConfig = field(default_factory=_default_model)


def _setting(values: Mapping[str, str | None], name: str, default: str | None = None) -> str | None:
    value = values.get(name)
    return value.strip() if value and value.strip() else default


def _positive_int(values: Mapping[str, str | None], name: str, default: int) -> int:
    try:
        value = int(_setting(values, name) or str(default))
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _model_config(
    values: Mapping[str, str | None], prefix: str, fallback: ProviderConfig | None = None,
) -> ProviderConfig:
    default_provider = fallback.provider if fallback else "openai"
    provider = normalize_provider(_setting(values, f"{prefix}_PROVIDER") or default_provider)
    if fallback is not None and fallback.provider == provider:
        default_model, default_key, default_url = fallback.model_name, fallback.api_key, fallback.base_url
    else:
        default_model = _DEFAULT_MODELS[provider]
        default_key = _setting(values, _API_KEY_VARIABLES[provider]) if provider in _API_KEY_VARIABLES else None
        if provider == "gemini" and default_key is None:
            default_key = _setting(values, "GOOGLE_API_KEY")
        default_url = _setting(values, _BASE_URL_VARIABLES[provider], _DEFAULT_BASE_URLS.get(provider))

    temperature_variable = f"{prefix}_TEMPERATURE"
    try:
        temperature = float(_setting(values, temperature_variable) or "0")
    except ValueError as exc:
        raise ValueError(f"{temperature_variable} must be a finite, nonnegative number.") from exc
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError(f"{temperature_variable} must be a finite, nonnegative number.")
    return ProviderConfig(
        provider=provider,
        model_name=_setting(values, f"{prefix}_MODEL") or default_model,
        temperature=temperature,
        api_key=_setting(values, f"{prefix}_API_KEY", default_key),
        base_url=_setting(values, f"{prefix}_BASE_URL", default_url),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Read root/.env, overlay the process environment, and create root/state.

    No model is constructed and no API key is required. python-dotenv is
    needed only when .env exists; SDKs are needed only when building live models.
    """
    root = (base_dir or _repo_root()).expanduser().resolve()
    values: dict[str, str | None] = {}
    env_path = root / ".env"
    if env_path.is_file():
        try:
            from dotenv import dotenv_values
        except ImportError as exc:
            raise ImportError("Reading .env requires: python -m pip install python-dotenv") from exc
        values.update(dotenv_values(env_path, encoding="utf-8"))
    values.update(os.environ)

    model = _model_config(values, "LLM")
    config = LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=root / "state",
        compact_threshold_tokens=_positive_int(values, "COMPACT_THRESHOLD_TOKENS", DEFAULT_COMPACT_THRESHOLD_TOKENS),
        compact_keep_messages=_positive_int(values, "COMPACT_KEEP_MESSAGES", DEFAULT_COMPACT_KEEP_MESSAGES),
        model=model,
        judge_model=_model_config(values, "JUDGE", fallback=model),
    )
    config.state_dir.mkdir(parents=True, exist_ok=True)
    return config
