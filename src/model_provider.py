from __future__ import annotations

import importlib
from dataclasses import dataclass, field


SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")
_MODEL_CLASSES = {
    "openai": ("langchain_openai", "ChatOpenAI"),
    "custom": ("langchain_openai", "ChatOpenAI"),
    "gemini": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic"),
    "ollama": ("langchain_ollama", "ChatOllama"),
    "openrouter": ("langchain_openrouter", "ChatOpenRouter"),
}


@dataclass
class ProviderConfig:
    """Provider settings shared by agents and the benchmark judge."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Normalize supported names and reject unknown providers immediately."""
    provider = value.strip().lower()
    provider = {"anthorpic": "anthropic"}.get(provider, provider)
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"Unknown provider {value!r}; supported providers: {supported}.")
    return provider


def build_chat_model(config: ProviderConfig):
    """Build a live model lazily, or return None for a cloud model without a key.

    Construction does not send a chat request. Call this only in live mode;
    local custom endpoints and Ollama do not require an API key.
    """
    provider = normalize_provider(config.provider)
    if provider == "custom" and not (config.base_url or "").strip():
        raise ValueError("The custom provider requires base_url (CUSTOM_BASE_URL or LLM_BASE_URL).")
    api_key = (config.api_key or "").strip()
    if provider not in {"custom", "ollama"} and not api_key:
        return None

    module_name, class_name = _MODEL_CLASSES[provider]
    try:
        model_class = getattr(importlib.import_module(module_name), class_name)
    except ImportError as exc:
        package = module_name.replace("_", "-")
        raise ImportError(
            f"Live provider {provider!r} requires {package}; "
            f"install it with: python -m pip install {package}"
        ) from exc

    kwargs = {"model": config.model_name, "temperature": config.temperature}
    if provider != "ollama":
        # The OpenAI SDK requires a nonempty key even for unauthenticated local servers.
        kwargs["api_key"] = api_key or "not-required"
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return model_class(**kwargs)
