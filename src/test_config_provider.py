from __future__ import annotations

import importlib
import os
from dataclasses import fields
from pathlib import Path

import pytest

import config
from config import LabConfig, load_config
from model_provider import ProviderConfig, build_chat_model, normalize_provider


@pytest.fixture(autouse=True)
def clean_config_environment(monkeypatch):
    prefixes = (
        "LLM_", "JUDGE_", "COMPACT_", "OPENAI_", "CUSTOM_", "GEMINI_",
        "GOOGLE_", "ANTHROPIC_", "OLLAMA_", "OPENROUTER_",
    )
    for name in tuple(os.environ):
        if name.startswith(prefixes):
            monkeypatch.delenv(name)


def test_config_paths_are_resolved_and_state_is_created(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(Path("lab"))
    assert cfg.base_dir == tmp_path / "lab"
    assert cfg.data_dir == tmp_path / "lab" / "data"
    assert cfg.state_dir == tmp_path / "lab" / "state"
    assert cfg.state_dir.is_dir()
    assert load_config(Path("lab")) == cfg


def test_default_root_does_not_depend_on_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg.base_dir == Path(config.__file__).resolve().parent.parent
    assert cfg.data_dir == cfg.base_dir / "data"


def test_dataclass_defaults_preserve_the_seven_field_contract():
    first = LabConfig()
    second = LabConfig()
    assert {field.name for field in fields(first)} == {
        "base_dir", "data_dir", "state_dir", "compact_threshold_tokens",
        "compact_keep_messages", "model", "judge_model",
    }
    assert first.base_dir.is_absolute()
    assert first.data_dir == first.base_dir / "data"
    assert first.state_dir == first.base_dir / "state"
    assert first.compact_threshold_tokens > 0
    assert first.compact_keep_messages > 0
    assert first.model is not first.judge_model
    assert first.model is not second.model


def test_config_without_keys_does_not_build_a_live_model(tmp_path):
    cfg = load_config(tmp_path)
    assert cfg.model.api_key is None
    assert cfg.judge_model.api_key is None
    assert build_chat_model(cfg.model) is None
    assert build_chat_model(cfg.judge_model) is None


def test_dotenv_is_read_from_the_selected_root_without_mutating_environment(tmp_path):
    (tmp_path / ".env").write_text(
        "LLM_PROVIDER=custom\nLLM_MODEL=local-model\n"
        "CUSTOM_BASE_URL=http://localhost:8000/v1\n"
        "COMPACT_THRESHOLD_TOKENS=900\nCOMPACT_KEEP_MESSAGES=4\n",
        encoding="utf-8",
    )
    cfg = load_config(tmp_path)
    assert cfg.model.provider == "custom"
    assert cfg.model.model_name == "local-model"
    assert cfg.model.base_url == "http://localhost:8000/v1"
    assert cfg.compact_threshold_tokens == 900
    assert cfg.compact_keep_messages == 4
    assert "LLM_PROVIDER" not in os.environ


def test_process_environment_overrides_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "LLM_PROVIDER=openai\nLLM_MODEL=file-model\nCOMPACT_THRESHOLD_TOKENS=900\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("COMPACT_THRESHOLD_TOKENS", "700")
    cfg = load_config(tmp_path)
    assert cfg.model.provider == "ollama"
    assert cfg.model.model_name == "env-model"
    assert cfg.compact_threshold_tokens == 700


@pytest.mark.parametrize("provider,key_variable", [
    ("openai", "OPENAI_API_KEY"), ("custom", "CUSTOM_API_KEY"),
    ("gemini", "GEMINI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY"),
    ("ollama", None), ("openrouter", "OPENROUTER_API_KEY"),
])
def test_config_reads_credentials_for_the_selected_provider(tmp_path, monkeypatch, provider, key_variable):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    if key_variable:
        monkeypatch.setenv(key_variable, "cp3-test-key")
    cfg = load_config(tmp_path)
    assert cfg.model.provider == provider
    assert cfg.model.model_name
    assert cfg.model.api_key == ("cp3-test-key" if key_variable else None)
    assert cfg.judge_model == cfg.model
    assert cfg.judge_model is not cfg.model


def test_judge_can_use_a_separate_provider_and_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "main-test-key")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.4")
    monkeypatch.setenv("JUDGE_PROVIDER", "anthorpic")
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("JUDGE_API_KEY", "judge-test-key")
    cfg = load_config(tmp_path)
    assert cfg.model.api_key == "main-test-key"
    assert cfg.model.temperature == 0.4
    assert cfg.judge_model.provider == "anthropic"
    assert cfg.judge_model.model_name == "judge-model"
    assert cfg.judge_model.api_key == "judge-test-key"
    assert cfg.judge_model.temperature == 0.0


@pytest.mark.parametrize("variable,value", [
    ("COMPACT_THRESHOLD_TOKENS", "0"), ("COMPACT_THRESHOLD_TOKENS", "-1"),
    ("COMPACT_THRESHOLD_TOKENS", "invalid"), ("COMPACT_KEEP_MESSAGES", "0"),
    ("LLM_TEMPERATURE", "-0.1"), ("JUDGE_TEMPERATURE", "nan"),
])
def test_invalid_numeric_config_names_the_setting(tmp_path, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=variable):
        load_config(tmp_path)


@pytest.mark.parametrize("value,expected", [
    (" OpenAI ", "openai"), ("custom", "custom"), ("Gemini", "gemini"),
    ("anthorpic", "anthropic"), ("anthropic", "anthropic"),
    ("ollama", "ollama"), ("OpenRouter", "openrouter"),
])
def test_provider_names_are_normalized(value, expected):
    assert normalize_provider(value) == expected


@pytest.mark.parametrize("value", ["", "unsupported-provider"])
def test_unknown_provider_has_an_explicit_error(value):
    with pytest.raises(ValueError, match="provider"):
        normalize_provider(value)


@pytest.mark.parametrize("provider,module,class_name,model_attribute", [
    ("openai", "langchain_openai", "ChatOpenAI", "model_name"),
    ("custom", "langchain_openai", "ChatOpenAI", "model_name"),
    ("gemini", "langchain_google_genai", "ChatGoogleGenerativeAI", "model"),
    ("anthropic", "langchain_anthropic", "ChatAnthropic", "model"),
    ("ollama", "langchain_ollama", "ChatOllama", "model"),
    ("openrouter", "langchain_openrouter", "ChatOpenRouter", "model_name"),
])
def test_live_models_are_constructed_without_api_requests(provider, module, class_name, model_attribute):
    sdk = pytest.importorskip(module)
    key = None if provider in {"custom", "ollama"} else "cp3-test-key"
    url = "http://localhost:8000/v1" if provider == "custom" else None
    model = build_chat_model(ProviderConfig(provider, "cp3-model", 0.0, key, url))
    assert isinstance(model, getattr(sdk, class_name))
    assert getattr(model, model_attribute) == "cp3-model"
    assert model.temperature == 0.0
    if provider == "custom":
        assert model.openai_api_base == url


@pytest.mark.parametrize("provider", ["openai", "gemini", "anthropic", "openrouter"])
def test_missing_cloud_key_preserves_offline_fallback(provider):
    assert build_chat_model(ProviderConfig(provider, "cp3-model", 0.0)) is None


def test_custom_provider_requires_an_explicit_base_url():
    with pytest.raises(ValueError, match="base_url"):
        build_chat_model(ProviderConfig("custom", "cp3-model", 0.0))


def test_missing_live_sdk_has_an_installation_hint(monkeypatch):
    def missing_sdk(name):
        raise ModuleNotFoundError(f"No module named '{name}'")

    monkeypatch.setattr(importlib, "import_module", missing_sdk)
    with pytest.raises(ImportError, match="langchain-openai"):
        build_chat_model(ProviderConfig("openai", "cp3-model", 0.0, "cp3-test-key"))
