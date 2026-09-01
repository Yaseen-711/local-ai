"""Unit tests for TOML configuration loader and validator."""

from pathlib import Path
import pytest

from core.common.errors import ConfigurationError
from core.common.types import ModelFormat, ModelRole
from core.config.loader import (
    load_model_definition,
    load_model_definitions_from_dir,
    load_settings,
)


def test_load_valid_model_toml(tmp_path: Path):
    """Verify loading a valid model TOML configuration."""
    toml_content = b"""
    [model]
    id = "test-model"
    display_name = "Test Model 7B"
    format = "gguf"
    path = "models/gguf/test-model.gguf"
    aliases = ["test", "default"]
    supported_providers = ["llama_cpp"]
    roles = ["general", "coding"]

    [capabilities]
    chat = true
    code = true
    reasoning = false
    structured_output = true
    context_window = 8192

    [metadata]
    quantization = "Q4_K_M"
    parameter_count = "7B"
    architecture = "llama"
    """
    toml_file = tmp_path / "test-model.toml"
    toml_file.write_bytes(toml_content)

    model_def = load_model_definition(toml_file)
    assert model_def.id == "test-model"
    assert model_def.display_name == "Test Model 7B"
    assert model_def.format == ModelFormat.GGUF
    assert model_def.relative_path == Path("models/gguf/test-model.gguf")
    assert model_def.supported_providers == ["llama_cpp"]
    assert model_def.aliases == ["test", "default"]
    assert model_def.roles == [ModelRole.GENERAL, ModelRole.CODING]
    assert model_def.capabilities.chat is True
    assert model_def.capabilities.code is True
    assert model_def.capabilities.context_window == 8192
    assert model_def.metadata["quantization"] == "Q4_K_M"
    assert model_def.metadata["parameter_count"] == "7B"


def test_load_model_missing_file():
    """Verify error when config file does not exist."""
    with pytest.raises(ConfigurationError, match="Model configuration file not found"):
        load_model_definition("/non/existent/path/model.toml")


def test_load_model_malformed_toml(tmp_path: Path):
    """Verify error when TOML syntax is invalid."""
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_bytes(b"[model\nincomplete section")

    with pytest.raises(ConfigurationError, match="Failed to parse TOML"):
        load_model_definition(bad_toml)


def test_load_model_missing_required_fields(tmp_path: Path):
    """Verify error when required fields are missing."""
    missing_fields_toml = tmp_path / "missing.toml"
    missing_fields_toml.write_bytes(b"""
    [model]
    id = "test-model"
    format = "gguf"
    # missing path and supported_providers
    """)

    with pytest.raises(ConfigurationError, match="Missing required field"):
        load_model_definition(missing_fields_toml)


def test_load_model_invalid_format(tmp_path: Path):
    """Verify error when format is invalid."""
    invalid_format_toml = tmp_path / "inv_format.toml"
    invalid_format_toml.write_bytes(b"""
    [model]
    id = "test-model"
    format = "unsupported_xyz"
    path = "models/test.bin"
    supported_providers = ["llama_cpp"]
    """)

    with pytest.raises(ConfigurationError, match="Unsupported model format"):
        load_model_definition(invalid_format_toml)


def test_load_model_invalid_role(tmp_path: Path):
    """Verify error when an unknown/invalid model role is specified."""
    invalid_role_toml = tmp_path / "inv_role.toml"
    invalid_role_toml.write_bytes(b"""
    [model]
    id = "test-model"
    format = "gguf"
    path = "models/test.gguf"
    supported_providers = ["llama_cpp"]
    roles = ["general", "invalid_role_xyz"]
    """)

    with pytest.raises(ConfigurationError, match="Unsupported model role 'invalid_role_xyz'"):
        load_model_definition(invalid_role_toml)


def test_load_model_empty_roles(tmp_path: Path):
    """Verify error when roles list is empty."""
    empty_role_toml = tmp_path / "empty_role.toml"
    empty_role_toml.write_bytes(b"""
    [model]
    id = "test-model"
    format = "gguf"
    path = "models/test.gguf"
    supported_providers = ["llama_cpp"]
    roles = []
    """)

    with pytest.raises(ConfigurationError, match="Field '\\[model\\]\\.roles' must be a non-empty list"):
        load_model_definition(empty_role_toml)


def test_load_model_definitions_from_dir(tmp_path: Path):
    """Verify loading multiple model files from a directory."""
    (tmp_path / "model1.toml").write_bytes(b"""
    [model]
    id = "model-1"
    format = "gguf"
    path = "models/1.gguf"
    supported_providers = ["llama_cpp"]
    """)

    (tmp_path / "model2.toml").write_bytes(b"""
    [model]
    id = "model-2"
    format = "safetensors"
    path = "models/2.safetensors"
    supported_providers = ["vllm"]
    """)

    defs = load_model_definitions_from_dir(tmp_path)
    assert len(defs) == 2
    assert "model-1" in defs
    assert "model-2" in defs
    assert defs["model-1"].format == ModelFormat.GGUF
    assert defs["model-2"].format == ModelFormat.SAFETENSORS


def test_load_model_definitions_duplicate_id(tmp_path: Path):
    """Verify error when duplicate model IDs exist in directory."""
    (tmp_path / "a.toml").write_bytes(b"""
    [model]
    id = "duplicate-id"
    format = "gguf"
    path = "models/a.gguf"
    supported_providers = ["llama_cpp"]
    """)
    (tmp_path / "b.toml").write_bytes(b"""
    [model]
    id = "duplicate-id"
    format = "gguf"
    path = "models/b.gguf"
    supported_providers = ["llama_cpp"]
    """)

    with pytest.raises(ConfigurationError, match="Duplicate model ID"):
        load_model_definitions_from_dir(tmp_path)


def test_load_settings(tmp_path: Path):
    """Verify settings.toml loading."""
    settings_file = tmp_path / "settings.toml"
    settings_file.write_bytes(b"""
    [foundation]
    environment = "test"
    models_dir = "custom_models"
    configs_dir = "custom_configs"

    [providers.llama_cpp]
    base_url = "http://127.0.0.1:9090"
    timeout_seconds = 45.0
    default_alias = "test-alias"
    """)

    settings = load_settings(settings_file)
    assert settings.foundation.environment == "test"
    assert settings.foundation.models_dir == "custom_models"
    assert settings.llama_cpp.base_url == "http://127.0.0.1:9090"
    assert settings.llama_cpp.timeout_seconds == 45.0
    assert settings.llama_cpp.default_alias == "test-alias"
