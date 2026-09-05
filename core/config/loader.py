"""TOML Configuration loader and validator using Python 3.12 tomllib."""

import tomllib
from pathlib import Path
from typing import Dict, List, Optional, Union

from core.common.errors import ConfigurationError
from core.common.types import ModelFormat, ModelRole
from core.config.settings import (
    DatabaseSettings,
    FoundationSettings,
    LlamaCppProviderSettings,
    Settings,
)
from core.models.schema import ModelCapabilities, ModelDefinition


def load_model_definition(toml_path: Union[str, Path]) -> ModelDefinition:
    """Load and validate a single model definition from a TOML file.
    
    Args:
        toml_path: Path to the .toml model definition file.
        
    Returns:
        Validated ModelDefinition instance.
        
    Raises:
        ConfigurationError: If the file is unreadable, invalid TOML, or fails schema validation.
    """
    path = Path(toml_path)
    if not path.is_file():
        raise ConfigurationError(f"Model configuration file not found: {path}")

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        raise ConfigurationError(f"Failed to parse TOML configuration from {path}: {e}") from e

    if "model" not in data or not isinstance(data["model"], dict):
        raise ConfigurationError(f"Missing required [model] section in {path}")

    m_section = data["model"]

    # Validate required model fields
    for req_field in ("id", "format", "path", "supported_providers"):
        if req_field not in m_section:
            raise ConfigurationError(f"Missing required field '[model].{req_field}' in {path}")

    model_id = str(m_section["id"]).strip()
    if not model_id:
        raise ConfigurationError(f"Field '[model].id' must not be empty in {path}")

    format_str = str(m_section["format"]).strip().lower()
    try:
        model_format = ModelFormat(format_str)
    except ValueError:
        raise ConfigurationError(
            f"Unsupported model format '{format_str}' in {path}. "
            f"Allowed values: {[f.value for f in ModelFormat]}"
        )

    display_name = str(m_section.get("display_name", model_id))
    relative_path = Path(m_section["path"])

    providers_raw = m_section["supported_providers"]
    if not isinstance(providers_raw, list) or not providers_raw:
        raise ConfigurationError(f"Field '[model].supported_providers' must be a non-empty list in {path}")
    supported_providers = [str(p) for p in providers_raw]

    aliases_raw = m_section.get("aliases", [])
    aliases = [str(a) for a in aliases_raw] if isinstance(aliases_raw, list) else []

    roles_raw = m_section.get("roles", ["general"])
    if not isinstance(roles_raw, list) or not roles_raw:
        raise ConfigurationError(f"Field '[model].roles' must be a non-empty list in {path}")

    roles: List[ModelRole] = []
    for r in roles_raw:
        role_str = str(r).strip().lower()
        try:
            roles.append(ModelRole(role_str))
        except ValueError:
            raise ConfigurationError(
                f"Unsupported model role '{role_str}' in {path}. "
                f"Allowed values: {[role.value for role in ModelRole]}"
            )

    # Capabilities parsing
    caps_section = data.get("capabilities", {})
    capabilities = ModelCapabilities(
        chat=bool(caps_section.get("chat", True)),
        code=bool(caps_section.get("code", False)),
        reasoning=bool(caps_section.get("reasoning", False)),
        structured_output=bool(caps_section.get("structured_output", False)),
        context_window=int(caps_section.get("context_window", 4096)),
        custom={k: v for k, v in caps_section.items() if k not in ("chat", "code", "reasoning", "structured_output", "context_window")},
    )

    # Metadata parsing (verified metadata only)
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    return ModelDefinition(
        id=model_id,
        display_name=display_name,
        format=model_format,
        relative_path=relative_path,
        supported_providers=supported_providers,
        aliases=aliases,
        roles=roles,
        capabilities=capabilities,
        metadata=metadata,
    )


def load_model_definitions_from_dir(configs_dir: Union[str, Path]) -> Dict[str, ModelDefinition]:
    """Scan and load all .toml model definitions in a directory.
    
    Args:
        configs_dir: Directory containing .toml configuration files.
        
    Returns:
        Dictionary mapping canonical model IDs to ModelDefinitions.
        
    Raises:
        ConfigurationError: If configs_dir is not a valid directory.
    """
    path = Path(configs_dir)
    if not path.is_dir():
        raise ConfigurationError(f"Model configurations directory not found: {path}")

    definitions: Dict[str, ModelDefinition] = {}
    for toml_file in sorted(path.glob("*.toml")):
        model_def = load_model_definition(toml_file)
        if model_def.id in definitions:
            raise ConfigurationError(
                f"Duplicate model ID '{model_def.id}' detected in {toml_file}. "
                f"Already defined by another file."
            )
        definitions[model_def.id] = model_def

    return definitions


def load_settings(settings_path: Union[str, Path]) -> Settings:
    """Load global settings from a settings.toml file.
    
    Args:
        settings_path: Path to settings.toml
        
    Returns:
        Settings instance.
    """
    path = Path(settings_path)
    if not path.is_file():
        # Fall back to default settings if file doesn't exist
        return Settings()

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        raise ConfigurationError(f"Failed to parse settings from {path}: {e}") from e

    found_data = data.get("foundation", {})
    llama_data = data.get("providers", {}).get("llama_cpp", {})

    foundation_settings = FoundationSettings(
        environment=str(found_data.get("environment", "development")),
        models_dir=str(found_data.get("models_dir", "models")),
        configs_dir=str(found_data.get("configs_dir", "configs/models")),
    )

    llama_settings = LlamaCppProviderSettings(
        base_url=str(llama_data.get("base_url", "http://127.0.0.1:8080")),
        timeout_seconds=float(llama_data.get("timeout_seconds", 60.0)),
        default_alias=str(llama_data.get("default_alias", "qwen3.5-9b")),
    )

    db_data = data.get("database", {})
    database_settings = DatabaseSettings(
        url=str(db_data.get("url", "postgresql+psycopg://postgres:postgres@localhost:5432/local_ai")),
        pool_size=int(db_data.get("pool_size", 5)),
        max_overflow=int(db_data.get("max_overflow", 10)),
        echo=bool(db_data.get("echo", False)),
    )

    return Settings(
        foundation=foundation_settings,
        llama_cpp=llama_settings,
        database=database_settings,
    )
