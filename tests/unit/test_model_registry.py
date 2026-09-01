"""Unit tests for ModelRegistry behavior, availability, and lifecycle."""

from pathlib import Path
import pytest

from core.common.errors import ModelNotFoundError, ModelRegistryError
from core.common.types import ModelFormat
from core.models.registry import ModelRegistry


@pytest.fixture
def registry_setup(tmp_path: Path):
    """Creates a temporary workspace with mock configs and mock model files."""
    repo_root = tmp_path / "repo"
    configs_dir = repo_root / "configs" / "models"
    models_dir = repo_root / "models" / "gguf"

    configs_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    # Create mock model weight files (dummy small files)
    model1_file = models_dir / "model1.gguf"
    model1_file.write_bytes(b"dummy model data 1")

    # Config for model 1 (present on disk)
    (configs_dir / "model1.toml").write_bytes(b"""
    [model]
    id = "model-1"
    display_name = "Model One"
    format = "gguf"
    path = "models/gguf/model1.gguf"
    aliases = ["m1", "first-model"]
    supported_providers = ["llama_cpp"]
    """)

    # Config for model 2 (file does NOT exist on disk)
    (configs_dir / "model2.toml").write_bytes(b"""
    [model]
    id = "model-2"
    display_name = "Model Two (Missing)"
    format = "gguf"
    path = "models/gguf/model2_missing.gguf"
    aliases = ["m2"]
    supported_providers = ["llama_cpp"]
    """)

    return {
        "repo_root": repo_root,
        "configs_dir": configs_dir,
        "models_dir": models_dir,
        "model1_file": model1_file,
    }


def test_registry_initialization(registry_setup):
    """Verify registry loads configs and checks advisory availability on init."""
    reg = ModelRegistry(
        configs_dir=registry_setup["configs_dir"],
        repo_root=registry_setup["repo_root"],
    )

    models = reg.list_models()
    assert len(models) == 2
    assert reg.is_known("model-1")
    assert reg.is_known("model-2")
    assert reg.is_known("m1")
    assert reg.is_known("m2")
    assert not reg.is_known("non-existent")


def test_registry_lookup_and_aliases(registry_setup):
    """Verify lookup by canonical ID and alias."""
    reg = ModelRegistry(
        configs_dir=registry_setup["configs_dir"],
        repo_root=registry_setup["repo_root"],
    )

    by_id = reg.get_model("model-1")
    by_alias1 = reg.get_model("m1")
    by_alias2 = reg.get_model("first-model")

    assert by_id.id == "model-1"
    assert by_alias1.id == "model-1"
    assert by_alias2.id == "model-1"
    assert by_id.display_name == "Model One"


def test_registry_unknown_model(registry_setup):
    """Verify ModelNotFoundError on unknown identifier."""
    reg = ModelRegistry(
        configs_dir=registry_setup["configs_dir"],
        repo_root=registry_setup["repo_root"],
    )

    with pytest.raises(ModelNotFoundError, match="is not configured in the registry"):
        reg.get_model("unknown-model-xyz")

    with pytest.raises(ModelNotFoundError):
        reg.get_availability("unknown-model-xyz")


def test_registry_advisory_availability(registry_setup):
    """Verify advisory availability distinction between existing and missing files."""
    reg = ModelRegistry(
        configs_dir=registry_setup["configs_dir"],
        repo_root=registry_setup["repo_root"],
    )

    # model-1 file exists
    avail1 = reg.get_availability("model-1")
    assert avail1.is_available is True
    assert avail1.resolved_path == registry_setup["model1_file"]
    assert avail1.size_bytes == len(b"dummy model data 1")
    assert avail1.error_message is None

    # model-2 file is missing
    avail2 = reg.get_availability("model-2")
    assert avail2.is_available is False
    assert avail2.size_bytes is None
    assert "does not exist" in (avail2.error_message or "")

    # Available models list
    avail_models = reg.list_available_models()
    assert len(avail_models) == 1
    assert avail_models[0].id == "model-1"


def test_registry_refresh_behavior(registry_setup):
    """Verify explicit refresh updates availability when files change on disk."""
    reg = ModelRegistry(
        configs_dir=registry_setup["configs_dir"],
        repo_root=registry_setup["repo_root"],
    )

    assert reg.get_availability("model-1").is_available is True

    # Simulate model-1 being deleted
    registry_setup["model1_file"].unlink()

    # Cached availability still reports last known state before refresh
    assert reg.get_availability("model-1").is_available is True

    # Explicit refresh updates state
    reg.refresh()
    assert reg.get_availability("model-1").is_available is False
    assert len(reg.list_available_models()) == 0


def test_registry_alias_collision(tmp_path: Path):
    """Verify collision error when two models claim the same alias."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()

    (configs_dir / "mod_a.toml").write_bytes(b"""
    [model]
    id = "model-a"
    format = "gguf"
    path = "models/a.gguf"
    aliases = ["common-alias"]
    supported_providers = ["llama_cpp"]
    """)

    (configs_dir / "mod_b.toml").write_bytes(b"""
    [model]
    id = "model-b"
    format = "gguf"
    path = "models/b.gguf"
    aliases = ["common-alias"]
    supported_providers = ["llama_cpp"]
    """)

    with pytest.raises(ModelRegistryError, match="Alias collision"):
        ModelRegistry(configs_dir=configs_dir, repo_root=tmp_path)
