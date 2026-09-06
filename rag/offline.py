"""Central offline-first and zero-internet runtime configuration for the RAG subsystem.

Enforces strict local-only execution:
- Application-level RAG_OFFLINE_MODE configuration (default: True).
- Pre-emptively sets HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, and HF_HUB_DISABLE_TELEMETRY=1.
- Tracks local model filesystem and Hugging Face cache locations.
- Provides strict fail-closed validation: if a model is not found locally, fails
  immediately with an informative error rather than attempting a network download.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Optional

ENV_RAG_OFFLINE_MODE = "RAG_OFFLINE_MODE"


class OfflineModelNotFoundError(RuntimeError):
    """Raised when a required model is missing locally in offline mode.

    Explicitly identifies the missing model, the requiring component, and the expected path.
    """

    def __init__(
        self,
        model_name: str,
        component: str,
        expected_location: str,
        details: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.component = component
        self.expected_location = expected_location
        self.details = details

        message = (
            f"Required model '{model_name}' is not available locally.\n"
            f"Component: {component}\n"
            f"Expected Location: {expected_location}\n"
            f"Offline mode is enabled (RAG_OFFLINE_MODE=true). Downloading models is disabled.\n"
            f"Install/provision the model before running this application."
        )
        if details:
            message += f"\nDetails: {details}"

        super().__init__(message)


def is_offline_mode() -> bool:
    """Return whether offline-only mode is active.

    Defaults to True (offline-first / zero-internet safe).
    Can be explicitly set via RAG_OFFLINE_MODE=false or 0.
    """
    val = os.environ.get(ENV_RAG_OFFLINE_MODE, "true").strip().lower()
    return val not in ("false", "0", "no", "off")


def ensure_offline_environment() -> None:
    """Configure environment variables to prevent network requests by third-party libraries.

    Sets:
    - HF_HUB_OFFLINE = 1
    - TRANSFORMERS_OFFLINE = 1
    - HF_HUB_DISABLE_TELEMETRY = 1
    - DOCLING_OFFLINE = 1
    - TOKENIZERS_PARALLELISM = false
    And synchronizes already-loaded huggingface_hub constants if present.
    """
    if not is_offline_mode():
        return

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DOCLING_OFFLINE"] = "1"
    if "TOKENIZERS_PARALLELISM" not in os.environ:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Synchronize huggingface_hub constant if module is already imported
    if "huggingface_hub.constants" in sys.modules:
        try:
            import huggingface_hub.constants as hf_constants

            hf_constants.HF_HUB_OFFLINE = True
            hf_constants.HF_HUB_DISABLE_TELEMETRY = True
        except Exception:
            pass


def get_hf_hub_cache_dir() -> Path:
    """Resolve the active Hugging Face hub cache directory."""
    custom_cache = os.environ.get("HF_HUB_CACHE")
    if custom_cache:
        return Path(custom_cache).expanduser().resolve()

    custom_home = os.environ.get("HF_HOME")
    if custom_home:
        return (Path(custom_home).expanduser() / "hub").resolve()

    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def get_expected_model_path(model_name: str) -> Path:
    """Return the expected filesystem or Hugging Face cache path for a model."""
    path_obj = Path(model_name)
    if path_obj.exists():
        return path_obj.resolve()

    hub_dir = get_hf_hub_cache_dir()
    repo_folder = "models--" + model_name.replace("/", "--")
    return hub_dir / repo_folder


def is_model_available_locally(model_name: str) -> bool:
    """Check if a model exists in the local filesystem or Hugging Face hub cache.

    Verifies that the target path exists and contains model snapshots or weight files.
    """
    expected_path = get_expected_model_path(model_name)
    if not expected_path.exists():
        return False

    if expected_path.is_file():
        return True

    # Check for direct model files (config.json, model.safetensors, etc.)
    if (expected_path / "config.json").exists() or (expected_path / "modules.json").exists():
        return True

    # Check Hugging Face hub snapshot structure
    snapshots_dir = expected_path / "snapshots"
    if snapshots_dir.exists() and snapshots_dir.is_dir():
        snapshots = [s for s in snapshots_dir.iterdir() if s.is_dir()]
        if snapshots:
            # Check that the snapshot contains files
            for snap in snapshots:
                if any(f.is_file() for f in snap.iterdir()):
                    return True

    return False
