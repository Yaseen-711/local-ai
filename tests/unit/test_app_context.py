"""Unit tests for AppContext – Application Composition Root.

All tests run on pure CPU with zero GPU, network, or live server dependencies.
Follows the same pattern as test_foundation_factory.py: constructs a tmp_path
fixture environment and patches / mocks the provider-level infer() call to
avoid any real HTTP calls.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from apps import AppContext
from connectors import FoundationInferenceConnector, InferenceConnector
from core.common.types import FinishReason
from core.inference.types import InferenceResponse, Message, TokenUsage
from workflows import TextAnalysisWorkflow


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _make_tmp_env(tmp_path: Path) -> dict:
    """Create a minimal but valid FoundationCore config environment."""
    configs_dir = tmp_path / "configs" / "models"
    models_dir = tmp_path / "models" / "gguf"
    configs_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    (models_dir / "test-model.gguf").write_bytes(b"mock-weight-data")

    (configs_dir / "test-model.toml").write_bytes(b"""\
[model]
id = "test-model"
format = "gguf"
path = "models/gguf/test-model.gguf"
supported_providers = ["llama_cpp"]
""")

    settings_file = tmp_path / "settings.toml"
    settings_file.write_bytes(b"""\
[foundation]
environment = "unit-test"
models_dir = "models"
configs_dir = "configs/models"

[providers.llama_cpp]
base_url = "http://127.0.0.1:8080"
""")

    return {
        "repo_root": tmp_path,
        "configs_dir": configs_dir,
        "settings_path": settings_file,
    }


def _make_mock_response(model_id: str = "test-model") -> InferenceResponse:
    return InferenceResponse(
        request_id="req-ctx-test",
        model_id=model_id,
        message=Message.assistant("Mock model output."),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        latency_ms=8.0,
    )


# --------------------------------------------------------------------------- #
# AppContext.create()                                                           #
# --------------------------------------------------------------------------- #

def test_app_context_create_initialises_core_and_inference(tmp_path: Path):
    """AppContext.create() wires FoundationCore and FoundationInferenceConnector."""
    env = _make_tmp_env(tmp_path)

    ctx = AppContext.create(**env)

    assert isinstance(ctx.core, object)
    assert ctx.core is not None
    assert isinstance(ctx.inference, FoundationInferenceConnector)


def test_app_context_core_has_expected_model(tmp_path: Path):
    """AppContext.core registry is populated from the configs_dir."""
    env = _make_tmp_env(tmp_path)

    ctx = AppContext.create(**env)

    assert ctx.core.registry.is_known("test-model")


def test_app_context_inference_connector_is_bound_to_core(tmp_path: Path):
    """The inference connector inside AppContext is wired to the same core instance."""
    env = _make_tmp_env(tmp_path)

    ctx = AppContext.create(**env)

    # FoundationInferenceConnector stores _core; verify it's the same object
    assert ctx.inference._core is ctx.core


def test_app_context_is_frozen(tmp_path: Path):
    """AppContext is a frozen dataclass – attributes cannot be reassigned."""
    env = _make_tmp_env(tmp_path)
    ctx = AppContext.create(**env)

    with pytest.raises((AttributeError, TypeError)):
        ctx.core = None  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# AppContext with explicit constructor (custom connector injection)             #
# --------------------------------------------------------------------------- #

def test_app_context_accepts_custom_connector():
    """AppContext can be constructed directly with any InferenceConnector-compatible object."""
    mock_core = MagicMock()
    mock_connector = MagicMock()
    mock_connector.infer_prompt.return_value = _make_mock_response()

    ctx = AppContext(core=mock_core, inference=mock_connector)

    assert ctx.core is mock_core
    assert ctx.inference is mock_connector


def test_app_context_inference_conforms_to_protocol(tmp_path: Path):
    """FoundationInferenceConnector stored in AppContext satisfies InferenceConnector protocol."""
    env = _make_tmp_env(tmp_path)
    ctx = AppContext.create(**env)

    # Runtime structural check (InferenceConnector is @runtime_checkable)
    assert isinstance(ctx.inference, InferenceConnector)


# --------------------------------------------------------------------------- #
# create_text_analysis_workflow()                                               #
# --------------------------------------------------------------------------- #

def test_create_text_analysis_workflow_returns_correct_type(tmp_path: Path):
    """create_text_analysis_workflow() returns a TextAnalysisWorkflow instance."""
    env = _make_tmp_env(tmp_path)
    ctx = AppContext.create(**env)

    workflow = ctx.create_text_analysis_workflow()

    assert isinstance(workflow, TextAnalysisWorkflow)


def test_create_text_analysis_workflow_uses_context_connector(tmp_path: Path):
    """The workflow produced by AppContext is wired to the context's inference connector."""
    env = _make_tmp_env(tmp_path)
    ctx = AppContext.create(**env)

    workflow = ctx.create_text_analysis_workflow()

    # TextAnalysisWorkflow stores _inference
    assert workflow._inference is ctx.inference


def test_create_text_analysis_workflow_returns_new_instance_each_call(tmp_path: Path):
    """Each call to create_text_analysis_workflow() returns a distinct workflow instance."""
    env = _make_tmp_env(tmp_path)
    ctx = AppContext.create(**env)

    wf1 = ctx.create_text_analysis_workflow()
    wf2 = ctx.create_text_analysis_workflow()

    assert wf1 is not wf2
    # Both are wired to the same connector (same AppContext)
    assert wf1._inference is wf2._inference


def test_workflow_factory_with_mock_connector_executes_analysis():
    """Workflow from AppContext with mock connector executes real domain logic."""
    mock_core = MagicMock()
    mock_connector = MagicMock()
    mock_connector.infer_prompt.return_value = _make_mock_response()

    ctx = AppContext(core=mock_core, inference=mock_connector)
    workflow = ctx.create_text_analysis_workflow()

    result = workflow.analyze("Artificial intelligence is transforming the software industry.")

    assert result is not None
    assert result.output.summary is not None
    # The mock returned "Mock model output." which contains no bullet markers,
    # so key_points resilient fallback kicks in – just verify the workflow ran.
    mock_connector.infer_prompt.assert_called_once()


# --------------------------------------------------------------------------- #
# Multiple contexts are independent                                             #
# --------------------------------------------------------------------------- #

def test_two_app_contexts_are_independent(tmp_path: Path):
    """Two AppContext instances created from the same env are fully independent objects."""
    env = _make_tmp_env(tmp_path)

    ctx_a = AppContext.create(**env)
    ctx_b = AppContext.create(**env)

    # Different objects; not sharing state
    assert ctx_a is not ctx_b
    assert ctx_a.core is not ctx_b.core
    assert ctx_a.inference is not ctx_b.inference


def test_app_context_type_hints_resolvable():
    """Verify runtime type hints on AppContext methods resolve without NameError."""
    import typing
    from typing import Optional
    from orchestration import GoalOrchestrator, InProcessPlanRunner, PlanRunner
    from core import FoundationCore

    hints_orch = typing.get_type_hints(AppContext.create_goal_orchestrator)
    assert hints_orch["runner"] == Optional[PlanRunner]
    assert hints_orch["return"] is GoalOrchestrator

    hints_runner = typing.get_type_hints(AppContext.create_in_process_plan_runner)
    assert hints_runner["return"] is InProcessPlanRunner

    hints_cls = typing.get_type_hints(AppContext)
    assert hints_cls["core"] is FoundationCore
    assert hints_cls["inference"] is InferenceConnector


def test_app_context_create_vision_inspection_capability(tmp_path: Path):
    """create_vision_inspection_capability returns VisionInspectionCapability wired to context."""
    from orchestration.capabilities.builtin.vision import VisionInspectionCapability

    env = _make_tmp_env(tmp_path)
    ctx = AppContext.create(**env)

    cap = ctx.create_vision_inspection_capability()
    assert isinstance(cap, VisionInspectionCapability)
    assert cap._connector is ctx.inference


def test_app_context_registry_includes_vision(tmp_path: Path):
    """create_base_capability_registry registers vision.inspect."""
    env = _make_tmp_env(tmp_path)
    ctx = AppContext.create(**env)

    registry = ctx.create_base_capability_registry()
    assert registry.has("vision.inspect")
