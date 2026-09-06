"""Unit tests for built-in capabilities."""

from typing import Optional
import pytest

from core.inference.types import (
    FinishReason,
    GenerationOptions,
    InferenceRequest,
    InferenceResponse,
    Message,
    TokenUsage,
)
from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.builtin.inference import InferencePromptCapability
from orchestration.capabilities.builtin.workflow import TextAnalysisCapability
from workflows.text_analysis import AnalysisDepth, AnalysisOptions, TextAnalysis, TextAnalysisWorkflow
from workflows.types import WorkflowResult


class FakeInferenceConnector:
    """Duck-typed connector recording calls and returning canned responses."""

    def __init__(self, response_text: str = "Test response") -> None:
        self.response_text = response_text
        self.last_prompt: Optional[str] = None
        self.last_model_id: Optional[str] = None
        self.last_system_prompt: Optional[str] = None
        self.last_options: Optional[GenerationOptions] = None

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        raise NotImplementedError

    def infer_prompt(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        options: Optional[GenerationOptions] = None,
        request_id: Optional[str] = None,
    ) -> InferenceResponse:
        self.last_model_id = model_id
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        self.last_options = options
        return InferenceResponse(
            request_id=request_id,
            model_id=model_id,
            message=Message.assistant(self.response_text),
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_ms=12.5,
            raw_response={},
        )


class FakeTextAnalysisWorkflow:
    """Duck-typed workflow recording calls and returning canned WorkflowResult."""

    def __init__(self) -> None:
        self.last_text: Optional[str] = None
        self.last_options: Optional[AnalysisOptions] = None

    def analyze(self, text: str, options: Optional[AnalysisOptions] = None) -> WorkflowResult[TextAnalysis]:
        self.last_text = text
        self.last_options = options
        depth = options.depth if options else AnalysisDepth.QUICK
        return WorkflowResult(
            output=TextAnalysis(
                summary="Sample summary",
                key_points=["Point 1", "Point 2"],
                word_count=len(text.split()),
                depth=depth,
                raw_output="Raw output",
            ),
            model_id="test-model",
            metadata={"latency_ms": 50.0},
        )


# ---------------------------------------------------------------------------
# InferencePromptCapability Tests
# ---------------------------------------------------------------------------

def test_inference_prompt_capability_success():
    """Verify inference capability execution and metadata extraction."""
    connector = FakeInferenceConnector("Generated answer")
    cap = InferencePromptCapability(connector)

    assert cap.capability_id == "inference.prompt"

    context = CapabilityContext(execution_id="exec-1")
    result = cap.execute(
        parameters={"model_id": "custom-model", "temperature": 0.5, "max_tokens": 256},
        inputs={"prompt": "What is the capital of France?"},
        context=context,
    )

    assert result.output == "Generated answer"
    assert result.metadata["model_id"] == "custom-model"
    assert result.metadata["prompt_tokens"] == 10
    assert result.metadata["completion_tokens"] == 5
    assert result.metadata["total_tokens"] == 15
    assert connector.last_prompt == "What is the capital of France?"
    assert connector.last_model_id == "custom-model"
    assert connector.last_options is not None
    assert connector.last_options.temperature == 0.5
    assert connector.last_options.max_tokens == 256


def test_inference_prompt_capability_prompt_in_parameters():
    """Verify prompt can be supplied in parameters if not in inputs."""
    connector = FakeInferenceConnector("Answer")
    cap = InferencePromptCapability(connector)

    result = cap.execute(
        parameters={"prompt": "Hello world"},
        inputs={},
        context=CapabilityContext(execution_id="e1"),
    )
    assert result.output == "Answer"
    assert connector.last_prompt == "Hello world"


def test_inference_prompt_capability_missing_prompt_raises():
    """Verify missing prompt raises ValueError."""
    connector = FakeInferenceConnector()
    cap = InferencePromptCapability(connector)

    with pytest.raises(ValueError, match="requires a non-empty string 'prompt'"):
        cap.execute({}, {}, CapabilityContext(execution_id="e1"))


def test_inference_prompt_capability_whitespace_prompt_raises():
    """Verify empty whitespace prompt raises ValueError."""
    connector = FakeInferenceConnector()
    cap = InferencePromptCapability(connector)

    with pytest.raises(ValueError, match="requires a non-empty string 'prompt'"):
        cap.execute({"prompt": "   \n\t  "}, {}, CapabilityContext(execution_id="e1"))


# ---------------------------------------------------------------------------
# TextAnalysisCapability Tests
# ---------------------------------------------------------------------------

def test_text_analysis_capability_success():
    """Verify text analysis capability execution and result mapping."""
    fake_workflow = FakeTextAnalysisWorkflow()
    cap = TextAnalysisCapability(fake_workflow)  # type: ignore[arg-type]

    assert cap.capability_id == "workflow.text_analysis"

    result = cap.execute(
        parameters={"depth": "detailed", "focus": "risks", "model_id": "analyst-model"},
        inputs={"text": "Quarterly earnings report showed high revenue but rising risks."},
        context=CapabilityContext(execution_id="exec-2"),
    )

    assert isinstance(result.output, TextAnalysis)
    assert result.output.summary == "Sample summary"
    assert result.output.key_points == ["Point 1", "Point 2"]
    assert result.metadata["latency_ms"] == 50.0
    assert fake_workflow.last_options is not None
    assert fake_workflow.last_options.depth == AnalysisDepth.DETAILED
    assert fake_workflow.last_options.focus == "risks"
    assert fake_workflow.last_options.model_id == "analyst-model"


def test_text_analysis_capability_missing_text_raises():
    """Verify missing text raises ValueError."""
    fake_workflow = FakeTextAnalysisWorkflow()
    cap = TextAnalysisCapability(fake_workflow)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="requires a non-empty string 'text'"):
        cap.execute({}, {}, CapabilityContext(execution_id="e2"))


def test_text_analysis_capability_invalid_depth_raises():
    """Verify invalid depth option raises ValueError."""
    fake_workflow = FakeTextAnalysisWorkflow()
    cap = TextAnalysisCapability(fake_workflow)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Invalid analysis depth 'invalid'"):
        cap.execute(
            parameters={"depth": "invalid"},
            inputs={"text": "Valid text"},
            context=CapabilityContext(execution_id="e2"),
        )
