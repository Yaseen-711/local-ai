"""Unit tests for TextAnalysisWorkflow."""

from unittest.mock import MagicMock
import pytest

from connectors import InferenceConnector
from core.common.errors import (
    InferenceError,
    ModelNotFoundError,
    ProviderResponseError,
    ProviderUnavailableError,
    WorkflowError,
)

from core.common.types import FinishReason, MessageRole
from core.inference.types import (
    GenerationOptions,
    InferenceRequest,
    InferenceResponse,
    Message,
    OutputConstraint,
    TokenUsage,
)
from workflows import (

    AnalysisDepth,
    AnalysisOptions,
    TextAnalysis,
    TextAnalysisWorkflow,
    WorkflowResult,
)


def _make_mock_response(
    text: str,
    model_id: str = "qwen3.5-9b",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    latency_ms: float = 15.0,
) -> InferenceResponse:
    return InferenceResponse(
        request_id="req-test-1",
        model_id=model_id,
        message=Message(role=MessageRole.ASSISTANT, content=text),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Instantiation and Validation Tests
# ---------------------------------------------------------------------------

def test_workflow_instantiation():
    """Verify workflow accepts InferenceConnector via constructor injection."""
    mock_connector = MagicMock(spec=InferenceConnector)
    workflow = TextAnalysisWorkflow(inference=mock_connector)
    assert workflow._inference is mock_connector


def test_analyze_empty_text_raises_value_error():
    """Verify input validation rejects empty or whitespace-only text."""
    mock_connector = MagicMock()
    workflow = TextAnalysisWorkflow(inference=mock_connector)

    with pytest.raises(ValueError, match="Input text must not be empty"):
        workflow.analyze("")

    with pytest.raises(ValueError, match="Input text must not be empty"):
        workflow.analyze("   \n\t  ")

    mock_connector.infer_prompt.assert_not_called()


# ---------------------------------------------------------------------------
# QUICK (Single-Pass) Execution Tests
# ---------------------------------------------------------------------------

def test_quick_analysis_execution():
    """Verify single-pass QUICK analysis makes one call and parses output."""
    mock_connector = MagicMock()
    mock_text = (
        "This is an executive summary of the document.\n\n"
        "- First key finding\n"
        "- Second key finding\n"
        "- Third key finding"
    )
    mock_connector.infer_prompt.return_value = _make_mock_response(
        text=mock_text,
        model_id="qwen3.5-9b",
        prompt_tokens=15,
        completion_tokens=25,
        latency_ms=12.5,
    )

    workflow = TextAnalysisWorkflow(inference=mock_connector)
    input_text = "Alice and Bob launched a distributed caching layer across three cloud regions."

    result = workflow.analyze(input_text)

    # Verify return envelope and typing
    assert isinstance(result, WorkflowResult)
    assert isinstance(result.output, TextAnalysis)
    assert result.model_id == "qwen3.5-9b"

    # Verify domain fields
    analysis = result.output
    assert analysis.summary == "This is an executive summary of the document."
    assert analysis.key_points == [
        "First key finding",
        "Second key finding",
        "Third key finding",
    ]
    assert analysis.word_count == 12
    assert analysis.depth == AnalysisDepth.QUICK
    assert analysis.raw_output == mock_text

    # Verify exact metric aggregation
    assert result.metadata["steps_executed"] == 1
    assert result.metadata["prompt_tokens"] == 15
    assert result.metadata["completion_tokens"] == 25
    assert result.metadata["total_tokens"] == 40
    assert result.metadata["total_inference_latency_ms"] == 12.5

    # Verify single call made with default options
    mock_connector.infer_prompt.assert_called_once()
    kwargs = mock_connector.infer_prompt.call_args.kwargs
    assert kwargs["model_id"] == "default"
    assert "Alice and Bob launched" in kwargs["prompt"]
    assert kwargs["options"].temperature == 0.2
    assert kwargs["options"].max_tokens == 1024
    assert kwargs["options"].constraint == OutputConstraint.json()



def test_quick_analysis_with_focus_and_custom_options():
    """Verify focus instructions and custom generation options are passed correctly."""
    mock_connector = MagicMock()
    mock_connector.infer_prompt.return_value = _make_mock_response("Summary text")

    workflow = TextAnalysisWorkflow(inference=mock_connector)
    opts = AnalysisOptions(
        depth=AnalysisDepth.QUICK,
        focus="security vulnerabilities",
        model_id="qwen-coding",
        temperature=0.0,
        max_tokens=512,
    )

    result = workflow.analyze("System authentication handles password hashes using bcrypt.", options=opts)

    mock_connector.infer_prompt.assert_called_once()
    kwargs = mock_connector.infer_prompt.call_args.kwargs
    assert kwargs["model_id"] == "qwen-coding"
    assert "Focus specifically on: security vulnerabilities" in kwargs["prompt"]
    assert kwargs["options"].temperature == 0.0
    assert kwargs["options"].max_tokens == 512


# ---------------------------------------------------------------------------
# DETAILED (Two-Pass) Execution Tests
# ---------------------------------------------------------------------------

def test_detailed_analysis_two_calls_sequence():
    """Verify DETAILED analysis executes extraction pass followed by synthesis pass."""
    mock_connector = MagicMock()
    step1_output = "- Point Alpha: 40% latency reduction\n- Point Beta: memory usage halved"
    step2_output = "Comprehensive synthesis: The refactor yielded major performance gains."

    mock_connector.infer_prompt.side_effect = [
        _make_mock_response(step1_output, prompt_tokens=10, completion_tokens=15, latency_ms=10.0),
        _make_mock_response(step2_output, prompt_tokens=25, completion_tokens=20, latency_ms=18.0),
    ]

    workflow = TextAnalysisWorkflow(inference=mock_connector)
    opts = AnalysisOptions(depth=AnalysisDepth.DETAILED)
    input_text = "We restructured database indexes and observed a 40% latency reduction with memory usage halved."

    result = workflow.analyze(input_text, options=opts)

    # Verify 2 calls executed
    assert mock_connector.infer_prompt.call_count == 2
    calls = mock_connector.infer_prompt.call_args_list

    # Step 1: Extraction
    call1_prompt = calls[0].kwargs["prompt"]
    assert "Extract the critical facts" in call1_prompt
    assert input_text in call1_prompt

    # Step 2: Synthesis contains findings from Step 1 and original text
    call2_prompt = calls[1].kwargs["prompt"]
    assert "synthesize a comprehensive executive summary" in call2_prompt
    assert "Point Alpha: 40% latency reduction" in call2_prompt
    assert input_text in call2_prompt

    # Verify output domain model
    analysis = result.output
    assert analysis.depth == AnalysisDepth.DETAILED
    assert analysis.summary == "Comprehensive synthesis: The refactor yielded major performance gains."
    assert "Point Alpha: 40% latency reduction" in analysis.key_points
    assert "Point Beta: memory usage halved" in analysis.key_points
    assert "=== Phase 1: Extraction ===" in analysis.raw_output
    assert "=== Phase 2: Synthesis ===" in analysis.raw_output


def test_detailed_analysis_metric_aggregation():
    """Verify exact token usage and inference latency aggregation across sequential passes."""
    mock_connector = MagicMock()
    mock_connector.infer_prompt.side_effect = [
        _make_mock_response("Point 1", prompt_tokens=12, completion_tokens=8, latency_ms=14.0),
        _make_mock_response("Summary", prompt_tokens=30, completion_tokens=15, latency_ms=22.5),
    ]

    workflow = TextAnalysisWorkflow(inference=mock_connector)
    result = workflow.analyze("Input text for aggregation test", options=AnalysisOptions(depth=AnalysisDepth.DETAILED))

    meta = result.metadata
    assert meta["steps_executed"] == 2
    assert meta["prompt_tokens"] == 12 + 30
    assert meta["completion_tokens"] == 8 + 15
    assert meta["total_tokens"] == (12 + 8) + (30 + 15)
    assert meta["total_inference_latency_ms"] == 14.0 + 22.5
    assert meta["phase_inference_latencies_ms"] == {"extraction": 14.0, "synthesis": 22.5}


# ---------------------------------------------------------------------------
# Resilient Parsing Tests
# ---------------------------------------------------------------------------

def test_bullet_parsing_various_styles():
    """Verify resilient parsing handles dashes, asterisks, numbers, and bullets."""
    mock_connector = MagicMock()
    varied_bullets = (
        "Overview paragraph before bullets.\n\n"
        "- Dash bullet point\n"
        "* Asterisk bullet point\n"
        "1. Numbered bullet point\n"
        "2) Numbered parenthesis bullet\n"
        "• Unicode bullet point\n\n"
        "Concluding summary remarks."
    )
    mock_connector.infer_prompt.return_value = _make_mock_response(varied_bullets)

    workflow = TextAnalysisWorkflow(inference=mock_connector)
    result = workflow.analyze("Some text content")

    analysis = result.output
    assert len(analysis.key_points) == 5
    assert analysis.key_points[0] == "Dash bullet point"
    assert analysis.key_points[1] == "Asterisk bullet point"
    assert analysis.key_points[2] == "Numbered bullet point"
    assert analysis.key_points[3] == "Numbered parenthesis bullet"
    assert analysis.key_points[4] == "Unicode bullet point"
    assert "Overview paragraph before bullets." in analysis.summary
    assert "Concluding summary remarks." in analysis.summary


def test_unbulleted_output_resilient_fallback():
    """Verify graceful handling when model returns only prose paragraphs without bullet points."""
    mock_connector = MagicMock()
    prose = "The report concludes that cloud adoption continues to accelerate across European financial institutions."
    mock_connector.infer_prompt.return_value = _make_mock_response(prose)

    workflow = TextAnalysisWorkflow(inference=mock_connector)
    result = workflow.analyze("Financial report excerpt")

    analysis = result.output
    assert analysis.summary == prose
    assert analysis.key_points == []


# ---------------------------------------------------------------------------
# Error Wrapping & Chaining Tests
# ---------------------------------------------------------------------------

def test_infrastructure_error_wrapped_in_quick_analysis():
    """Verify ProviderUnavailableError during quick analysis is wrapped in WorkflowError with cause."""
    mock_connector = MagicMock()
    infra_err = ProviderUnavailableError("Connection refused to llama-server")
    mock_connector.infer_prompt.side_effect = infra_err

    workflow = TextAnalysisWorkflow(inference=mock_connector)

    with pytest.raises(WorkflowError, match="Text analysis failed during analysis phase") as exc_info:
        workflow.analyze("Some text")

    assert exc_info.value.__cause__ is infra_err


def test_infrastructure_error_wrapped_in_detailed_extraction_phase():
    """Verify failure in phase 1 of detailed analysis identifies the extraction phase."""
    mock_connector = MagicMock()
    infra_err = ModelNotFoundError("Model 'custom-model' not found")
    mock_connector.infer_prompt.side_effect = infra_err

    workflow = TextAnalysisWorkflow(inference=mock_connector)

    with pytest.raises(WorkflowError, match="Text analysis failed during extraction phase") as exc_info:
        workflow.analyze("Some text", options=AnalysisOptions(depth=AnalysisDepth.DETAILED))

    assert exc_info.value.__cause__ is infra_err


def test_infrastructure_error_wrapped_in_detailed_synthesis_phase():
    """Verify failure in phase 2 of detailed analysis identifies the synthesis phase."""
    mock_connector = MagicMock()
    infra_err = InferenceError("CUDA out of memory during generation")
    # Step 1 succeeds, Step 2 fails
    mock_connector.infer_prompt.side_effect = [
        _make_mock_response("- Extracted point"),
        infra_err,
    ]

    workflow = TextAnalysisWorkflow(inference=mock_connector)

    with pytest.raises(WorkflowError, match="Text analysis failed during synthesis phase") as exc_info:
        workflow.analyze("Some text", options=AnalysisOptions(depth=AnalysisDepth.DETAILED))

    assert exc_info.value.__cause__ is infra_err


def test_unrelated_exception_not_swallowed():
    """Verify non-infrastructure exceptions (e.g. unexpected bug) propagate unmodified."""
    mock_connector = MagicMock()
    mock_connector.infer_prompt.side_effect = TypeError("Unexpected argument error")

    workflow = TextAnalysisWorkflow(inference=mock_connector)

    with pytest.raises(TypeError, match="Unexpected argument error"):
        workflow.analyze("Some text")


def test_provider_response_error_wrapped_in_quick_analysis():
    """Verify ProviderResponseError (e.g. malformed response) is wrapped in WorkflowError with cause."""
    mock_connector = MagicMock()
    provider_err = ProviderResponseError("Server returned invalid non-JSON payload")
    mock_connector.infer_prompt.side_effect = provider_err

    workflow = TextAnalysisWorkflow(inference=mock_connector)

    with pytest.raises(WorkflowError, match="Text analysis failed during analysis phase") as exc_info:
        workflow.analyze("Source text")

    assert exc_info.value.__cause__ is provider_err


def test_provider_response_error_wrapped_in_detailed_phases():
    """Verify ProviderResponseError in both phases of detailed analysis wraps with phase context."""
    mock_connector = MagicMock()
    provider_err = ProviderResponseError("Invalid choices format")

    # Phase 1 failure
    mock_connector.infer_prompt.side_effect = provider_err
    workflow = TextAnalysisWorkflow(inference=mock_connector)

    with pytest.raises(WorkflowError, match="Text analysis failed during extraction phase") as exc_info:
        workflow.analyze("Source text", options=AnalysisOptions(depth=AnalysisDepth.DETAILED))
    assert exc_info.value.__cause__ is provider_err

    # Phase 2 failure
    mock_connector.infer_prompt.side_effect = [
        _make_mock_response("- Extracted fact"),
        provider_err,
    ]
    with pytest.raises(WorkflowError, match="Text analysis failed during synthesis phase") as exc_info2:
        workflow.analyze("Source text", options=AnalysisOptions(depth=AnalysisDepth.DETAILED))
    assert exc_info2.value.__cause__ is provider_err


def test_prompt_construction_delimiters_and_system_prompt():
    """Verify prompt structural delimiters and system_prompt instruction boundary."""
    mock_connector = MagicMock()
    mock_connector.infer_prompt.return_value = _make_mock_response(
        '{"summary": "Safe summary", "key_points": ["Safe point"]}'
    )

    workflow = TextAnalysisWorkflow(inference=mock_connector)

    # 1. Quick mode
    workflow.analyze("Sample input text", options=AnalysisOptions(depth=AnalysisDepth.QUICK))
    quick_call = mock_connector.infer_prompt.call_args
    assert quick_call.kwargs["system_prompt"] is not None
    assert "untrusted data" in quick_call.kwargs["system_prompt"]
    assert "[SOURCE TEXT TO ANALYZE]" in quick_call.kwargs["prompt"]
    assert "[/SOURCE TEXT TO ANALYZE]" in quick_call.kwargs["prompt"]

    # 2. Detailed mode
    mock_connector.reset_mock()
    mock_connector.infer_prompt.side_effect = [
        _make_mock_response('{"key_points": ["Fact 1"]}'),
        _make_mock_response('{"summary": "Synthesized summary"}'),
    ]
    workflow.analyze("Detailed input text", options=AnalysisOptions(depth=AnalysisDepth.DETAILED))
    assert mock_connector.infer_prompt.call_count == 2
    phase1_call = mock_connector.infer_prompt.call_args_list[0]
    phase2_call = mock_connector.infer_prompt.call_args_list[1]

    assert "[SOURCE TEXT TO ANALYZE]" in phase1_call.kwargs["prompt"]
    assert "[EXTRACTED FINDINGS (UNTRUSTED DATA)]" in phase2_call.kwargs["prompt"]
    assert "[ORIGINAL SOURCE TEXT (UNTRUSTED DATA)]" in phase2_call.kwargs["prompt"]
    assert "untrusted data" in phase2_call.kwargs["system_prompt"]



# ---------------------------------------------------------------------------
# Duck-Typing and Reusability Tests
# ---------------------------------------------------------------------------

def test_duck_typed_fake_connector_compatibility():
    """Verify workflow runs seamlessly against any object satisfying InferenceConnector."""
    class PureFakeConnector:
        """Custom fake connector with no inheritance from any project class."""
        def __init__(self):
            self.prompts_received = []

        def infer(self, request: InferenceRequest) -> InferenceResponse:
            raise NotImplementedError

        def infer_prompt(
            self,
            model_id: str,
            prompt: str,
            system_prompt=None,
            options=None,
            request_id=None,
        ) -> InferenceResponse:
            self.prompts_received.append(prompt)
            return _make_mock_response(text="Summary: Success.\n- Point 1")

    fake = PureFakeConnector()
    assert isinstance(fake, InferenceConnector)

    workflow = TextAnalysisWorkflow(inference=fake)
    result = workflow.analyze("Testing duck-typed connector compatibility.")

    assert result.output.summary == "Summary: Success."
    assert result.output.key_points == ["Point 1"]
    assert len(fake.prompts_received) == 1


def test_workflow_reusability_across_multiple_calls():
    """Verify that a single workflow instance can be safely called multiple times sequentially."""
    mock_connector = MagicMock()
    mock_connector.infer_prompt.side_effect = [
        _make_mock_response("First run\n- Key A"),
        _make_mock_response("Second run\n- Key B"),
    ]

    workflow = TextAnalysisWorkflow(inference=mock_connector)

    res1 = workflow.analyze("Text one")
    res2 = workflow.analyze("Text two")

    assert res1.output.summary == "First run"
    assert res1.output.key_points == ["Key A"]
    assert res2.output.summary == "Second run"
    assert res2.output.key_points == ["Key B"]
    assert mock_connector.infer_prompt.call_count == 2


def test_quick_analysis_with_structured_json_response():
    """Verify single-pass QUICK analysis parses clean JSON output."""
    mock_connector = MagicMock()
    json_output = '{"summary": "Structured executive summary.", "key_points": ["Insight 1", "Insight 2"]}'
    mock_connector.infer_prompt.return_value = _make_mock_response(text=json_output)

    workflow = TextAnalysisWorkflow(inference=mock_connector)
    result = workflow.analyze("Input text for structured analysis.")

    assert result.output.summary == "Structured executive summary."
    assert result.output.key_points == ["Insight 1", "Insight 2"]
    assert result.output.depth == AnalysisDepth.QUICK
    assert result.output.raw_output == json_output

    kwargs = mock_connector.infer_prompt.call_args.kwargs
    assert kwargs["options"].constraint == OutputConstraint.json()


def test_quick_analysis_with_fenced_json_response():
    """Verify single-pass QUICK analysis handles markdown-fenced JSON output."""
    mock_connector = MagicMock()
    fenced_output = """```json
{
  "summary": "Fenced summary output.",
  "key_points": ["Key A", "Key B"]
}
```"""
    mock_connector.infer_prompt.return_value = _make_mock_response(text=fenced_output)

    workflow = TextAnalysisWorkflow(inference=mock_connector)
    result = workflow.analyze("Input text for fenced analysis.")

    assert result.output.summary == "Fenced summary output."
    assert result.output.key_points == ["Key A", "Key B"]


def test_detailed_analysis_with_structured_json_response():
    """Verify two-pass DETAILED analysis coordinates structured JSON extraction and synthesis."""
    mock_connector = MagicMock()
    phase1_json = '{"key_points": ["Finding Alpha", "Finding Beta"]}'
    phase2_json = '{"summary": "Synthesized executive summary from findings."}'

    mock_connector.infer_prompt.side_effect = [
        _make_mock_response(phase1_json, prompt_tokens=10, completion_tokens=15, latency_ms=12.0),
        _make_mock_response(phase2_json, prompt_tokens=20, completion_tokens=25, latency_ms=18.0),
    ]

    workflow = TextAnalysisWorkflow(inference=mock_connector)
    result = workflow.analyze("Detailed document input.", options=AnalysisOptions(depth=AnalysisDepth.DETAILED))

    assert result.output.depth == AnalysisDepth.DETAILED
    assert result.output.summary == "Synthesized executive summary from findings."
    assert result.output.key_points == ["Finding Alpha", "Finding Beta"]
    assert result.metadata["steps_executed"] == 2
    assert result.metadata["total_tokens"] == (10 + 15) + (20 + 25)

    # Verify both calls requested OutputConstraint.json()
    calls = mock_connector.infer_prompt.call_args_list
    assert calls[0].kwargs["options"].constraint == OutputConstraint.json()
    assert calls[1].kwargs["options"].constraint == OutputConstraint.json()


def test_structured_output_non_string_summary_raises_workflow_error():
    """Verify that structured JSON with non-string summary raises WorkflowError without str coercion."""
    mock_connector = MagicMock()
    mock_connector.infer_prompt.return_value = _make_mock_response('{"summary": 12345, "key_points": ["Point"]}')
    workflow = TextAnalysisWorkflow(inference=mock_connector)

    with pytest.raises(WorkflowError, match="'summary' must be a string"):
        workflow.analyze("Some text")


def test_structured_output_non_list_key_points_raises_workflow_error():
    """Verify that structured JSON with non-list key_points raises WorkflowError."""
    mock_connector = MagicMock()
    mock_connector.infer_prompt.return_value = _make_mock_response('{"summary": "Valid summary", "key_points": "not a list"}')
    workflow = TextAnalysisWorkflow(inference=mock_connector)

    with pytest.raises(WorkflowError, match="'key_points' must be a list of strings"):
        workflow.analyze("Some text")


def test_structured_output_non_string_key_point_item_raises_workflow_error():
    """Verify that structured JSON with non-string elements in key_points raises WorkflowError."""
    mock_connector = MagicMock()
    mock_connector.infer_prompt.return_value = _make_mock_response('{"summary": "Valid summary", "key_points": ["Valid point", 42]}')
    workflow = TextAnalysisWorkflow(inference=mock_connector)

    with pytest.raises(WorkflowError, match="must be a string"):
        workflow.analyze("Some text")


def test_structured_output_missing_required_keys_raises_workflow_error():
    """Verify that structured JSON missing summary and key_points raises WorkflowError."""
    mock_connector = MagicMock()
    mock_connector.infer_prompt.return_value = _make_mock_response('{"irrelevant_key": "some value"}')
    workflow = TextAnalysisWorkflow(inference=mock_connector)

    with pytest.raises(WorkflowError, match="missing required 'summary' or 'key_points'"):
        workflow.analyze("Some text")


