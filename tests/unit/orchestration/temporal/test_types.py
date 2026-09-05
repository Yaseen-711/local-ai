"""Unit tests for Temporal DTO types."""

from orchestration.execution.temporal.types import (
    InputReferenceDTO,
    PlanWorkflowInput,
    PlanWorkflowOutput,
    TaskActivityInput,
    TaskActivityOutput,
    TaskDefinitionDTO,
)


def test_task_activity_input_defaults():
    """Verify TaskActivityInput fields and defaults."""
    act_input = TaskActivityInput(
        task_id="t1",
        capability_id="test.echo",
        attempt_id="att-1",
    )
    assert act_input.task_id == "t1"
    assert act_input.capability_id == "test.echo"
    assert act_input.attempt_id == "att-1"
    assert act_input.parameters == {}
    assert act_input.inputs == {}


def test_task_activity_output_success():
    """Verify TaskActivityOutput success fields."""
    output = TaskActivityOutput(
        task_id="t1",
        attempt_id="att-1",
        status="COMPLETED",
        output={"text": "hello"},
        references={"out": "ref-1"},
        artifacts=[{"uri": "file:///tmp/artifact.txt"}],
    )
    assert output.status == "COMPLETED"
    assert output.output == {"text": "hello"}
    assert output.error_message is None


def test_task_activity_output_failure():
    """Verify TaskActivityOutput failure fields."""
    output = TaskActivityOutput(
        task_id="t1",
        attempt_id="att-1",
        status="FAILED",
        error_message="Boom",
        error_category="EXECUTION",
        error_code="RUNTIME_ERROR",
        error_details={"line": 42},
    )
    assert output.status == "FAILED"
    assert output.error_message == "Boom"
    assert output.error_category == "EXECUTION"
    assert output.error_code == "RUNTIME_ERROR"
    assert output.error_details["line"] == 42


def test_plan_workflow_input_and_output():
    """Verify PlanWorkflowInput and PlanWorkflowOutput."""
    task_dto = TaskDefinitionDTO(
        task_id="t1",
        capability_id="test.echo",
        attempt_id="att-t1-1",
        parameters={"msg": "hi"},
        dependencies=[],
        input_references={"ref": InputReferenceDTO(key="val", source_task_id="t0")},
    )
    wf_input = PlanWorkflowInput(
        plan_id="p-1",
        goal_id="g-1",
        tasks=[task_dto],
    )
    assert wf_input.plan_id == "p-1"
    assert len(wf_input.tasks) == 1
    assert wf_input.tasks[0].task_id == "t1"
    assert wf_input.tasks[0].attempt_id == "att-t1-1"

    wf_output = PlanWorkflowOutput(
        plan_id="p-1",
        status="COMPLETED",
        task_results={"t1": "hi"},
        task_errors={},
        task_statuses={"t1": "COMPLETED"},
        task_attempts={"t1": "att-t1-1"},
    )
    assert wf_output.status == "COMPLETED"
    assert wf_output.task_results["t1"] == "hi"
    assert wf_output.task_attempts["t1"] == "att-t1-1"
