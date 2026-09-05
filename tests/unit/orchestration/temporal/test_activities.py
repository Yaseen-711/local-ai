"""Unit tests for TaskExecutionActivity."""

from typing import Any, Dict
import pytest

from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.results import TaskResult
from orchestration.execution.temporal.activities import TaskExecutionActivity
from orchestration.execution.temporal.types import TaskActivityInput


class MockEchoCapability:
    @property
    def capability_id(self) -> str:
        return "test.echo"

    def execute(
        self,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        context: CapabilityContext,
    ) -> TaskResult:
        msg = inputs.get("msg") or parameters.get("msg", "")
        return TaskResult(output=f"echo:{msg}")


class MockCrashingCapability:
    @property
    def capability_id(self) -> str:
        return "test.crash"

    def execute(
        self,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        context: CapabilityContext,
    ) -> TaskResult:
        raise RuntimeError("Simulated crash in capability")


import asyncio


def test_activity_execute_success():
    """Verify successful capability execution via activity."""
    async def _test():
        registry = CapabilityRegistry()
        registry.register(MockEchoCapability())
        activity = TaskExecutionActivity(registry)

        act_input = TaskActivityInput(
            task_id="t1",
            capability_id="test.echo",
            attempt_id="att-1",
            parameters={"msg": "hello"},
        )
        output = await activity.execute_task(act_input)

        assert output.status == "COMPLETED"
        assert output.output == "echo:hello"
        assert output.task_id == "t1"
        assert output.attempt_id == "att-1"
        assert output.error_message is None

    asyncio.run(_test())


def test_activity_missing_capability():
    """Verify missing capability returns FAILED with CAPABILITY category."""
    async def _test():
        registry = CapabilityRegistry()
        activity = TaskExecutionActivity(registry)

        act_input = TaskActivityInput(
            task_id="t-unknown",
            capability_id="nonexistent.cap",
            attempt_id="att-2",
        )
        output = await activity.execute_task(act_input)

        assert output.status == "FAILED"
        assert output.error_category == "CAPABILITY"
        assert "nonexistent.cap" in output.error_message

    asyncio.run(_test())


def test_activity_capability_exception_handling():
    """Verify capability exceptions are caught and returned as FAILED facts."""
    async def _test():
        registry = CapabilityRegistry()
        registry.register(MockCrashingCapability())
        activity = TaskExecutionActivity(registry)

        act_input = TaskActivityInput(
            task_id="t-crash",
            capability_id="test.crash",
            attempt_id="att-3",
        )
        output = await activity.execute_task(act_input)

        assert output.status == "FAILED"
        assert output.error_category == "EXECUTION"
        assert "Simulated crash" in output.error_message
        assert output.error_code == "RuntimeError"

    asyncio.run(_test())


def test_activity_serializes_references_and_artifacts():
    """Verify TaskExecutionActivity serializes full DataReference and ArtifactReference metadata."""
    from orchestration.domain.references import ArtifactReference, DataReference

    class MockArtifactCapability:
        @property
        def capability_id(self) -> str:
            return "test.artifacts"

        def execute(self, parameters: Dict[str, Any], inputs: Dict[str, Any], context: CapabilityContext) -> TaskResult:
            return TaskResult(
                output="file-created",
                references=[DataReference(key="ref1", uri="uri://1", mime_type="text/plain", metadata={"a": 1})],
                artifacts=[ArtifactReference(artifact_id="art-1", name="doc.pdf", uri="uri://art", mime_type="application/pdf", size_bytes=500, metadata={"b": 2})],
            )

    async def _test():
        registry = CapabilityRegistry()
        registry.register(MockArtifactCapability())
        activity = TaskExecutionActivity(registry)

        act_input = TaskActivityInput(
            task_id="t-art",
            capability_id="test.artifacts",
            attempt_id="att-art",
        )
        output = await activity.execute_task(act_input)

        assert output.status == "COMPLETED"
        assert len(output.references) == 1
        assert output.references[0] == {
            "key": "ref1",
            "source_task_id": None,
            "uri": "uri://1",
            "mime_type": "text/plain",
            "metadata": {"a": 1},
        }
        assert len(output.artifacts) == 1
        assert output.artifacts[0] == {
            "artifact_id": "art-1",
            "name": "doc.pdf",
            "uri": "uri://art",
            "mime_type": "application/pdf",
            "size_bytes": 500,
            "metadata": {"b": 2},
        }

    asyncio.run(_test())
