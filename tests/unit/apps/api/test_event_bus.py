"""Unit tests for OrchestrationEventBus (SSE pub/sub and safety filtering)."""

import asyncio
import json
import pytest

from apps.api.events import OrchestrationEventBus, _sanitize_event_data


def test_sanitize_event_data_strips_thoughts():
    """Verify recursive stripping of internal thought and thinking fields."""
    raw_payload = {
        "task_id": "task-1",
        "thought": "Internal LLM reasoning that must not be exposed",
        "thinking": "Deep chain of thought",
        "details": {
            "model": "qwen3.5-9b",
            "chain_of_thought": "Sensitive intermediate token",
            "tokens": 42,
        },
        "nested_list": [
            {"safe": True, "thought": "drop this"},
            {"safe": True},
        ],
    }

    clean = _sanitize_event_data(raw_payload)

    assert "thought" not in clean
    assert "thinking" not in clean
    assert "chain_of_thought" not in clean["details"]
    assert clean["details"]["tokens"] == 42
    assert "thought" not in clean["nested_list"][0]
    assert clean["nested_list"][0]["safe"] is True


def test_event_bus_suppresses_disallowed_event_type():
    """Verify that agent.thought and thought event types are dropped."""
    async def _run():
        bus = OrchestrationEventBus()
        goal_id = "goal-test-suppress"

        # Subscribe
        sub_iter = bus.subscribe(goal_id)
        # Consume stream.connected
        init_event = await anext(sub_iter)
        assert "stream.connected" in init_event

        # Publish disallowed event
        await bus.publish(
            goal_id=goal_id,
            event_type="agent.thought",
            data={"text": "This should never be sent"},
        )

        # Publish safe event
        await bus.publish(
            goal_id=goal_id,
            event_type="task.completed",
            data={"task_id": "task-1", "status": "completed"},
        )

        # Receive next event: should be task.completed, NOT agent.thought
        recv = await anext(sub_iter)
        assert "event: task.completed" in recv
        assert "agent.thought" not in recv

    asyncio.run(_run())


def test_event_bus_pub_sub_lifecycle():
    """Verify subscription, fanout delivery, and queue cleanup on disconnect."""
    async def _run():
        bus = OrchestrationEventBus()
        goal_id = "goal-test-lifecycle"

        sub = bus.subscribe(goal_id)
        init_event = await anext(sub)
        assert "stream.connected" in init_event

        # Publish plan.started
        await bus.publish(
            goal_id=goal_id,
            event_type="plan.started",
            data={"plan_id": "plan-1"},
        )
        msg1 = await anext(sub)
        assert "event: plan.started" in msg1
        assert '"plan_id": "plan-1"' in msg1

        # Publish terminal event (goal.completed)
        await bus.publish(
            goal_id=goal_id,
            event_type="goal.completed",
            data={"status": "completed"},
        )
        msg2 = await anext(sub)
        assert "event: goal.completed" in msg2

        # Generator terminates after terminal event
        with pytest.raises(StopAsyncIteration):
            await anext(sub)

    asyncio.run(_run())
