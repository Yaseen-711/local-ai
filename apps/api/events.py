"""In-memory Pub/Sub event bus for Server-Sent Events (SSE) streaming.

Explicitly single-process and in-memory.
Streams safe execution facts (goal/task state changes, model selection, tool calls,
artifacts, terminal status) without exposing internal chain-of-thought.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional, Set

logger = logging.getLogger(__name__)

# Strictly disallowed event types and keys to prevent raw thought leakage
DISALLOWED_EVENT_TYPES = frozenset({"agent.thought", "thought", "chain_of_thought"})
DISALLOWED_DATA_KEYS = frozenset({"thought", "thinking", "chain_of_thought"})


def _sanitize_event_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively strip any raw thinking/chain-of-thought fields."""
    clean: Dict[str, Any] = {}
    for k, v in data.items():
        if k.lower() in DISALLOWED_DATA_KEYS:
            continue
        if isinstance(v, dict):
            clean[k] = _sanitize_event_data(v)
        elif isinstance(v, list):
            clean[k] = [
                _sanitize_event_data(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            clean[k] = v
    return clean


class OrchestrationEventBus:
    """In-memory event bus managing SSE subscriptions per goal."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()
        self._seq_counter = 0

    async def publish(
        self,
        goal_id: str,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Publish a lifecycle event to all active subscribers for a goal."""
        if event_type.lower() in DISALLOWED_EVENT_TYPES:
            logger.debug("Suppressed disallowed event type: %s", event_type)
            return

        sanitized_data = _sanitize_event_data(data)

        async with self._lock:
            self._seq_counter += 1
            seq = self._seq_counter
            queues = list(self._subscribers.get(goal_id, set()))

        if not queues:
            return

        payload_str = json.dumps(sanitized_data)
        sse_message = f"event: {event_type}\nid: {seq}\ndata: {payload_str}\n\n"

        for q in queues:
            try:
                q.put_nowait(sse_message)
            except asyncio.QueueFull:
                logger.warning("Event queue full for subscriber on goal '%s'. Dropping.", goal_id)

    async def subscribe(self, goal_id: str) -> AsyncGenerator[str, None]:
        """Subscribe to an SSE stream for a goal."""
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=200)

        async with self._lock:
            if goal_id not in self._subscribers:
                self._subscribers[goal_id] = set()
            self._subscribers[goal_id].add(queue)

        try:
            # Send an initial connection event
            initial_msg = f"event: stream.connected\ndata: {json.dumps({'goal_id': goal_id})}\n\n"
            yield initial_msg

            while True:
                msg = await queue.get()
                yield msg
                # Terminal events signal stream end
                if "goal.completed" in msg or "goal.failed" in msg or "goal.cancelled" in msg:
                    break
        except asyncio.CancelledError:
            pass
        finally:
            async with self._lock:
                if goal_id in self._subscribers:
                    self._subscribers[goal_id].discard(queue)
                    if not self._subscribers[goal_id]:
                        del self._subscribers[goal_id]


# Singleton in-memory bus for the application process
_GLOBAL_EVENT_BUS: Optional[OrchestrationEventBus] = None


def get_event_bus() -> OrchestrationEventBus:
    """Obtain or initialize the process-level singleton event bus."""
    global _GLOBAL_EVENT_BUS
    if _GLOBAL_EVENT_BUS is None:
        _GLOBAL_EVENT_BUS = OrchestrationEventBus()
    return _GLOBAL_EVENT_BUS
