"""AgentCapability — Adaptive PydanticAI execution capability.

Provides the Foundation-facing entry point for bounded agentic task execution
satisfying the Capability protocol. Encapsulates PydanticAI behind standard
Foundation seams: ModelSelectionPolicy, CapabilityToolAdapter, and AgentExecutionPolicy.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import time
from typing import Any, Dict, List, Optional

try:
    from pydantic_ai import Agent, CancellationToken
    from pydantic_ai.usage import UsageLimitExceeded, UsageLimits
    _PYDANTIC_AI_AVAILABLE = True
except ImportError:
    _PYDANTIC_AI_AVAILABLE = False
    Agent = None  # type: ignore
    CancellationToken = None  # type: ignore
    UsageLimits = None  # type: ignore
    UsageLimitExceeded = Exception  # type: ignore

from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.builtin.agent.model_adapter import (
    FoundationPydanticAIModel,
)
from orchestration.capabilities.builtin.agent.policy import (
    AgentExecutionPolicy,
)
from orchestration.capabilities.builtin.agent.tool_adapter import (
    CapabilityToolAdapter,
)
from orchestration.capabilities.builtin.agent.types import (
    AgentExecutionOutput,
    AgentParameters,
    AgentProposal,
)
from orchestration.capabilities.descriptor import CapabilityDescriptor
from orchestration.domain.results import TaskResult
from orchestration.errors import CapabilityUnavailableError

logger = logging.getLogger(__name__)


class AgentCapability:
    """Bounded adaptive execution capability wrapping PydanticAI.

    Semantic contract:
        Inputs:
            - 'prompt' (str) or 'objective' (str, required): Task description or goal prompt.
        Parameters:
            - 'allowed_capabilities' (List[str], optional): Whitelist of capabilities exposed as tools.
            - 'model_tier' (str or ModelTier, optional): 'lightweight' or 'reasoning' (default: reasoning).
            - 'max_iterations' (int, default: 10): Max model interaction turns.
            - 'max_tool_calls' (int, default: 20): Max tool executions.
            - 'timeout_seconds' (float, default: 120.0): Overall execution timeout.
            - 'system_prompt' (str, optional): Custom instructions for the agent.
    """

    def __init__(
        self,
        model_adapter: FoundationPydanticAIModel,
        tool_adapter: CapabilityToolAdapter,
        policy: Optional[AgentExecutionPolicy] = None,
    ) -> None:
        self._model_adapter = model_adapter
        self._tool_adapter = tool_adapter
        self._policy = policy or AgentExecutionPolicy()

    @property
    def capability_id(self) -> str:
        """Canonical declarative identifier for this capability."""
        return "agent.pydantic_ai"

    @property
    def is_available(self) -> bool:
        """Advisory check indicating whether PydanticAI dependencies are available."""
        return _PYDANTIC_AI_AVAILABLE

    def get_descriptor(self) -> CapabilityDescriptor:
        """Declarative catalog descriptor for this capability."""
        return CapabilityDescriptor(
            capability_id=self.capability_id,
            description="Adaptive agent capability powered by PydanticAI with bounded tool execution.",
            parameter_schema={
                "allowed_capabilities": {"type": "array", "items": {"type": "string"}},
                "model_tier": {"type": "string", "enum": ["lightweight", "reasoning"], "default": "reasoning"},
                "max_iterations": {"type": "integer", "default": 10},
                "max_tool_calls": {"type": "integer", "default": 20},
                "timeout_seconds": {"type": "number", "default": 120.0},
                "system_prompt": {"type": "string"},
            },
            input_schema={
                "prompt": {"type": "string", "required": True},
            },
            output_schema={
                "response": {"type": "string"},
                "iterations": {"type": "integer"},
                "tool_calls": {"type": "array"},
                "finish_reason": {"type": "string"},
                "proposal": {"type": "object"},
            },
            is_available=self.is_available,
        )

    def _extract_prompt(self, inputs: Dict[str, Any], parameters: Dict[str, Any]) -> str:
        """Resolve prompt or objective string from inputs or parameters."""
        prompt = (
            inputs.get("prompt")
            or inputs.get("objective")
            or inputs.get("text")
            or inputs.get("description")
            or parameters.get("prompt")
            or parameters.get("objective")
            or parameters.get("text")
            or parameters.get("description")
        )
        if not prompt or not str(prompt).strip():
            raise ValueError(
                f"Capability '{self.capability_id}' requires a non-empty 'prompt' or 'objective' in inputs."
            )
        return str(prompt).strip()

    def _extract_proposal_if_present(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Extract structured advisory replan proposal if present in agent output."""
        text = response_text.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                data = json.loads(text)
                if isinstance(data, dict) and "proposal" in data and isinstance(data["proposal"], dict):
                    prop = data["proposal"]
                    return AgentProposal(
                        type=str(prop.get("type", "replanning")),
                        reason=str(prop.get("reason", "")),
                        suggested_tasks=list(prop.get("suggested_tasks", [])),
                        metadata=dict(prop.get("metadata", {})),
                    ).to_dict()
            except Exception:
                pass
        return None

    def execute(
        self,
        parameters: Dict[str, Any],
        inputs: Dict[str, Any],
        context: CapabilityContext,
    ) -> TaskResult:
        """Execute the bounded agent loop synchronously adhering to Capability protocol."""
        if not self.is_available:
            raise CapabilityUnavailableError(
                f"Capability '{self.capability_id}' is unavailable: "
                "pydantic_ai is not installed in the current environment. "
                "Install 'pydantic-ai-slim~=2.40.0' to enable agent capabilities."
            )

        # 1. Parse parameters & inputs
        agent_params = AgentParameters.from_dict(parameters)
        prompt = self._extract_prompt(inputs, parameters)

        # 2. Check early cancellation
        if context.metadata.get("cancelled", False):
            return TaskResult(
                output={
                    "response": "Execution cancelled before start.",
                    "iterations": 0,
                    "tool_calls": [],
                    "finish_reason": "cancelled",
                    "proposal": None,
                },
                metadata={"status": "cancelled", "execution_id": context.execution_id},
            )

        # 3. Configure model adapter tier
        self._model_adapter.set_tier(agent_params.model_tier)

        # 4. Build tools and reset adapter state
        self._tool_adapter.reset_state()
        allowed_set = set(agent_params.allowed_capabilities)
        tools = self._tool_adapter.build_tools(
            allowed_capabilities=allowed_set,
            parent_context=context,
        )

        # 5. Build Agent
        system_prompt = agent_params.system_prompt or "You are a helpful and precise AI assistant."
        agent = Agent(
            model=self._model_adapter,
            tools=tools,
            system_prompt=system_prompt,
        )

        # 6. Prepare limits and cancellation token
        cancel_token = CancellationToken()
        usage_limits = UsageLimits(
            request_limit=agent_params.max_iterations,
            tool_calls_limit=agent_params.max_tool_calls,
        )

        # 7. Execute async agent run with timeout
        start_time = time.perf_counter()

        async def _run_agent() -> Any:
            return await agent.run(
                prompt,
                usage_limits=usage_limits,
                cancellation_token=cancel_token,
            )

        finish_reason = "stop"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        try:
            # Enforce timeout_seconds safely whether called from sync context or running event loop
            if loop and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    run_result = pool.submit(
                        lambda: asyncio.run(
                            asyncio.wait_for(_run_agent(), timeout=agent_params.timeout_seconds)
                        )
                    ).result()
            else:
                run_result = asyncio.run(
                    asyncio.wait_for(_run_agent(), timeout=agent_params.timeout_seconds)
                )
            response_text = str(run_result.output) if run_result and run_result.output is not None else ""
            messages = run_result.all_messages() if hasattr(run_result, "all_messages") else []
            iterations = max(1, len([m for m in messages if getattr(m, "kind", None) == "request"]))
        except UsageLimitExceeded as exc:
            logger.warning("Agent budget exhausted: %s", exc)
            finish_reason = "budget_exceeded"
            response_text = f"Execution halted: budget exceeded ({exc})."
            iterations = agent_params.max_iterations
        except (TimeoutError, asyncio.TimeoutError) as exc:
            logger.error("Agent execution timed out after %s seconds", agent_params.timeout_seconds)
            raise TimeoutError(
                f"Agent execution timed out after {agent_params.timeout_seconds} seconds"
            ) from exc
        except Exception as exc:
            logger.exception("Agent execution failed: %s", exc)
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # 8. Extract any advisory orchestration proposals
        proposal = self._extract_proposal_if_present(response_text)

        # 9. Format typed execution output
        tool_records = [t.to_dict() for t in self._tool_adapter.tool_traces]
        execution_output = AgentExecutionOutput(
            response=response_text,
            iterations=iterations,
            tool_calls=tool_records,
            finish_reason=finish_reason,
            proposal=proposal,
        )

        # 10. Assemble and return TaskResult with provenance
        return TaskResult(
            output=execution_output.to_dict(),
            references=self._tool_adapter.collected_references,
            artifacts=self._tool_adapter.collected_artifacts,
            metadata={
                "execution_id": context.execution_id,
                "model_id": self._model_adapter.model_name,
                "model_tier": agent_params.model_tier.value,
                "iterations": iterations,
                "tool_calls_count": len(tool_records),
                "duration_ms": duration_ms,
                "finish_reason": finish_reason,
            },
        )
