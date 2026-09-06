"""DecisionEngine coordinating staged intent routing, validation, and orchestrator execution."""

import asyncio
from typing import Any, Callable, Dict, Optional

from orchestration.decision.types import DecisionPolicy, DecisionResult
from orchestration.domain.goals import Goal
from orchestration.domain.types import GoalStatus
from orchestration.orchestrator import GoalOrchestrator
from orchestration.planning.base import Planner
from orchestration.planning.types import PlanningContext
from orchestration.routing.base import IntentRouter
from orchestration.routing.types import ExecutionStrategy, RouteResult
from orchestration.validation.validator import PlanValidator


class DecisionEngine:
    """Entrypoint decision coordinator sitting directly above GoalOrchestrator.
    
    Responsibilities:
      1. Classify incoming Goals via StagedEscalationRouter (Deterministic -> Semantic -> LLM -> Fallback).
      2. Dispatch DIRECT_DETERMINISTIC and DIRECT_CAPABILITY goals to GoalOrchestrator direct execution,
         preserving Goal lifecycle and milestone persistence while keeping results out of Goal.context.
      3. For PLAN_REQUIRED goals: invoke Planner to propose CandidatePlan, run 4-stage deterministic
         validation, and if valid, submit domain Plan to GoalOrchestrator.
      4. Reject invalid or unauthorized requests cleanly.
    """

    def __init__(
        self,
        router: IntentRouter,
        orchestrator: GoalOrchestrator,
        planner: Planner,
        validator: PlanValidator,
        policy: Optional[DecisionPolicy] = None,
        deterministic_handlers: Optional[Dict[str, Callable[[Goal], Any]]] = None,
    ) -> None:
        self._router = router
        self._orchestrator = orchestrator
        self._planner = planner
        self._validator = validator
        self._policy = policy or DecisionPolicy()
        self._deterministic_handlers = deterministic_handlers or {}

    def process_goal(self, goal: Goal, execute: bool = True) -> DecisionResult:
        """Synchronously process a Goal through routing, planning/validation, and orchestration."""
        # 1. Staged routing
        route_res = self._router.route(goal)

        # 2. Strategy dispatch
        if route_res.strategy == ExecutionStrategy.DIRECT_DETERMINISTIC:
            return self._handle_direct_deterministic(goal, route_res, execute)

        elif route_res.strategy == ExecutionStrategy.DIRECT_CAPABILITY:
            return self._handle_direct_capability(goal, route_res, execute)

        elif route_res.strategy == ExecutionStrategy.PLAN_REQUIRED:
            return self._handle_plan_required(goal, route_res, execute)

        elif route_res.strategy == ExecutionStrategy.REJECT:
            return self._handle_reject(goal, route_res)

        else:
            raise ValueError(f"Unsupported execution strategy: {route_res.strategy}")

    async def process_goal_async(self, goal: Goal, execute: bool = True) -> DecisionResult:
        """Asynchronously process a Goal through routing, planning/validation, and orchestration."""
        # 1. Staged routing
        route_res = await self._router.route_async(goal)

        # 2. Strategy dispatch
        if route_res.strategy == ExecutionStrategy.DIRECT_DETERMINISTIC:
            return await self._handle_direct_deterministic_async(goal, route_res, execute)

        elif route_res.strategy == ExecutionStrategy.DIRECT_CAPABILITY:
            return await self._handle_direct_capability_async(goal, route_res, execute)

        elif route_res.strategy == ExecutionStrategy.PLAN_REQUIRED:
            return await self._handle_plan_required_async(goal, route_res, execute)

        elif route_res.strategy == ExecutionStrategy.REJECT:
            return self._handle_reject(goal, route_res)

        else:
            raise ValueError(f"Unsupported execution strategy: {route_res.strategy}")

    def _build_deterministic_invoker(self, goal: Goal, route_res: RouteResult) -> Callable[[], Any]:
        import inspect

        raw_handler = self._deterministic_handlers.get(
            route_res.route_name,
            lambda: {"status": "ok", "route": route_res.route_name},
        )

        def _invoker() -> Any:
            try:
                sig = inspect.signature(raw_handler)
                if len(sig.parameters) > 0:
                    return raw_handler(goal)
                return raw_handler()
            except TypeError:
                return raw_handler()

        return _invoker

    def _handle_direct_deterministic(
        self, goal: Goal, route_res: RouteResult, execute: bool
    ) -> DecisionResult:
        if not execute:
            return DecisionResult(
                decision_type=ExecutionStrategy.DIRECT_DETERMINISTIC,
                goal_id=goal.goal_id,
                route_result=route_res,
            )

        invoker = self._build_deterministic_invoker(goal, route_res)
        direct_res = self._orchestrator.execute_deterministic_goal(goal, handler=invoker)
        return DecisionResult(
            decision_type=ExecutionStrategy.DIRECT_DETERMINISTIC,
            goal_id=goal.goal_id,
            route_result=route_res,
            direct_result=direct_res,
            error=direct_res.error.message if direct_res.error else None,
        )

    async def _handle_direct_deterministic_async(
        self, goal: Goal, route_res: RouteResult, execute: bool
    ) -> DecisionResult:
        if not execute:
            return DecisionResult(
                decision_type=ExecutionStrategy.DIRECT_DETERMINISTIC,
                goal_id=goal.goal_id,
                route_result=route_res,
            )

        invoker = self._build_deterministic_invoker(goal, route_res)
        direct_res = await self._orchestrator.execute_deterministic_goal_async(goal, handler=invoker)
        return DecisionResult(
            decision_type=ExecutionStrategy.DIRECT_DETERMINISTIC,
            goal_id=goal.goal_id,
            route_result=route_res,
            direct_result=direct_res,
            error=direct_res.error.message if direct_res.error else None,
        )

    def _handle_direct_capability(
        self, goal: Goal, route_res: RouteResult, execute: bool
    ) -> DecisionResult:
        cap_id = route_res.target_capability_id or route_res.route_name
        if not execute:
            return DecisionResult(
                decision_type=ExecutionStrategy.DIRECT_CAPABILITY,
                goal_id=goal.goal_id,
                route_result=route_res,
            )

        params = goal.context.get("parameters", {})
        inputs = goal.context.get("inputs", {})
        direct_res = self._orchestrator.execute_direct_goal(
            goal=goal,
            capability_id=cap_id,
            parameters=params,
            inputs=inputs,
        )
        return DecisionResult(
            decision_type=ExecutionStrategy.DIRECT_CAPABILITY,
            goal_id=goal.goal_id,
            route_result=route_res,
            direct_result=direct_res,
            error=direct_res.error.message if direct_res.error else None,
        )

    async def _handle_direct_capability_async(
        self, goal: Goal, route_res: RouteResult, execute: bool
    ) -> DecisionResult:
        cap_id = route_res.target_capability_id or route_res.route_name
        if not execute:
            return DecisionResult(
                decision_type=ExecutionStrategy.DIRECT_CAPABILITY,
                goal_id=goal.goal_id,
                route_result=route_res,
            )

        params = goal.context.get("parameters", {})
        inputs = goal.context.get("inputs", {})
        direct_res = await self._orchestrator.execute_direct_goal_async(
            goal=goal,
            capability_id=cap_id,
            parameters=params,
            inputs=inputs,
        )
        return DecisionResult(
            decision_type=ExecutionStrategy.DIRECT_CAPABILITY,
            goal_id=goal.goal_id,
            route_result=route_res,
            direct_result=direct_res,
            error=direct_res.error.message if direct_res.error else None,
        )

    def _handle_plan_required(
        self, goal: Goal, route_res: RouteResult, execute: bool
    ) -> DecisionResult:
        context = PlanningContext(goal=goal)
        candidate = self._planner.plan(context)

        # 4-stage validation
        val_res = self._validator.validate(candidate, available_task_ids=context.available_task_ids)
        if not val_res.is_valid:
            err_msg = "; ".join(e.message for e in val_res.errors)
            return DecisionResult(
                decision_type=ExecutionStrategy.PLAN_REQUIRED,
                goal_id=goal.goal_id,
                route_result=route_res,
                candidate_plan=candidate,
                validation_result=val_res,
                error=f"Plan validation failed: {err_msg}",
            )

        domain_plan = candidate.to_plan()

        if execute:
            self._orchestrator.execute_goal(goal, domain_plan)

        return DecisionResult(
            decision_type=ExecutionStrategy.PLAN_REQUIRED,
            goal_id=goal.goal_id,
            plan_id=domain_plan.plan_id,
            route_result=route_res,
            candidate_plan=candidate,
            validation_result=val_res,
        )

    async def _handle_plan_required_async(
        self, goal: Goal, route_res: RouteResult, execute: bool
    ) -> DecisionResult:
        context = PlanningContext(goal=goal)
        candidate = await self._planner.plan_async(context)

        # 4-stage validation
        val_res = await asyncio.to_thread(
            self._validator.validate, candidate, context.available_task_ids
        )
        if not val_res.is_valid:
            err_msg = "; ".join(e.message for e in val_res.errors)
            return DecisionResult(
                decision_type=ExecutionStrategy.PLAN_REQUIRED,
                goal_id=goal.goal_id,
                route_result=route_res,
                candidate_plan=candidate,
                validation_result=val_res,
                error=f"Plan validation failed: {err_msg}",
            )

        domain_plan = candidate.to_plan()

        if execute:
            await self._orchestrator.execute_goal_async(goal, domain_plan)

        return DecisionResult(
            decision_type=ExecutionStrategy.PLAN_REQUIRED,
            goal_id=goal.goal_id,
            plan_id=domain_plan.plan_id,
            route_result=route_res,
            candidate_plan=candidate,
            validation_result=val_res,
        )

    def _handle_reject(self, goal: Goal, route_res: RouteResult) -> DecisionResult:
        reason = route_res.metadata.get("reason", "Request rejected by policy.")
        if goal.status == GoalStatus.PENDING:
            goal.activate(f"rejected:{route_res.route_name}")
        if goal.status == GoalStatus.ACTIVE:
            goal.mark_failed()
        if self._orchestrator.repository is not None:
            self._orchestrator.repository.save_goal(goal)

        return DecisionResult(
            decision_type=ExecutionStrategy.REJECT,
            goal_id=goal.goal_id,
            route_result=route_res,
            error=reason,
        )
