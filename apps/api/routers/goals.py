"""Router for goal lifecycle, DAG orchestration, and SSE streaming."""

import asyncio
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from apps.api.dependencies import get_app_context
from apps.api.events import OrchestrationEventBus, get_event_bus
from apps.api.schemas.common import ArtifactReferenceSchema, DataReferenceSchema
from apps.api.schemas.goals import (
    CandidatePlanResponse,
    CancelGoalResponse,
    CreateGoalRequest,
    GoalDetailResponse,
    GoalExecutionResponse,
    GoalResponse,
    TaskSchema,
)
from apps.context import AppContext
from orchestration.decision.types import DecisionResult
from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan
from orchestration.domain.types import GoalStatus, PlanStatus, TaskStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/goals", tags=["Goals & Orchestration"])

# Process-level in-memory cache to support standalone operation without DB
_IN_MEMORY_GOALS: Dict[str, Goal] = {}
_IN_MEMORY_PLANS: Dict[str, Plan] = {}
_IN_MEMORY_DECISIONS: Dict[str, DecisionResult] = {}


def _get_goal_or_404(goal_id: str, context: AppContext) -> Goal:
    """Retrieve goal from repository or in-memory fallback cache."""
    repo = None
    try:
        repo = context.create_orchestration_repository()
    except Exception:
        repo = None

    if repo is not None:
        try:
            stored = repo.goals.get(goal_id)
            if stored is not None:
                return stored
        except Exception as exc:
            logger.warning("Repository lookup for goal '%s' failed: %s", goal_id, exc)

    if goal_id in _IN_MEMORY_GOALS:
        return _IN_MEMORY_GOALS[goal_id]

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Goal '{goal_id}' not found.")


def _save_goal(goal: Goal, context: AppContext) -> None:
    """Persist goal to repository or in-memory fallback cache."""
    _IN_MEMORY_GOALS[goal.goal_id] = goal
    try:
        repo = context.create_orchestration_repository()
        if repo is not None:
            repo.goals.save(goal)
    except Exception as exc:
        logger.debug("Optional repository save for goal '%s' bypassed: %s", goal.goal_id, exc)


def _format_goal_response(goal: Goal) -> GoalResponse:
    title = goal.context.get("title") if isinstance(goal.context, dict) else None
    return GoalResponse(
        goal_id=goal.goal_id,
        title=title or goal.description,
        description=goal.description,
        status=goal.status.value,
        active_plan_id=goal.active_plan_id,
        created_at=goal.created_at.isoformat() if hasattr(goal, "created_at") and goal.created_at else datetime.now(timezone.utc).isoformat(),
        context=goal.context,
        links={
            "self": f"/api/v1/goals/{goal.goal_id}",
            "decide": f"/api/v1/goals/{goal.goal_id}/decide",
            "execute": f"/api/v1/goals/{goal.goal_id}/execute",
            "events": f"/api/v1/goals/{goal.goal_id}/events",
            "cancel": f"/api/v1/goals/{goal.goal_id}/cancel",
        },
    )


@router.post("", response_model=GoalResponse, status_code=status.HTTP_201_CREATED)
async def create_goal(
    req: CreateGoalRequest,
    context: AppContext = Depends(get_app_context),
    event_bus: OrchestrationEventBus = Depends(get_event_bus),
) -> GoalResponse:
    """Submit a new industrial Goal statement with optional inputs and parameters."""
    goal_id = f"goal-{uuid.uuid4().hex[:8]}"
    desc = req.description or req.title or "Industrial Goal"
    inputs = dict(req.inputs) if req.inputs else {}
    file_id = inputs.get("file_id")
    if file_id and "image_path" not in inputs:
        from apps.api.dependencies import get_staging_dir
        matches = list(get_staging_dir().glob(f"{file_id}_*"))
        if matches and matches[0].is_file():
            resolved_p = str(matches[0].resolve())
            inputs["image_path"] = resolved_p
            inputs["path"] = resolved_p
            inputs["file_path"] = resolved_p

    combined_context: Dict[str, Any] = {
        "title": req.title or desc,
        "inputs": inputs,
        "parameters": req.parameters,
    }

    goal = Goal(
        goal_id=goal_id,
        description=desc,
        context=combined_context,
    )

    _save_goal(goal, context)

    await event_bus.publish(
        goal_id=goal.goal_id,
        event_type="goal.created",
        data={
            "goal_id": goal.goal_id,
            "description": goal.description,
            "status": goal.status.value,
        },
    )

    return _format_goal_response(goal)


@router.post("/{goal_id}/decide", response_model=CandidatePlanResponse)
async def decide_goal(
    goal_id: str,
    context: AppContext = Depends(get_app_context),
    event_bus: OrchestrationEventBus = Depends(get_event_bus),
) -> CandidatePlanResponse:
    """Run staged routing and DAG planning/validation for a goal without executing."""
    goal = _get_goal_or_404(goal_id, context)

    decision_engine = context.create_decision_engine()

    # Process through decision engine without dispatching execution
    decision_result: DecisionResult = await decision_engine.process_goal_async(goal, execute=False)

    candidate = decision_result.candidate_plan
    plan_id = candidate.plan_id if candidate else None
    validation = decision_result.validation_result

    val_errors = [e.message for e in validation.errors] if validation and not validation.is_valid else []
    tasks_schema: List[TaskSchema] = []

    if candidate:
        domain_plan = candidate.to_plan()
        _IN_MEMORY_PLANS[domain_plan.plan_id] = domain_plan
        goal.active_plan_id = domain_plan.plan_id
        _save_goal(goal, context)
        for t_id, task in domain_plan.tasks.items():
            tasks_schema.append(
                TaskSchema(
                    task_id=task.task_id,
                    capability_id=task.capability_id,
                    title=task.title,
                    status=task.status.value,
                    dependencies=[
                        d.upstream_task_id if hasattr(d, "upstream_task_id") else str(d)
                        for d in task.dependencies
                    ],
                    parameters=task.parameters,
                )
            )

    strategy_val = (
        decision_result.decision_type.value
        if hasattr(decision_result.decision_type, "value")
        else str(decision_result.decision_type)
    )
    route_name = (
        decision_result.route_result.route_name
        if decision_result.route_result
        else None
    )

    await event_bus.publish(
        goal_id=goal.goal_id,
        event_type="decision.routed",
        data={
            "goal_id": goal.goal_id,
            "strategy": strategy_val,
            "route": route_name,
            "is_valid": validation.is_valid if validation else True,
        },
    )

    return CandidatePlanResponse(
        goal_id=goal.goal_id,
        plan_id=plan_id,
        strategy=strategy_val,
        route=route_name,
        is_valid=validation.is_valid if validation else True,
        validation_errors=val_errors,
        tasks=tasks_schema,
    )


async def _run_execution_background(
    goal: Goal,
    context: AppContext,
    event_bus: OrchestrationEventBus,
) -> None:
    """Background task executing the decided plan and publishing progress events."""
    goal_id = goal.goal_id
    try:
        await event_bus.publish(
            goal_id=goal_id,
            event_type="plan.started",
            data={"goal_id": goal_id, "status": "active"},
        )

        decision_engine = context.create_decision_engine()
        # Execute through decision engine
        decision_result = await decision_engine.process_goal_async(goal, execute=True)

        _IN_MEMORY_DECISIONS[goal_id] = decision_result

        if decision_result.candidate_plan:
            domain_plan = decision_result.candidate_plan.to_plan()
            _IN_MEMORY_PLANS[domain_plan.plan_id] = domain_plan
            if not goal.active_plan_id:
                goal.active_plan_id = domain_plan.plan_id

        if decision_result.direct_result and decision_result.direct_result.result:
            from apps.api.routers.artifacts import register_artifact
            for art in decision_result.direct_result.result.artifacts:
                if art.uri and art.uri.startswith("file://"):
                    register_artifact(art.artifact_id, Path(art.uri.replace("file://", "")))

        _save_goal(goal, context)

        if goal.status == GoalStatus.COMPLETED:
            await event_bus.publish(
                goal_id=goal_id,
                event_type="goal.completed",
                data={"goal_id": goal_id, "status": "completed"},
            )
        elif goal.status == GoalStatus.CANCELLED:
            await event_bus.publish(
                goal_id=goal_id,
                event_type="goal.cancelled",
                data={"goal_id": goal_id, "status": "cancelled"},
            )
        else:
            await event_bus.publish(
                goal_id=goal_id,
                event_type="goal.failed",
                data={
                    "goal_id": goal_id,
                    "status": "failed",
                    "error": decision_result.error or "Goal execution failed",
                },
            )

    except Exception as exc:
        logger.exception("Background execution failed for goal '%s': %s", goal_id, exc)
        if goal.status == GoalStatus.ACTIVE:
            goal.mark_failed()
            _save_goal(goal, context)
        await event_bus.publish(
            goal_id=goal_id,
            event_type="goal.failed",
            data={"goal_id": goal_id, "status": "failed", "error": str(exc)},
        )


@router.post("/{goal_id}/execute", response_model=GoalExecutionResponse, status_code=status.HTTP_202_ACCEPTED)
async def execute_goal(
    goal_id: str,
    background_tasks: BackgroundTasks,
    context: AppContext = Depends(get_app_context),
    event_bus: OrchestrationEventBus = Depends(get_event_bus),
) -> GoalExecutionResponse:
    """Dispatch asynchronous execution of a Goal and return streaming endpoint link."""
    goal = _get_goal_or_404(goal_id, context)
    if goal.status != GoalStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot execute goal '{goal_id}': status is '{goal.status.value}', expected 'pending'.",
        )

    # Launch execution in background
    asyncio.create_task(_run_execution_background(goal, context, event_bus))

    return GoalExecutionResponse(
        goal_id=goal.goal_id,
        plan_id=goal.active_plan_id,
        status="active",
        events_url=f"/api/v1/goals/{goal.goal_id}/events",
    )


@router.get("/{goal_id}", response_model=GoalDetailResponse)
async def get_goal_detail(
    goal_id: str,
    context: AppContext = Depends(get_app_context),
) -> GoalDetailResponse:
    """Get status, active plan, task statuses, and outputs for a goal."""
    goal = _get_goal_or_404(goal_id, context)

    plan = None
    if goal.active_plan_id:
        if goal.active_plan_id in _IN_MEMORY_PLANS:
            plan = _IN_MEMORY_PLANS[goal.active_plan_id]
        else:
            try:
                repo = context.create_orchestration_repository()
                if repo is not None:
                    plan = repo.plans.get(goal.active_plan_id)
            except Exception as exc:
                logger.debug("Repository lookup for plan '%s' bypassed: %s", goal.active_plan_id, exc)

    tasks_schema: List[TaskSchema] = []
    results_map: Dict[str, Any] = {}
    artifacts_list: List[ArtifactReferenceSchema] = []

    from apps.api.routers.artifacts import register_artifact

    if plan:
        for t_id, task in plan.tasks.items():
            err_msg = task.error.message if task.error else None
            out_val = task.result.output if task.result else None
            if task.result:
                results_map[t_id] = out_val
                for art in task.result.artifacts:
                    if art.uri and art.uri.startswith("file://"):
                        register_artifact(art.artifact_id, Path(art.uri.replace("file://", "")))
                    artifacts_list.append(
                        ArtifactReferenceSchema(
                            artifact_id=art.artifact_id,
                            name=art.name,
                            uri=art.uri,
                            mime_type=art.mime_type,
                            size_bytes=art.size_bytes,
                            download_url=f"/api/v1/artifacts/{art.artifact_id}/download",
                            metadata=art.metadata,
                        )
                    )

            tasks_schema.append(
                TaskSchema(
                    task_id=task.task_id,
                    capability_id=task.capability_id,
                    title=task.title,
                    status=task.status.value,
                    dependencies=[
                        d.upstream_task_id if hasattr(d, "upstream_task_id") else str(d)
                        for d in task.dependencies
                    ],
                    parameters=task.parameters,
                    output=out_val,
                    error=err_msg,
                )
            )

    # Populate direct capability execution outputs and generated artifacts
    if goal_id in _IN_MEMORY_DECISIONS:
        dec = _IN_MEMORY_DECISIONS[goal_id]
        if dec.direct_result and dec.direct_result.result:
            dir_res = dec.direct_result.result
            if dir_res.output is not None and "direct" not in results_map:
                results_map["direct"] = dir_res.output
            for art in dir_res.artifacts:
                if art.uri and art.uri.startswith("file://"):
                    register_artifact(art.artifact_id, Path(art.uri.replace("file://", "")))
                art_schema = ArtifactReferenceSchema(
                    artifact_id=art.artifact_id,
                    name=art.name,
                    uri=art.uri,
                    mime_type=art.mime_type,
                    size_bytes=art.size_bytes,
                    download_url=f"/api/v1/artifacts/{art.artifact_id}/download",
                    metadata=art.metadata,
                )
                if not any(a.artifact_id == art.artifact_id for a in artifacts_list):
                    artifacts_list.append(art_schema)

    plan_resp = None
    if plan:
        plan_resp = CandidatePlanResponse(
            goal_id=goal.goal_id,
            plan_id=plan.plan_id,
            strategy="plan_required",
            is_valid=True,
            tasks=tasks_schema,
        )

    return GoalDetailResponse(
        goal=_format_goal_response(goal),
        plan=plan_resp,
        tasks=tasks_schema,
        results=results_map,
        artifacts=artifacts_list,
    )


@router.get("/{goal_id}/events")
async def stream_goal_events(
    goal_id: str,
    event_bus: OrchestrationEventBus = Depends(get_event_bus),
) -> StreamingResponse:
    """Stream real-time lifecycle events via Server-Sent Events (SSE)."""
    return StreamingResponse(
        event_bus.subscribe(goal_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{goal_id}/cancel", response_model=CancelGoalResponse)
async def cancel_goal(
    goal_id: str,
    context: AppContext = Depends(get_app_context),
    event_bus: OrchestrationEventBus = Depends(get_event_bus),
) -> CancelGoalResponse:
    """Request graceful cancellation of an active or pending goal."""
    goal = _get_goal_or_404(goal_id, context)
    if goal.status == GoalStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel goal '{goal_id}': already completed.",
        )

    orchestrator = context.create_goal_orchestrator()
    plan = _IN_MEMORY_PLANS.get(goal.active_plan_id) if goal.active_plan_id else None

    try:
        orchestrator.cancel_goal(goal, plan=plan)
    except Exception as exc:
        logger.warning("Orchestrator cancel_goal error (marking goal cancelled): %s", exc)
        goal.cancel()

    _save_goal(goal, context)

    await event_bus.publish(
        goal_id=goal.goal_id,
        event_type="goal.cancelled",
        data={"goal_id": goal.goal_id, "status": "cancelled"},
    )

    return CancelGoalResponse(goal_id=goal.goal_id, status="cancelled")
