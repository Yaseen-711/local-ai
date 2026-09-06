"""LLM-based planner using structured JSON inference."""

import asyncio
import json
from typing import Any, Dict, List, Optional
import uuid

from connectors import InferenceConnector
from core.common.parsing import parse_json_payload
from core.inference.types import GenerationOptions, OutputConstraint
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.dependencies import Dependency
from orchestration.domain.references import DataReference
from orchestration.planning.base import Planner
from orchestration.planning.types import CandidatePlan, CandidateTask, PlanningContext


class LLMPlanner(Planner):
    """Generates candidate plans through structured LLM reasoning.
    
    Proposes CandidatePlans; does NOT activate, execute, or persist them.
    """

    def __init__(
        self,
        connector: InferenceConnector,
        capability_registry: CapabilityRegistry,
        model_id: str = "default",
    ) -> None:
        self._connector = connector
        self._registry = capability_registry
        self._model_id = model_id

    def plan(self, context: PlanningContext) -> CandidatePlan:
        """Synchronously propose a CandidatePlan using the LLM."""
        capabilities_info = []
        for desc in self._registry.list_descriptors():
            if desc.is_available and not desc.is_deprecated:
                capabilities_info.append({
                    "capability_id": desc.capability_id,
                    "description": desc.description,
                    "parameter_schema": desc.parameter_schema,
                    "input_schema": desc.input_schema,
                    "output_schema": desc.output_schema,
                })

        system_prompt = (
            "You are an expert task planner for an autonomous local AI system.\n"
            "Given a user goal and a catalog of available capabilities, construct a structured DAG execution plan.\n"
            "Respond STRICTLY in JSON with the following structure:\n"
            "{\n"
            '  "title": "Plan title",\n'
            '  "tasks": [\n'
            "    {\n"
            '      "task_id": "t1",\n'
            '      "title": "Task title",\n'
            '      "capability_id": "chosen_capability_id",\n'
            '      "description": "What this task does",\n'
            '      "parameters": {},\n'
            '      "input_references": [\n'
            '        {"key": "ref_key", "source_task_id": "upstream_task_id"}\n'
            "      ],\n"
            '      "dependencies": ["upstream_task_id"]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Rules:\n"
            "1. Every task MUST reference a valid capability_id from the catalog.\n"
            "2. Dependencies must form a strict Directed Acyclic Graph (DAG) with no cycles.\n"
            "3. If a task uses an input_reference from another task in the plan, that task MUST be listed in dependencies.\n"
            "4. When creating a formal executive summary, report, or deliverable, use 'artifact.generate' with artifact_type (e.g. docx).\n"
            "5. For input_references to an upstream task, use key 'output' (e.g. {\"key\": \"output\", \"source_task_id\": \"t1\"}).\n"
        )

        inputs = (
            context.goal.context.get("inputs", {})
            if isinstance(getattr(context.goal, "context", None), dict)
            else {}
        )

        user_content = {
            "goal_id": context.goal.goal_id,
            "goal_description": context.goal.description,
            "inputs": inputs,
            "available_capabilities": capabilities_info,
        }

        if context.completed_tasks:
            user_content["already_completed_tasks"] = list(context.completed_tasks.keys())

        user_prompt = (
            f"Goal and Context:\n{json.dumps(user_content, indent=2)}\n\n"
            "Generate the execution plan JSON:"
        )

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        response = self._connector.infer_prompt(
            model_id=self._model_id,
            prompt=full_prompt,
            options=GenerationOptions(
                temperature=0.1,
                max_tokens=2048,
                constraint=OutputConstraint.json(),
            ),
        )

        data = parse_json_payload(response.text)
        if not isinstance(data, dict):
            raise ValueError(f"LLM planner returned non-dictionary JSON: {type(data)}")

        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        title = data.get("title", f"Plan for {context.goal.description[:30]}")
        tasks_data = data.get("tasks", [])

        candidate_tasks: List[CandidateTask] = []
        plan_dependencies: List[Dependency] = []

        for idx, td in enumerate(tasks_data):
            tid = td.get("task_id") or f"task_{idx+1}"
            c_title = td.get("title", f"Task {idx+1}")
            cap_id = td.get("capability_id", "")
            desc = td.get("description", "")
            params = td.get("parameters", {}) or {}

            # Parse input references
            raw_refs = td.get("input_references", []) or []
            refs_dict: Dict[str, DataReference] = {}
            for r in raw_refs:
                if isinstance(r, dict):
                    logical_name = r.get("name") or r.get("key", "output")
                    if logical_name in ("ref_key", "input", ""):
                        logical_name = "output"
                    source_key = r.get("source_key") or "output"
                    if logical_name in ("candidates", "answer", "tags", "text", "content", "stdout"):
                        source_key = logical_name
                    refs_dict[logical_name] = DataReference(
                        key=source_key,
                        source_task_id=r.get("source_task_id"),
                        uri=r.get("uri"),
                        mime_type=r.get("mime_type", "application/json"),
                    )

            # Parse dependencies
            raw_deps = td.get("dependencies", []) or []
            task_deps: List[Dependency] = []
            for upstream in raw_deps:
                dep = Dependency(upstream_task_id=upstream, downstream_task_id=tid)
                task_deps.append(dep)
                plan_dependencies.append(dep)

            candidate_tasks.append(
                CandidateTask(
                    task_id=tid,
                    title=c_title,
                    capability_id=cap_id,
                    description=desc,
                    parameters=params,
                    input_references=refs_dict,
                    dependencies=task_deps,
                )
            )

        return CandidatePlan(
            plan_id=plan_id,
            goal_id=context.goal.goal_id,
            title=title,
            tasks=candidate_tasks,
            dependencies=plan_dependencies,
            metadata={"planner": "llm", "model_id": self._model_id, "inputs": inputs},
        )

    async def plan_async(self, context: PlanningContext) -> CandidatePlan:
        """Asynchronously propose a CandidatePlan."""
        return await asyncio.to_thread(self.plan, context)
