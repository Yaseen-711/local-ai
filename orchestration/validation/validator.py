"""Deterministic 4-stage plan validator."""

from collections import deque
from typing import Any, Dict, List, Optional, Set, Union

from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.plans import Plan
from orchestration.domain.references import DataReference
from orchestration.validation.types import ValidationError, ValidationResult, ValidationStage


class PlanValidator:
    """Deterministic validator enforcing plan structural, capability, constraint, and feasibility rules.
    
    Stages:
      1. Structural: Acyclicity, referential integrity, no self-loops, disconnected components allowed.
      2. Capability: All task capabilities exist in registry, required parameters present.
      3. Hard Constraints: Max tasks limit, max DAG depth limit.
      4. Feasibility: Execution-produced DataReferences must have DAG ordering paths,
         while already-available references (prior completed tasks, context URIs) require no ordering.
    """

    def __init__(
        self,
        capability_registry: CapabilityRegistry,
        max_tasks: int = 50,
        max_depth: int = 10,
    ) -> None:
        self.registry = capability_registry
        self.max_tasks = max_tasks
        self.max_depth = max_depth

    def validate(
        self,
        plan: Any,
        available_task_ids: Optional[Set[str]] = None,
        available_artifact_ids: Optional[Set[str]] = None,
    ) -> ValidationResult:
        """Validate a domain Plan or CandidatePlan across all 4 deterministic stages.
        
        Args:
            plan: Domain Plan or CandidatePlan object.
            available_task_ids: IDs of tasks already completed in previous revisions/executions.
            available_artifact_ids: IDs of artifacts already produced and available.
            
        Returns:
            ValidationResult containing pass/fail status, errors, and warnings.
        """
        result = ValidationResult()
        available_tasks = available_task_ids or set()
        available_artifacts = available_artifact_ids or set()

        tasks_info = self._extract_plan_info(plan)
        if not tasks_info["tasks"]:
            result.add_error(
                stage=ValidationStage.STRUCTURAL,
                code="EMPTY_PLAN",
                message="Plan does not contain any tasks.",
            )
            return result

        # Stage 1: Structural Validation
        self._validate_structural(tasks_info, result)
        if not result.is_valid:
            # If structural validation failed (e.g. cycles, unknown dependencies), abort deeper checks
            return result

        # Stage 2: Capability Validation
        self._validate_capabilities(tasks_info, result)

        # Stage 3: Hard Constraints
        self._validate_hard_constraints(tasks_info, result)

        # Stage 4: Feasibility Validation
        self._validate_feasibility(tasks_info, available_tasks, available_artifacts, result)

        return result

    def _extract_plan_info(self, plan: Any) -> Dict[str, Any]:
        """Normalize domain Plan or CandidatePlan into a standardized structure."""
        tasks_map = {}
        # Support domain Plan (plan.tasks is dict or list)
        if hasattr(plan, "tasks"):
            raw_tasks = plan.tasks
            if isinstance(raw_tasks, dict):
                raw_tasks_list = list(raw_tasks.values())
            else:
                raw_tasks_list = list(raw_tasks)
        else:
            raw_tasks_list = []

        dependencies = []
        for t in raw_tasks_list:
            task_id = getattr(t, "task_id")
            cap_id = getattr(t, "capability_id")
            params = getattr(t, "parameters", {}) or {}
            raw_input_refs = getattr(t, "input_references", {}) or {}
            if isinstance(raw_input_refs, dict):
                input_refs = list(raw_input_refs.values())
            else:
                input_refs = list(raw_input_refs)
            task_deps = getattr(t, "dependencies", []) or []

            tasks_map[task_id] = {
                "task_id": task_id,
                "capability_id": cap_id,
                "parameters": params,
                "input_references": input_refs,
                "dependencies": task_deps,
            }

            for dep in task_deps:
                upstream = getattr(dep, "upstream_task_id")
                downstream = getattr(dep, "downstream_task_id", task_id)
                dependencies.append((upstream, downstream))

        # Also support explicit top-level dependencies if present (e.g. on CandidatePlan)
        if hasattr(plan, "dependencies"):
            for dep in plan.dependencies:
                upstream = getattr(dep, "upstream_task_id")
                downstream = getattr(dep, "downstream_task_id")
                if (upstream, downstream) not in dependencies:
                    dependencies.append((upstream, downstream))

        return {"tasks": tasks_map, "dependencies": dependencies}

    def _validate_structural(self, info: Dict[str, Any], result: ValidationResult) -> None:
        """Verify DAG acyclicity, referential integrity, and validate work graph."""
        tasks = info["tasks"]
        dependencies = info["dependencies"]
        task_ids = set(tasks.keys())

        # Check dependency references and self-loops
        adj: Dict[str, List[str]] = {tid: [] for tid in task_ids}
        reverse_adj: Dict[str, List[str]] = {tid: [] for tid in task_ids}

        for upstream, downstream in dependencies:
            if upstream == downstream:
                result.add_error(
                    stage=ValidationStage.STRUCTURAL,
                    code="SELF_DEPENDENCY",
                    message=f"Task '{downstream}' cannot depend on itself.",
                    task_id=downstream,
                )
                continue

            if upstream not in task_ids:
                result.add_error(
                    stage=ValidationStage.STRUCTURAL,
                    code="UNKNOWN_UPSTREAM_TASK",
                    message=f"Dependency references unknown upstream task '{upstream}'.",
                    task_id=downstream,
                    details={"upstream_task_id": upstream},
                )
                continue

            if downstream not in task_ids:
                result.add_error(
                    stage=ValidationStage.STRUCTURAL,
                    code="UNKNOWN_DOWNSTREAM_TASK",
                    message=f"Dependency references unknown downstream task '{downstream}'.",
                    task_id=downstream,
                    details={"downstream_task_id": downstream},
                )
                continue

            adj[upstream].append(downstream)
            reverse_adj[downstream].append(upstream)

        if not result.is_valid:
            return

        # DFS Cycle Detection
        visited: Dict[str, int] = {tid: 0 for tid in task_ids}

        def _has_cycle(node: str) -> bool:
            visited[node] = 1  # visiting
            for neighbor in adj[node]:
                if visited[neighbor] == 1:
                    result.add_error(
                        stage=ValidationStage.STRUCTURAL,
                        code="DEPENDENCY_CYCLE",
                        message=f"Cycle detected in plan DAG involving task '{node}' and '{neighbor}'.",
                        task_id=node,
                        details={"from_task": node, "to_task": neighbor},
                    )
                    return True
                if visited[neighbor] == 0:
                    if _has_cycle(neighbor):
                        return True
            visited[node] = 2  # visited
            return False

        for tid in task_ids:
            if visited[tid] == 0:
                if _has_cycle(tid):
                    break

        info["adj"] = adj
        info["reverse_adj"] = reverse_adj

    def _validate_capabilities(self, info: Dict[str, Any], result: ValidationResult) -> None:
        """Ensure all capabilities exist in registry and parameter schemas are satisfied."""
        for tid, t_data in info["tasks"].items():
            cap_id = t_data["capability_id"]
            if not self.registry.has(cap_id):
                result.add_error(
                    stage=ValidationStage.CAPABILITY,
                    code="UNKNOWN_CAPABILITY",
                    message=f"Task '{tid}' references unregistered capability '{cap_id}'.",
                    task_id=tid,
                    details={"capability_id": cap_id},
                )
                continue

            descriptor = self.registry.get_descriptor(cap_id)
            if descriptor is not None:
                if descriptor.is_deprecated:
                    result.add_warning(
                        f"Task '{tid}' uses deprecated capability '{cap_id}': {descriptor.deprecation_reason or 'No reason provided'}"
                    )

                # Validate required parameters if schema provides required list
                req_props = descriptor.parameter_schema.get("required", [])
                params = t_data["parameters"]
                for req in req_props:
                    if req not in params:
                        result.add_error(
                            stage=ValidationStage.CAPABILITY,
                            code="MISSING_REQUIRED_PARAMETER",
                            message=f"Task '{tid}' missing required parameter '{req}' for capability '{cap_id}'.",
                            task_id=tid,
                            details={"parameter": req, "capability_id": cap_id},
                        )

    def _validate_hard_constraints(self, info: Dict[str, Any], result: ValidationResult) -> None:
        """Validate complexity limits such as max tasks count and max DAG depth."""
        tasks = info["tasks"]
        task_count = len(tasks)
        if task_count > self.max_tasks:
            result.add_error(
                stage=ValidationStage.HARD_CONSTRAINTS,
                code="EXCEEDED_MAX_TASKS",
                message=f"Plan task count {task_count} exceeds maximum allowed limit of {self.max_tasks}.",
                details={"task_count": task_count, "max_tasks": self.max_tasks},
            )

        # Compute max DAG depth (longest path)
        adj = info.get("adj", {})
        task_ids = set(tasks.keys())
        memo_depth: Dict[str, int] = {}

        def _calc_depth(node: str) -> int:
            if node in memo_depth:
                return memo_depth[node]
            downstream_nodes = adj.get(node, [])
            if not downstream_nodes:
                memo_depth[node] = 1
                return 1
            max_child_depth = max(_calc_depth(child) for child in downstream_nodes)
            memo_depth[node] = 1 + max_child_depth
            return memo_depth[node]

        max_depth_found = max((_calc_depth(tid) for tid in task_ids), default=0)
        if max_depth_found > self.max_depth:
            result.add_error(
                stage=ValidationStage.HARD_CONSTRAINTS,
                code="EXCEEDED_MAX_DEPTH",
                message=f"Plan DAG depth {max_depth_found} exceeds maximum allowed depth of {self.max_depth}.",
                details={"max_depth_found": max_depth_found, "max_depth_allowed": self.max_depth},
            )

    def _validate_feasibility(
        self,
        info: Dict[str, Any],
        available_task_ids: Set[str],
        available_artifact_ids: Set[str],
        result: ValidationResult,
    ) -> None:
        """Validate data-flow feasibility.
        
        Distinguishes already-available data from execution-produced data.
        Execution-produced references require an ordering path in the DAG.
        """
        tasks = info["tasks"]
        task_ids = set(tasks.keys())
        adj = info.get("adj", {})

        # Helper to check DAG path reachability from src to dst
        reachability_cache: Dict[str, Set[str]] = {}

        def _get_reachable_downstreams(src: str) -> Set[str]:
            if src in reachability_cache:
                return reachability_cache[src]
            reachable = set()
            queue = deque([src])
            while queue:
                curr = queue.popleft()
                for neighbor in adj.get(curr, []):
                    if neighbor not in reachable:
                        reachable.add(neighbor)
                        queue.append(neighbor)
            reachability_cache[src] = reachable
            return reachable

        for tid, t_data in tasks.items():
            for ref in t_data["input_references"]:
                # If reference declares a source_task_id
                src_task_id = getattr(ref, "source_task_id", None)
                if src_task_id:
                    if src_task_id in available_task_ids:
                        # Case 1: Data was produced by an already completed task / prior revision.
                        # No execution ordering edge required in the current plan!
                        continue

                    if src_task_id in task_ids:
                        # Case 2: Data is produced by another task in the current plan.
                        # Execution ordering is strictly required: src_task_id must precede tid.
                        downstreams = _get_reachable_downstreams(src_task_id)
                        if tid not in downstreams:
                            result.add_error(
                                stage=ValidationStage.FEASIBILITY,
                                code="UNORDERED_DATA_REFERENCE",
                                message=(
                                    f"Task '{tid}' references output '{getattr(ref, 'key', 'unknown')}' from "
                                    f"task '{src_task_id}', but '{src_task_id}' does not precede '{tid}' in the execution DAG."
                                ),
                                task_id=tid,
                                details={"source_task_id": src_task_id, "data_key": getattr(ref, "key", None)},
                            )
                    else:
                        # Case 3: Data source task is unknown
                        result.add_error(
                            stage=ValidationStage.FEASIBILITY,
                            code="UNKNOWN_DATA_SOURCE",
                            message=(
                                f"Task '{tid}' references output from unknown task '{src_task_id}' "
                                f"which is not completed and not part of the plan."
                            ),
                            task_id=tid,
                            details={"source_task_id": src_task_id},
                        )
