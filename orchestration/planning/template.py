"""Deterministic template-based planner."""

import asyncio
from typing import Callable, Dict, Optional
import uuid

from orchestration.domain.dependencies import Dependency
from orchestration.planning.base import Planner
from orchestration.planning.types import CandidatePlan, CandidateTask, PlanningContext

TemplateFactory = Callable[[PlanningContext], CandidatePlan]


class TemplatePlanner(Planner):
    """Deterministic planner generating CandidatePlans using predefined templates."""

    def __init__(self, templates: Optional[Dict[str, TemplateFactory]] = None) -> None:
        self._templates: Dict[str, TemplateFactory] = templates or {}

    def register_template(self, name: str, factory: TemplateFactory) -> None:
        """Register a template factory function under a name."""
        self._templates[name] = factory

    def plan(self, context: PlanningContext) -> CandidatePlan:
        """Produce candidate plan using a registered template."""
        # 1. Check goal.context for explicit template name
        template_name = getattr(context.goal, "context", {}).get("template")
        if template_name and template_name in self._templates:
            return self._templates[template_name](context)

        # 2. Check context.metadata for template name
        ctx_template = context.metadata.get("template")
        if ctx_template and ctx_template in self._templates:
            return self._templates[ctx_template](context)

        # 3. Default single-task or generic template if 'default' is registered
        if "default" in self._templates:
            return self._templates["default"](context)

        raise ValueError(
            f"No matching plan template found for goal '{context.goal.goal_id}'. "
            f"Available templates: {list(self._templates.keys())}"
        )

    async def plan_async(self, context: PlanningContext) -> CandidatePlan:
        """Asynchronously produce candidate plan."""
        return await asyncio.to_thread(self.plan, context)
