"""AppContext – Lightweight Composition Root for Local AI Foundation.

This module provides AppContext, a minimal frozen dataclass that wires
FoundationCore to the InferenceConnector and acts as a typed factory for
all domain workflow instances.

Design rules:
  - core/, connectors/, and workflows/ must NEVER import this module.
    Dependencies point strictly downward only.
  - AppContext does NOT implement domain logic. It only composes and
    provides access to the components that do.
  - Every factory method returns a freshly constructed workflow, allowing
    callers to hold them for a request's lifetime or share them freely.
  - AppContext is framework-agnostic: it works identically in CLI entry
    points (Click/Typer/argparse), HTTP servers (FastAPI/Flask/Starlette),
    UI apps (Streamlit/Gradio), standalone scripts, notebooks, and agent
    systems. No runtime dependency on any of those frameworks is introduced.
  - Path resolution is delegated to FoundationCore.create(), so behaviour
    is portable across Linux, Windows, and macOS without extra work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from connectors import FoundationInferenceConnector, InferenceConnector
from core import FoundationCore
from orchestration import (
    GoalOrchestrator,
    InProcessPlanRunner,
    OrchestrationRepository,
    PlanRunner,
    PostgresOrchestrationRepository,
)
from workflows import TextAnalysisWorkflow


@dataclass(frozen=True)
class AppContext:
    """Process-level composition root for Local AI Foundation applications.

    Holds initialised long-lived singletons (FoundationCore, InferenceConnector)
    and provides typed factory methods for domain workflows.

    Attributes:
        core:       Initialised FoundationCore instance.
        inference:  Ready InferenceConnector wired to core.

    Usage::

        # Bootstrap once per process / server startup:
        ctx = AppContext.create()

        # Obtain a workflow and run it:
        workflow = ctx.create_text_analysis_workflow()
        result   = workflow.analyze("Quarterly earnings report text...")

        # Or substitute a custom connector for testing / alternate providers:
        ctx = AppContext(core=core, inference=my_custom_connector)
    """

    core: FoundationCore
    inference: InferenceConnector

    @classmethod
    def create(
        cls,
        repo_root: Optional[Union[str, Path]] = None,
        configs_dir: Optional[Union[str, Path]] = None,
        settings_path: Optional[Union[str, Path]] = None,
    ) -> "AppContext":
        """Bootstrap the Foundation stack and return a fully wired AppContext.

        Delegates all path resolution to FoundationCore.create(), so portable
        across all operating systems and deployment layouts.

        Args:
            repo_root:     Repository root directory. Defaults to ``Path.cwd()``.
            configs_dir:   Directory containing model TOML config files.
                           Defaults to ``<repo_root>/configs/models/``.
            settings_path: Path to ``settings.toml``.
                           Defaults to ``<repo_root>/configs/settings.toml``.

        Returns:
            Fully initialised ``AppContext`` ready for workflow construction.
        """
        core = FoundationCore.create(
            repo_root=repo_root,
            configs_dir=configs_dir,
            settings_path=settings_path,
        )
        inference = FoundationInferenceConnector(core=core)
        return cls(core=core, inference=inference)

    # ------------------------------------------------------------------ #
    # Domain Workflow Factories                                            #
    # ------------------------------------------------------------------ #

    def create_text_analysis_workflow(self) -> TextAnalysisWorkflow:
        """Create a TextAnalysisWorkflow wired to this context's InferenceConnector.

        Returns a new workflow instance each call. Instances are stateless and
        safe to hold for the lifetime of a request or to reuse across requests.

        Returns:
            Ready-to-use ``TextAnalysisWorkflow``.
        """
        return TextAnalysisWorkflow(inference=self.inference)

    # ------------------------------------------------------------------ #
    # Orchestration Factories                                            #
    # ------------------------------------------------------------------ #

    def create_in_process_plan_runner(self) -> InProcessPlanRunner:
        """Create an InProcessPlanRunner pre-wired with standard capabilities.

        Returns:
            Configured InProcessPlanRunner instance.
        """
        from orchestration.capabilities import CapabilityRegistry
        from orchestration.capabilities.builtin import (
            InferencePromptCapability,
            TextAnalysisCapability,
        )
        from orchestration.execution import InProcessPlanRunner

        registry = CapabilityRegistry()
        registry.register(InferencePromptCapability(connector=self.inference))
        registry.register(
            TextAnalysisCapability(workflow=self.create_text_analysis_workflow())
        )
        return InProcessPlanRunner(registry=registry)

    def create_orchestration_repository(
        self,
        db_url: Optional[str] = None,
    ) -> PostgresOrchestrationRepository:
        """Create a PostgresOrchestrationRepository wired to the configured database.

        Args:
            db_url: Optional database URL override. Defaults to self.core.settings.database.url.

        Returns:
            Configured PostgresOrchestrationRepository instance.
        """
        from orchestration.persistence import (
            PostgresOrchestrationRepository,
            create_db_engine,
            create_session_factory,
        )

        url = db_url or self.core.settings.database.url
        engine = create_db_engine(
            url=url,
            pool_size=self.core.settings.database.pool_size,
            max_overflow=self.core.settings.database.max_overflow,
            echo=self.core.settings.database.echo,
        )
        factory = create_session_factory(engine)
        return PostgresOrchestrationRepository(session_or_factory=factory)

    def create_goal_orchestrator(
        self,
        runner: Optional[PlanRunner] = None,
        repository: Optional[OrchestrationRepository] = None,
    ) -> GoalOrchestrator:
        """Create a GoalOrchestrator wired to a PlanRunner execution boundary and optional repository.

        Args:
            runner: Optional PlanRunner implementation. Defaults to a standard
                InProcessPlanRunner wired with this context's default capabilities.
            repository: Optional OrchestrationRepository implementation.

        Returns:
            Configured GoalOrchestrator instance.
        """
        from orchestration.orchestrator import GoalOrchestrator

        if runner is None:
            runner = self.create_in_process_plan_runner()
        return GoalOrchestrator(runner=runner, repository=repository)

    # ------------------------------------------------------------------ #
    # Decision & Planning Layer Factories                                 #
    # ------------------------------------------------------------------ #

    def create_intent_router(
        self,
        routes: Optional[list] = None,
        enable_llm: bool = False,
        policy: Optional["ModelSelectionPolicy"] = None,
    ) -> "StagedEscalationRouter":
        """Create a StagedEscalationRouter configured with default system routes.

        Args:
            routes: Optional list of RouteDefinitions. If omitted, standard
                system routes are configured.
            enable_llm: If True, attach an LLMIntentClassifier backed by
                ModelSelectionPolicy for Stage 3 & 4 escalation.
            policy: Optional ModelSelectionPolicy instance. Defaults to one
                created using this context's core.model_registry.

        Returns:
            Configured StagedEscalationRouter instance.
        """
        from orchestration.routing import (
            AurelioSemanticRouter,
            DeterministicRuleMatcher,
            ExecutionStrategy,
            LLMIntentClassifier,
            ModelSelectionPolicy,
            RouteDefinition,
            StagedEscalationRouter,
        )

        if routes is None:
            routes = [
                RouteDefinition(
                    name="system_ping",
                    strategy=ExecutionStrategy.DIRECT_DETERMINISTIC,
                    metadata={"prefixes": ["ping", "health"]},
                ),
                RouteDefinition(
                    name="text_analysis",
                    strategy=ExecutionStrategy.DIRECT_CAPABILITY,
                    target_capability_id="workflow.text_analysis",
                    utterances=["analyze text", "extract key points from document"],
                ),
                RouteDefinition(
                    name="complex_workflow",
                    strategy=ExecutionStrategy.PLAN_REQUIRED,
                    utterances=["create report and summarize", "multi-step pipeline"],
                ),
            ]

        llm_classifier = None
        if enable_llm:
            model_policy = policy or ModelSelectionPolicy(registry=self.core.model_registry)
            llm_classifier = LLMIntentClassifier(
                connector=self.inference,
                routes=routes,
                model_selection_policy=model_policy,
            )

        return StagedEscalationRouter(routes=routes, llm_classifier=llm_classifier)

    def create_plan_validator(
        self,
        registry: Optional["CapabilityRegistry"] = None,
        max_tasks: int = 50,
        max_depth: int = 10,
    ) -> "PlanValidator":
        """Create a deterministic 4-stage PlanValidator.

        Args:
            registry: Optional CapabilityRegistry. Defaults to runner registry.
            max_tasks: Maximum task count limit.
            max_depth: Maximum DAG critical path depth.

        Returns:
            Configured PlanValidator instance.
        """
        from orchestration.capabilities import CapabilityRegistry
        from orchestration.capabilities.builtin import (
            InferencePromptCapability,
            TextAnalysisCapability,
        )
        from orchestration.validation import PlanValidator

        if registry is None:
            registry = CapabilityRegistry()
            registry.register(InferencePromptCapability(connector=self.inference))
            registry.register(
                TextAnalysisCapability(workflow=self.create_text_analysis_workflow())
            )

        return PlanValidator(
            capability_registry=registry,
            max_tasks=max_tasks,
            max_depth=max_depth,
        )

    def create_decision_engine(
        self,
        orchestrator: Optional[GoalOrchestrator] = None,
        router: Optional["IntentRouter"] = None,
        planner: Optional["Planner"] = None,
        validator: Optional["PlanValidator"] = None,
    ) -> "DecisionEngine":
        """Create a DecisionEngine composing routing, planning, validation, and orchestration.

        Args:
            orchestrator: Optional GoalOrchestrator instance.
            router: Optional IntentRouter instance.
            planner: Optional Planner instance.
            validator: Optional PlanValidator instance.

        Returns:
            Configured DecisionEngine instance.
        """
        from orchestration.decision import DecisionEngine
        from orchestration.planning import LLMPlanner

        if orchestrator is None:
            orchestrator = self.create_goal_orchestrator()

        if router is None:
            router = self.create_intent_router()

        if validator is None:
            validator = self.create_plan_validator(registry=orchestrator.registry)

        if planner is None:
            planner = LLMPlanner(
                connector=self.inference,
                capability_registry=validator.registry,
            )

        return DecisionEngine(
            router=router,
            orchestrator=orchestrator,
            planner=planner,
            validator=validator,
        )
