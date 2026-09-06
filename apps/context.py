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
from typing import Any, Optional, Union


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

    def create_code_repair_workflow(
        self,
        workspace_capability: Optional[Any] = None,
        model_id: str = "default",
        generator_fn: Optional[Any] = None,
    ) -> Any:
        """Create a CodeTestRepairWorkflow wired to InferenceConnector and WorkspaceCodingCapability."""
        from workflows.code_repair import CodeTestRepairWorkflow

        ws_cap = workspace_capability or self.create_workspace_coding_capability()
        return CodeTestRepairWorkflow(
            connector=self.inference,
            workspace_capability=ws_cap,
            model_id=model_id,
            generator_fn=generator_fn,
        )

    # ------------------------------------------------------------------ #
    # Capability Factories                                              #
    # ------------------------------------------------------------------ #

    def create_document_understanding_capability(
        self,
        parser: Optional[Any] = None,
    ) -> Any:
        """Create a DocumentUnderstandingCapability wired to configured parser."""
        from orchestration.capabilities.builtin.document import (
            DocumentParseOptions,
            DocumentUnderstandingCapability,
        )

        if parser is None:
            doc_settings = getattr(getattr(self.core, "settings", None), "document", None)
            default_parser = getattr(doc_settings, "default_parser", None) if isinstance(getattr(doc_settings, "default_parser", None), str) else "docling"
            if default_parser == "docling":
                try:
                    from orchestration.capabilities.builtin.document.docling_parser import (
                        DoclingDocumentParser,
                    )
                    enable_ocr = getattr(doc_settings, "enable_ocr", True)
                    enable_tables = getattr(doc_settings, "enable_tables", True)
                    enable_figures = getattr(doc_settings, "enable_figures", False)
                    options = DocumentParseOptions(
                        do_ocr=bool(enable_ocr) if isinstance(enable_ocr, bool) else True,
                        extract_tables=bool(enable_tables) if isinstance(enable_tables, bool) else True,
                        extract_figures=bool(enable_figures) if isinstance(enable_figures, bool) else False,
                    )

                    parser = DoclingDocumentParser(options=options)
                except ImportError:
                    from orchestration.capabilities.builtin.document.fallback_parser import (
                        FallbackDocumentParser,
                    )
                    parser = FallbackDocumentParser()
            else:
                from orchestration.capabilities.builtin.document.fallback_parser import (
                    FallbackDocumentParser,
                )
                parser = FallbackDocumentParser()

        return DocumentUnderstandingCapability(parser=parser)

    def create_artifact_generation_capability(
        self,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Create an ArtifactGenerationCapability wired to output storage directory."""
        from orchestration.capabilities.builtin.artifact import (
            ArtifactGenerationCapability,
        )

        art_settings = getattr(getattr(self.core, "settings", None), "artifact", None)
        output_dir_str = getattr(art_settings, "output_dir", None) if isinstance(getattr(art_settings, "output_dir", None), str) else "artifacts"

        repo_root = getattr(self.core, "repo_root", None)
        if not isinstance(repo_root, Path):
            repo_root = Path.cwd()

        out_path = Path(output_dir or output_dir_str)
        if not out_path.is_absolute():
            out_path = repo_root / out_path

        return ArtifactGenerationCapability(output_dir=out_path)

    def create_workspace_coding_capability(
        self,
        executor: Optional[Any] = None,
        workspace_dir: Optional[Union[str, Path]] = None,
        default_executor_type: Optional[str] = None,
    ) -> Any:
        """Create a WorkspaceCodingCapability wired to isolated workspace sandboxing."""
        from orchestration.capabilities.builtin.code import (
            WorkspaceCodingCapability,
        )

        ws_settings = getattr(getattr(self.core, "settings", None), "workspace", None)
        exec_type = default_executor_type or (
            getattr(ws_settings, "default_executor", None)
            if isinstance(getattr(ws_settings, "default_executor", None), str)
            else "docker"
        )
        base_dir = (
            getattr(ws_settings, "base_workspaces_dir", None)
            if isinstance(getattr(ws_settings, "base_workspaces_dir", None), str)
            else ".workspaces"
        )

        repo_root = getattr(self.core, "repo_root", None)
        if not isinstance(repo_root, Path):
            repo_root = Path.cwd()

        ws_path = Path(workspace_dir or base_dir)
        if not ws_path.is_absolute():
            ws_path = repo_root / ws_path

        return WorkspaceCodingCapability(
            executor=executor,
            workspace_dir=ws_path,
            default_executor_type=exec_type,
        )


    def create_vision_inspection_capability(self) -> Any:
        """Create a VisionInspectionCapability wired to this context's InferenceConnector."""
        from orchestration.capabilities.builtin.vision import VisionInspectionCapability

        return VisionInspectionCapability(connector=self.inference)

    def create_code_repair_capability(self, workflow: Optional[Any] = None) -> Any:
        """Create a CodeVerificationRepairCapability wired to CodeTestRepairWorkflow."""
        from orchestration.capabilities.builtin.code_repair import CodeVerificationRepairCapability

        wf = workflow or self.create_code_repair_workflow()
        return CodeVerificationRepairCapability(workflow=wf)

    def create_base_capability_registry(self) -> Any:
        """Create a CapabilityRegistry with base capabilities registered (excluding agent)."""
        from orchestration.capabilities import CapabilityRegistry
        from orchestration.capabilities.builtin import (
            CodeVerificationRepairCapability,
            InferencePromptCapability,
            TextAnalysisCapability,
            VisionInspectionCapability,
        )

        registry = CapabilityRegistry()
        registry.register(InferencePromptCapability(connector=self.inference))
        registry.register(
            TextAnalysisCapability(workflow=self.create_text_analysis_workflow())
        )
        registry.register(self.create_document_understanding_capability())
        registry.register(self.create_artifact_generation_capability())
        registry.register(self.create_workspace_coding_capability())
        registry.register(self.create_vision_inspection_capability())
        registry.register(self.create_code_repair_capability())
        return registry

    def create_agent_capability(
        self,
        registry: Optional[Any] = None,
        model_tier: Optional[Any] = None,
    ) -> Any:
        """Create an AgentCapability wired to FoundationInferenceConnector and capability tools."""
        from orchestration.capabilities.builtin.agent import (
            AgentCapability,
            AgentExecutionPolicy,
            CapabilityToolAdapter,
            FoundationPydanticAIModel,
        )
        from orchestration.routing import ModelSelectionPolicy
        from orchestration.routing.types import ModelTier

        target_registry = registry or self.create_base_capability_registry()
        reg = getattr(self.core, "registry", getattr(self.core, "model_registry", None))
        model_policy = ModelSelectionPolicy(registry=reg)
        model_adapter = FoundationPydanticAIModel(
            connector=self.inference,
            model_policy=model_policy,
            default_tier=model_tier or ModelTier.REASONING,
        )
        auth_policy = AgentExecutionPolicy()
        tool_adapter = CapabilityToolAdapter(
            registry=target_registry,
            policy=auth_policy,
        )
        return AgentCapability(
            model_adapter=model_adapter,
            tool_adapter=tool_adapter,
            policy=auth_policy,
        )

    def create_capability_registry(self) -> Any:
        """Create a standard CapabilityRegistry with all built-in capabilities registered.

        Follows strict acyclic construction order:
          1. Base capabilities constructed and registered into registry.
          2. AgentExecutionPolicy and CapabilityToolAdapter constructed against populated registry.
          3. AgentCapability constructed with model adapter, tool adapter, and policy.
          4. AgentCapability registered into the registry as 'agent.pydantic_ai'.
        """
        registry = self.create_base_capability_registry()
        agent_cap = self.create_agent_capability(registry=registry)
        registry.register(agent_cap)
        return registry

    # ------------------------------------------------------------------ #
    # Orchestration Factories                                            #
    # ------------------------------------------------------------------ #

    def create_in_process_plan_runner(
        self,
        registry: Optional[Any] = None,
    ) -> InProcessPlanRunner:
        """Create an InProcessPlanRunner pre-wired with standard capabilities.

        Args:
            registry: Optional CapabilityRegistry override. Defaults to full standard registry.

        Returns:
            Configured InProcessPlanRunner instance.
        """
        from orchestration.execution import InProcessPlanRunner

        reg = registry or self.create_capability_registry()
        return InProcessPlanRunner(registry=reg)


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
                    name="document_understanding",
                    strategy=ExecutionStrategy.DIRECT_CAPABILITY,
                    target_capability_id="document.understand",
                    utterances=[
                        "parse document",
                        "extract tables from pdf",
                        "read scanned document",
                        "understand document",
                    ],
                ),
                RouteDefinition(
                    name="artifact_generation",
                    strategy=ExecutionStrategy.DIRECT_CAPABILITY,
                    target_capability_id="artifact.generate",
                    utterances=[
                        "generate spreadsheet",
                        "create excel file",
                        "generate pdf report",
                        "create docx document",
                        "create presentation",
                        "generate pptx slides",
                    ],
                ),
                RouteDefinition(
                    name="code_workspace",
                    strategy=ExecutionStrategy.DIRECT_CAPABILITY,
                    target_capability_id="code.workspace",
                    utterances=[
                        "run code in sandbox",
                        "execute command in workspace",
                        "inspect workspace files",
                    ],
                ),
                RouteDefinition(
                    name="vision_inspection",
                    strategy=ExecutionStrategy.DIRECT_CAPABILITY,
                    target_capability_id="vision.inspect",
                    utterances=[
                        "inspect image",
                        "analyze diagram",
                        "read p&id drawing",
                        "inspect drawing",
                    ],
                ),
                RouteDefinition(
                    name="complex_workflow",
                    strategy=ExecutionStrategy.PLAN_REQUIRED,
                    utterances=["create report and summarize", "multi-step pipeline"],
                ),
                RouteDefinition(
                    name="agent_execution",
                    strategy=ExecutionStrategy.DIRECT_CAPABILITY,
                    target_capability_id="agent.pydantic_ai",
                    utterances=[
                        "run agent task",
                        "solve problem using agent tools",
                        "agent investigation",
                    ],
                ),
            ]

        llm_classifier = None
        if enable_llm:
            reg = getattr(self.core, "registry", getattr(self.core, "model_registry", None))
            model_policy = policy or ModelSelectionPolicy(registry=reg)
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
            registry: Optional CapabilityRegistry. Defaults to standard capability registry.
            max_tasks: Maximum task count limit.
            max_depth: Maximum DAG critical path depth.

        Returns:
            Configured PlanValidator instance.
        """
        from orchestration.validation import PlanValidator

        reg = registry or self.create_capability_registry()

        return PlanValidator(
            capability_registry=reg,
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
