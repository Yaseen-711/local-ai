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
