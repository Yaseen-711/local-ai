"""End-to-end integration test for document ingestion, artifact generation, and code execution.

Validates the full capability pipeline:
1. document.understand parses document structure and extracts tabular data.
2. artifact.generate consumes upstream table references and creates an Excel spreadsheet.
3. code.workspace executes an isolated verification command inside the workspace.
4. GoalOrchestrator coordinates the DAG and snapshot-persists to PostgreSQL.
"""

import os
from pathlib import Path
import socket
import pytest

from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.capabilities.builtin.artifact import ArtifactGenerationCapability
from orchestration.capabilities.builtin.code import WorkspaceCodingCapability
from orchestration.capabilities.builtin.document import DocumentUnderstandingCapability
from orchestration.domain.dependencies import Dependency
from orchestration.domain.goals import Goal
from orchestration.domain.plans import Plan
from orchestration.domain.references import DataReference
from orchestration.domain.tasks import Task
from orchestration.domain.types import GoalStatus, PlanStatus, TaskStatus
from orchestration.execution.runner import InProcessPlanRunner
from orchestration.orchestrator import GoalOrchestrator
from orchestration.persistence.engine import create_db_engine, create_session_factory
from orchestration.persistence.models import Base
from orchestration.persistence.repository import PostgresOrchestrationRepository


def is_postgres_available(host: str = "127.0.0.1", port: int = 5432) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


POSTGRES_AVAILABLE = is_postgres_available()
POSTGRES_URL = os.getenv(
    "LOCAL_AI_DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@127.0.0.1:5432/local_ai",
)


@pytest.fixture
def pipeline_env(tmp_path: Path):
    """Set up isolated workspace and artifact directories."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True)
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True)

    # Create sample document with table data
    csv_file = input_dir / "sales_summary.csv"
    csv_file.write_text(
        "Quarter,Region,Revenue,Expenses,NetProfit\n"
        "Q1,North,$500000,$350000,$150000\n"
        "Q2,North,$620000,$400000,$220000\n"
        "Q3,South,$480000,$310000,$170000\n"
        "Q4,South,$710000,$450000,$260000\n"
    )

    return {
        "csv_file": csv_file,
        "artifact_dir": artifact_dir,
        "workspace_dir": workspace_dir,
    }


def test_file_coding_pipeline_e2e(pipeline_env):
    """Verify 3-step DAG: document.understand -> artifact.generate -> code.workspace."""
    csv_file = pipeline_env["csv_file"]
    artifact_dir = pipeline_env["artifact_dir"]
    workspace_dir = pipeline_env["workspace_dir"]

    # 1. Registry setup
    registry = CapabilityRegistry()
    registry.register(DocumentUnderstandingCapability())
    registry.register(ArtifactGenerationCapability(output_dir=artifact_dir))
    registry.register(
        WorkspaceCodingCapability(
            workspace_dir=workspace_dir,
            default_executor_type="local_subprocess",
        )
    )

    runner = InProcessPlanRunner(registry=registry)
    orchestrator = GoalOrchestrator(runner=runner)

    goal = Goal(
        goal_id="goal-pipeline-1",
        description="E2E Document Ingestion, Artifact Generation, and Code Workspace Execution",
    )
    plan = Plan(
        plan_id="plan-pipeline-1",
        goal_id=goal.goal_id,
        title="Document to Code Pipeline Plan",
    )

    # 2. Build DAG Plan
    # Task 1: Parse document
    t1 = Task(
        task_id="task-1-understand",
        plan_id=plan.plan_id,
        title="Parse Sales Summary Document",
        capability_id="document.understand",
        parameters={"file_path": str(csv_file), "force_fallback": True},
    )

    # Task 2: Generate XLSX spreadsheet from extracted tables
    t2 = Task(
        task_id="task-2-generate-xlsx",
        plan_id=plan.plan_id,
        title="Generate Sales Spreadsheet",
        capability_id="artifact.generate",
        parameters={"artifact_type": "xlsx", "filename": "sales_report.xlsx", "title": "FY2025 Sales"},
        input_references={"data": DataReference(key="tables", source_task_id="task-1-understand")},
        dependencies=[Dependency(upstream_task_id="task-1-understand", downstream_task_id="task-2-generate-xlsx")],
    )

    import sys

    # Task 3: Workspace command inspecting generated spreadsheet
    verify_script = (
        f"{sys.executable} -c \""
        f"import openpyxl; "
        f"wb = openpyxl.load_workbook('{artifact_dir / 'sales_report.xlsx'}'); "
        f"ws = wb.active; "
        f"headers = [c.value for c in ws[1]]; "
        f"assert 'Quarter' in headers and 'Revenue' in headers; "
        f"print('Spreadsheet validated successfully: ' + str(headers));"
        f"\""
    )

    t3 = Task(
        task_id="task-3-workspace-verify",
        plan_id=plan.plan_id,
        title="Verify Generated Spreadsheet in Workspace",
        capability_id="code.workspace",
        parameters={"action": "run_command", "command": verify_script},
        dependencies=[Dependency(upstream_task_id="task-2-generate-xlsx", downstream_task_id="task-3-workspace-verify")],
    )

    plan.add_task(t1)
    plan.add_task(t2)
    plan.add_task(t3)

    # 3. Execute via GoalOrchestrator
    orchestrator.execute_goal(goal, plan)

    # 4. Verify Goal and Plan status
    assert goal.status == GoalStatus.COMPLETED
    assert plan.status == PlanStatus.COMPLETED
    assert len(plan.tasks) == 3

    # Task 1 verification
    res1 = plan.tasks["task-1-understand"].result
    assert res1 is not None
    assert "tables" in res1.output
    assert len(res1.output["tables"]) == 1
    assert res1.output["tables"][0]["num_rows"] == 5

    # Task 2 verification
    res2 = plan.tasks["task-2-generate-xlsx"].result
    assert res2 is not None
    assert len(res2.artifacts) == 1
    art = res2.artifacts[0]
    assert art.name == "sales_report.xlsx"
    assert (artifact_dir / "sales_report.xlsx").exists()
    assert art.size_bytes > 0

    # Task 3 verification
    res3 = plan.tasks["task-3-workspace-verify"].result
    assert res3 is not None
    assert res3.output["success"] is True
    assert "Spreadsheet validated successfully" in res3.output["stdout"]



@pytest.mark.integration
@pytest.mark.skipif(not POSTGRES_AVAILABLE, reason="PostgreSQL not reachable on 127.0.0.1:5432")
def test_file_coding_pipeline_with_postgres_persistence(pipeline_env):
    """Verify DAG execution with relational PostgreSQL milestone persistence."""
    csv_file = pipeline_env["csv_file"]
    artifact_dir = pipeline_env["artifact_dir"]
    workspace_dir = pipeline_env["workspace_dir"]

    engine = create_db_engine(POSTGRES_URL)
    Base.metadata.create_all(bind=engine)
    session_factory = create_session_factory(engine)
    repository = PostgresOrchestrationRepository(session_or_factory=session_factory)

    registry = CapabilityRegistry()
    registry.register(DocumentUnderstandingCapability())
    registry.register(ArtifactGenerationCapability(output_dir=artifact_dir))
    registry.register(
        WorkspaceCodingCapability(
            workspace_dir=workspace_dir,
            default_executor_type="local_subprocess",
        )
    )

    runner = InProcessPlanRunner(registry=registry)
    orchestrator = GoalOrchestrator(runner=runner, repository=repository)

    goal = Goal(goal_id="g-pipeline-pg-1", description="PostgreSQL Pipeline Persistence Goal")
    plan = Plan(plan_id="p-pipeline-pg-1", goal_id=goal.goal_id, title="PostgreSQL Pipeline Plan")

    t1 = Task(
        task_id="pg-t1-doc",
        plan_id=plan.plan_id,
        title="Parse Document",
        capability_id="document.understand",
        parameters={"file_path": str(csv_file), "force_fallback": True},
    )
    t2 = Task(
        task_id="pg-t2-art",
        plan_id=plan.plan_id,
        title="Generate Spreadsheet",
        capability_id="artifact.generate",
        parameters={"artifact_type": "xlsx", "filename": "pg_sales.xlsx"},
        input_references={"data": DataReference(key="tables", source_task_id="pg-t1-doc")},
        dependencies=[Dependency(upstream_task_id="pg-t1-doc", downstream_task_id="pg-t2-art")],
    )

    plan.add_task(t1)
    plan.add_task(t2)

    orchestrator.execute_goal(goal, plan)

    assert goal.status == GoalStatus.COMPLETED
    assert plan.status == PlanStatus.COMPLETED

    # Verify repository reload
    loaded_goal = repository.goals.get(goal.goal_id)
    assert loaded_goal is not None
    assert loaded_goal.status == GoalStatus.COMPLETED

    loaded_plan = repository.plans.get(plan.plan_id)
    assert loaded_plan is not None
    assert loaded_plan.status == PlanStatus.COMPLETED
    assert len(loaded_plan.tasks) == 2

    reloaded_t2 = loaded_plan.tasks["pg-t2-art"]
    assert reloaded_t2.status == TaskStatus.COMPLETED
    assert reloaded_t2.result is not None
    assert len(reloaded_t2.result.artifacts) == 1
    assert reloaded_t2.result.artifacts[0].name == "pg_sales.xlsx"

