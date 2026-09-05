"""Task dependency declaration for orchestration plans.

A Dependency represents a directed edge in the plan's task DAG.
In this phase, the only implemented semantic is simple success dependency:
Task B cannot become READY until Task A has successfully COMPLETED.

The design is extensible for future dependency policies (e.g.
must-finish, data-input) but only the simple success semantic is
active in this slice.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Dependency:
    """Explicit prerequisite link between two tasks in a plan.

    Semantics (this phase): The downstream task cannot become READY
    until the upstream task has reached COMPLETED status. If the
    upstream task fails, is blocked, or is cancelled, the downstream
    task becomes BLOCKED.

    Attributes:
        upstream_task_id: ID of the task that must complete first.
        downstream_task_id: ID of the task that depends on the upstream.
    """
    upstream_task_id: str
    downstream_task_id: str
