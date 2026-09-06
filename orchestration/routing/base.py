"""Protocols for intent routers and embedding encoders."""

from typing import List, Protocol, runtime_checkable

from orchestration.domain.goals import Goal
from orchestration.routing.types import RouteResult


@runtime_checkable
class IntentRouter(Protocol):
    """Protocol for intent and strategy classification."""

    def route(self, goal: Goal) -> RouteResult:
        """Classify a Goal into an execution strategy and route."""
        ...

    async def route_async(self, goal: Goal) -> RouteResult:
        """Asynchronously classify a Goal into an execution strategy and route."""
        ...


@runtime_checkable
class SemanticRouterEncoder(Protocol):
    """Protocol for text vector encoding used by semantic routing.

    Keeps the Aurelio Semantic Router integration decoupled from concrete
    embedding libraries, ensuring strict air-gapped isolation.
    """

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into dense embedding vectors."""
        ...
