"""In-memory registry for capability resolution."""

from typing import Dict, List

from orchestration.capabilities.base import Capability
from orchestration.errors import CapabilityNotFoundError, CapabilityRegistryError


class CapabilityRegistry:
    """In-memory catalog mapping capability_id to Capability implementations.

    The registry is solely responsible for registering and resolving capabilities
    by identifier. It has no awareness of tasks, plans, or execution state.
    """

    def __init__(self) -> None:
        self._capabilities: Dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        """Register a capability instance.

        Args:
            capability: Capability satisfying the Capability protocol.

        Raises:
            CapabilityRegistryError: If capability_id is already registered.
        """
        cid = capability.capability_id
        if cid in self._capabilities:
            raise CapabilityRegistryError(
                f"Capability '{cid}' is already registered."
            )
        self._capabilities[cid] = capability

    def get(self, capability_id: str) -> Capability:
        """Resolve a capability by its identifier.

        Args:
            capability_id: Identifier of the capability to look up.

        Returns:
            The registered Capability instance.

        Raises:
            CapabilityNotFoundError: If no capability is registered with this ID.
        """
        if capability_id not in self._capabilities:
            raise CapabilityNotFoundError(
                f"Capability '{capability_id}' not found in registry."
            )
        return self._capabilities[capability_id]

    def has(self, capability_id: str) -> bool:
        """Check if a capability identifier is registered.

        Args:
            capability_id: Identifier to check.

        Returns:
            True if registered, False otherwise.
        """
        return capability_id in self._capabilities

    def list_capabilities(self) -> List[str]:
        """Return a sorted list of all registered capability identifiers."""
        return sorted(self._capabilities.keys())
