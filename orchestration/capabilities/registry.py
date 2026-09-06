"""In-memory registry for capability resolution and catalog discovery."""

from typing import Dict, List, Optional

from orchestration.capabilities.base import Capability
from orchestration.capabilities.descriptor import CapabilityDescriptor
from orchestration.errors import CapabilityNotFoundError, CapabilityRegistryError


class CapabilityRegistry:
    """In-memory catalog mapping capability_id to Capability implementations and descriptors.

    The registry is solely responsible for registering and resolving capabilities
    by identifier and providing metadata for planning and validation. It has no
    awareness of tasks, plans, or execution state.
    """

    def __init__(self) -> None:
        self._capabilities: Dict[str, Capability] = {}
        self._descriptors: Dict[str, CapabilityDescriptor] = {}

    def register(
        self,
        capability: Capability,
        descriptor: Optional[CapabilityDescriptor] = None,
    ) -> None:
        """Register a capability instance with optional catalog descriptor.

        Args:
            capability: Capability satisfying the Capability protocol.
            descriptor: Optional CapabilityDescriptor with metadata and schemas.

        Raises:
            CapabilityRegistryError: If capability_id is already registered.
        """
        cid = capability.capability_id
        if cid in self._capabilities:
            raise CapabilityRegistryError(
                f"Capability '{cid}' is already registered."
            )
        self._capabilities[cid] = capability
        if descriptor is not None:
            self._descriptors[cid] = descriptor
        elif cid not in self._descriptors:
            if hasattr(capability, "get_descriptor") and callable(capability.get_descriptor):
                self._descriptors[cid] = capability.get_descriptor()
            else:
                # Create a default minimal descriptor
                self._descriptors[cid] = CapabilityDescriptor(
                    capability_id=cid,
                    description=f"Capability {cid}",
                )

    def register_descriptor(self, descriptor: CapabilityDescriptor) -> None:
        """Register or update a descriptor for a capability.

        Args:
            descriptor: CapabilityDescriptor to register.
        """
        self._descriptors[descriptor.capability_id] = descriptor

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

    def get_descriptor(self, capability_id: str) -> Optional[CapabilityDescriptor]:
        """Get the descriptor for a capability if registered."""
        return self._descriptors.get(capability_id)

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

    def list_descriptors(self) -> List[CapabilityDescriptor]:
        """Return descriptors for all registered capabilities."""
        return [self._descriptors[cid] for cid in sorted(self._capabilities.keys()) if cid in self._descriptors]
