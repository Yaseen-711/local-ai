"""Capability layer for Local AI Foundation orchestration.

Provides structural contracts and registry for semantic system functions:
Capability protocol, narrow CapabilityContext, and CapabilityRegistry.
"""

from orchestration.capabilities.base import Capability, CapabilityContext
from orchestration.capabilities.descriptor import CapabilityDescriptor
from orchestration.capabilities.registry import CapabilityRegistry

__all__ = [
    "Capability",
    "CapabilityContext",
    "CapabilityDescriptor",
    "CapabilityRegistry",
]
