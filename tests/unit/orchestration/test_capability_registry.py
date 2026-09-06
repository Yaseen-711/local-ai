"""Unit tests for CapabilityRegistry."""

import pytest

from orchestration.capabilities.base import CapabilityContext
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.results import TaskResult
from orchestration.errors import CapabilityNotFoundError, CapabilityRegistryError


class DummyCapability:
    def __init__(self, cid: str) -> None:
        self._cid = cid

    @property
    def capability_id(self) -> str:
        return self._cid

    def execute(self, parameters, inputs, context) -> TaskResult:
        return TaskResult(output=self._cid)


def test_registry_register_and_get():
    """Verify registering and retrieving a capability."""
    registry = CapabilityRegistry()
    cap = DummyCapability("test.action")
    registry.register(cap)

    assert registry.has("test.action") is True
    assert registry.get("test.action") is cap


def test_registry_duplicate_registration_raises():
    """Verify registering duplicate capability_id raises CapabilityRegistryError."""
    registry = CapabilityRegistry()
    registry.register(DummyCapability("test.action"))

    with pytest.raises(CapabilityRegistryError, match="already registered"):
        registry.register(DummyCapability("test.action"))


def test_registry_get_unregistered_raises():
    """Verify retrieving unknown capability_id raises CapabilityNotFoundError."""
    registry = CapabilityRegistry()
    with pytest.raises(CapabilityNotFoundError, match="not found in registry"):
        registry.get("nonexistent.capability")


def test_registry_has():
    """Verify has() returns correct boolean."""
    registry = CapabilityRegistry()
    assert registry.has("unknown") is False
    registry.register(DummyCapability("known"))
    assert registry.has("known") is True


def test_registry_list_capabilities():
    """Verify list_capabilities returns sorted IDs."""
    registry = CapabilityRegistry()
    registry.register(DummyCapability("b.cap"))
    registry.register(DummyCapability("a.cap"))
    assert registry.list_capabilities() == ["a.cap", "b.cap"]
