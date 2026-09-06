"""Unit tests for CapabilityDescriptor and registry integration."""

from orchestration.capabilities.base import Capability, CapabilityContext
from orchestration.capabilities.descriptor import CapabilityDescriptor
from orchestration.capabilities.registry import CapabilityRegistry
from orchestration.domain.results import TaskResult


class FakeCapability:
    @property
    def capability_id(self) -> str:
        return "fake.test"

    def execute(self, parameters, inputs, context):
        return TaskResult(output={"ok": True})


def test_descriptor_creation():
    desc = CapabilityDescriptor(
        capability_id="test.cap",
        description="Test capability",
        parameter_schema={"param1": {"type": "string", "required": True}},
        input_schema={"in1": {"type": "text"}},
        output_schema={"out1": {"type": "summary"}},
        is_available=True,
    )
    assert desc.capability_id == "test.cap"
    assert desc.is_available is True
    assert "param1" in desc.parameter_schema


def test_registry_descriptor_auto_generated():
    registry = CapabilityRegistry()
    cap = FakeCapability()
    registry.register(cap)

    desc = registry.get_descriptor("fake.test")
    assert desc is not None
    assert desc.capability_id == "fake.test"
    assert desc.is_available is True


def test_registry_explicit_descriptor():
    registry = CapabilityRegistry()
    cap = FakeCapability()
    custom_desc = CapabilityDescriptor(
        capability_id="fake.test",
        description="Custom descriptor",
        parameter_schema={"p": {"type": "int"}},
    )
    registry.register(cap, descriptor=custom_desc)

    desc = registry.get_descriptor("fake.test")
    assert desc is not None
    assert desc.description == "Custom descriptor"
    assert "p" in desc.parameter_schema
