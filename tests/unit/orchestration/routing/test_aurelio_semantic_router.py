"""Unit tests for Aurelio Semantic Router seam and adapter (Stage 2 routing).

Verifies:
1. AurelioEncoderAdapter correctly bridges Foundation SemanticRouterEncoder to DenseEncoder.
2. ControlledSemanticEncoder proves exact mathematical threshold cutoff (>= 0.60 matches, < 0.60 returns None).
3. RouteResult preserves RouteDefinition strategy, capability, and model tier authoritatively.
4. Missing Aurelio package raises truthful ImportError (no silent production fallback).
5. Router operates in strict air-gapped offline environment with zero network calls.
6. DeterministicHashEncoder functions as a zero-download offline fixture.
"""

from typing import Any, Dict, List
from unittest.mock import patch
import pytest

from orchestration.domain.goals import Goal
from orchestration.routing.base import SemanticRouterEncoder
from orchestration.routing.encoders import AurelioEncoderAdapter, DeterministicHashEncoder
from orchestration.routing.semantic import AurelioSemanticRouter, HAS_AURELIO
from orchestration.routing.types import ExecutionStrategy, ModelTier, RouteDefinition


class ControlledSemanticEncoder(SemanticRouterEncoder):
    """Deterministic, mathematically known vector encoder for semantic routing tests.

    Emits known orthogonal or near-collinear unit vectors to test similarity scoring
    and thresholding with 100% mathematical precision without relying on lexical hashes.
    """

    def __init__(self, vector_map: Dict[str, List[float]], default_vector: List[float]) -> None:
        self._vector_map = {k.lower().strip(): v for k, v in vector_map.items()}
        self._default_vector = default_vector

    def encode(self, texts: List[str]) -> List[List[float]]:
        results = []
        for t in texts:
            cleaned = t.lower().strip()
            results.append(self._vector_map.get(cleaned, self._default_vector))
        return results


@pytest.fixture
def controlled_encoder() -> ControlledSemanticEncoder:
    # 4D unit vectors
    # Route 1: text_analysis (direction [1, 0, 0, 0])
    # Route 2: code_workspace (direction [0, 1, 0, 0])
    # Route 3: complex_workflow (direction [0, 0, 1, 0])
    # Query 1: high similarity to Route 1 ([0.96, 0.28, 0.0, 0.0] -> cos_sim ~ 0.96)
    # Query 2: high similarity to Route 2 ([0.28, 0.96, 0.0, 0.0] -> cos_sim ~ 0.96)
    # Query 3: sub-threshold to all 3 axes ([0.577, 0.577, 0.577, 0.0] -> cos_sim = 0.577 < 0.60)
    # Out of domain default: orthogonal unit vector [0, 0, 0, 1] -> cos_sim = 0.0 (no division by zero)
    vectors = {
        "test": [1.0, 0.0, 0.0, 0.0],
        # Route 1 utterances
        "summarize financial earnings report": [1.0, 0.0, 0.0, 0.0],
        "extract key metrics from document": [1.0, 0.0, 0.0, 0.0],
        # Route 2 utterances
        "run python code in sandbox": [0.0, 1.0, 0.0, 0.0],
        "execute shell script in workspace": [0.0, 1.0, 0.0, 0.0],
        # Route 3 utterances
        "multi-step pipeline across services": [0.0, 0.0, 1.0, 0.0],
        # Query 1 (matches Route 1)
        "please summarize this report": [0.96, 0.28, 0.0, 0.0],
        # Query 2 (matches Route 2)
        "execute python test script": [0.28, 0.96, 0.0, 0.0],
        # Query 3 (sub-threshold ~0.577 similarity to all routes, strictly < 0.60)
        "ambiguous mixed task": [0.577, 0.577, 0.577, 0.0],
    }
    return ControlledSemanticEncoder(vectors, default_vector=[0.0, 0.0, 0.0, 1.0])


@pytest.fixture
def sample_routes() -> List[RouteDefinition]:
    return [
        RouteDefinition(
            name="text_analysis",
            strategy=ExecutionStrategy.DIRECT_CAPABILITY,
            target_capability_id="workflow.text_analysis",
            target_model_tier=ModelTier.LIGHTWEIGHT,
            utterances=[
                "summarize financial earnings report",
                "extract key metrics from document",
            ],
            description="Text analysis and summarization",
        ),
        RouteDefinition(
            name="code_workspace",
            strategy=ExecutionStrategy.DIRECT_CAPABILITY,
            target_capability_id="code.workspace",
            utterances=[
                "run python code in sandbox",
                "execute shell script in workspace",
            ],
            description="Sandbox code execution",
        ),
        RouteDefinition(
            name="complex_workflow",
            strategy=ExecutionStrategy.PLAN_REQUIRED,
            target_model_tier=ModelTier.REASONING,
            utterances=["multi-step pipeline across services"],
            description="Complex workflow requiring planning",
        ),
    ]


def test_aurelio_encoder_adapter_protocol():
    """Verify AurelioEncoderAdapter satisfies Aurelio DenseEncoder calling conventions."""
    hash_encoder = DeterministicHashEncoder(dimension=16)
    adapter = AurelioEncoderAdapter(inner=hash_encoder, score_threshold=0.60)

    assert adapter.name == "foundation_adapter"
    assert adapter.type == "custom"
    assert adapter.score_threshold == 0.60

    vecs = adapter(["hello world", "test"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 16


def test_controlled_semantic_encoder_exact_matching(controlled_encoder, sample_routes):
    """Verify router matches with exact mathematical precision on controlled vectors."""
    router = AurelioSemanticRouter(
        routes=sample_routes,
        encoder=controlled_encoder,
        threshold=0.60,
    )

    # 1. High similarity match for text_analysis (score ~ 0.96 >= 0.60)
    g1 = Goal(goal_id="g1", description="please summarize this report")
    r1 = router.match(g1)
    assert r1 is not None
    assert r1.route_name == "text_analysis"
    assert r1.strategy == ExecutionStrategy.DIRECT_CAPABILITY
    assert r1.stage_resolved == "semantic_router"
    assert r1.target_capability_id == "workflow.text_analysis"
    assert r1.target_model_tier == ModelTier.LIGHTWEIGHT
    assert r1.confidence >= 0.60

    # 2. High similarity match for code_workspace (score ~ 0.96 >= 0.60)
    g2 = Goal(goal_id="g2", description="execute python test script")
    r2 = router.match(g2)
    assert r2 is not None
    assert r2.route_name == "code_workspace"
    assert r2.strategy == ExecutionStrategy.DIRECT_CAPABILITY
    assert r2.stage_resolved == "semantic_router"
    assert r2.target_capability_id == "code.workspace"
    assert r2.confidence >= 0.60

    # 3. Sub-threshold match returns None (allows Stage 3 escalation)
    g3 = Goal(goal_id="g3", description="ambiguous mixed task")
    r3 = router.match(g3)
    assert r3 is None, f"Expected None for sub-threshold query, got {r3}"

    # 4. Out-of-domain orthogonal query returns None
    g4 = Goal(goal_id="g4", description="completely unknown out of domain query")
    r4 = router.match(g4)
    assert r4 is None


def test_threshold_rejection_boundary(controlled_encoder, sample_routes):
    """Verify that threshold boundary is strictly observed."""
    # When threshold is set high (0.98), even 0.96 similarity is rejected
    strict_router = AurelioSemanticRouter(
        routes=sample_routes,
        encoder=controlled_encoder,
        threshold=0.98,
    )
    g = Goal(goal_id="g1", description="please summarize this report")
    assert strict_router.match(g) is None

    # When threshold is set lower (0.50), it matches
    lenient_router = AurelioSemanticRouter(
        routes=sample_routes,
        encoder=controlled_encoder,
        threshold=0.50,
    )
    res = lenient_router.match(g)
    assert res is not None
    assert res.route_name == "text_analysis"


def test_missing_aurelio_raises_truthful_import_error():
    """Verify truthful capability error when semantic-router package is missing."""
    with patch("orchestration.routing.semantic.HAS_AURELIO", False):
        with pytest.raises(ImportError, match="Aurelio Semantic Router requires 'semantic-router'"):
            AurelioSemanticRouter(routes=[])


def test_route_strategy_integrity(controlled_encoder, sample_routes):
    """Verify RouteDefinition.strategy is authoritative and preserved without mutation."""
    router = AurelioSemanticRouter(
        routes=sample_routes,
        encoder=controlled_encoder,
        threshold=0.60,
    )
    res = router.match(Goal(goal_id="g1", description="please summarize this report"))
    assert res is not None
    # RouteDefinition for text_analysis specifies DIRECT_CAPABILITY
    assert res.strategy == ExecutionStrategy.DIRECT_CAPABILITY
    assert res.target_capability_id == "workflow.text_analysis"


def test_deterministic_hash_encoder_as_zero_download_fixture(sample_routes):
    """Verify DeterministicHashEncoder works as zero-download offline fixture."""
    router = AurelioSemanticRouter(
        routes=sample_routes,
        encoder=DeterministicHashEncoder(dimension=128),
        threshold=0.60,
    )

    # Exact utterance match
    g = Goal(goal_id="g1", description="summarize financial earnings report")
    res = router.match(g)
    assert res is not None
    assert res.route_name == "text_analysis"
    assert res.stage_resolved == "semantic_router"

    # Unrelated query returns None
    g_unrelated = Goal(goal_id="g2", description="xyz completely unrelated 999")
    assert router.match(g_unrelated) is None


def test_air_gapped_offline_operation(controlled_encoder, sample_routes, monkeypatch):
    """Verify router executes with zero network connections."""
    import socket

    def forbidden_connect(*args, **kwargs):
        raise AssertionError("Network connection attempted during air-gapped semantic routing!")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)

    router = AurelioSemanticRouter(
        routes=sample_routes,
        encoder=controlled_encoder,
        threshold=0.60,
    )
    res = router.match(Goal(goal_id="g1", description="please summarize this report"))
    assert res is not None
    assert res.route_name == "text_analysis"
