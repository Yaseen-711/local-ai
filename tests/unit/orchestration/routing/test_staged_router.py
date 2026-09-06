"""Unit tests for staged intent routing and offline semantic matching."""

from core.models.schema import ModelCapabilities, ModelDefinition
from core.models.registry import ModelRegistry
from orchestration.domain.goals import Goal
from orchestration.routing import (
    AurelioSemanticRouter,
    DeterministicHashEncoder,
    DeterministicRuleMatcher,
    ExecutionStrategy,
    ModelSelectionPolicy,
    ModelTier,
    RouteDefinition,
    StagedEscalationRouter,
)


def test_deterministic_matcher_exact_and_prefix():
    routes = [
        RouteDefinition(
            name="system_status",
            strategy=ExecutionStrategy.DIRECT_DETERMINISTIC,
            metadata={"prefixes": ["/status", "status"]},
        ),
        RouteDefinition(
            name="echo",
            strategy=ExecutionStrategy.DIRECT_CAPABILITY,
            target_capability_id="test.echo",
            metadata={"patterns": [r"^echo\s+(?P<message>.+)"]},
        ),
    ]
    matcher = DeterministicRuleMatcher(routes=routes)

    # Prefix match
    g1 = Goal(goal_id="g1", description="/status check")
    r1 = matcher.match(g1)
    assert r1 is not None
    assert r1.route_name == "system_status"
    assert r1.strategy == ExecutionStrategy.DIRECT_DETERMINISTIC
    assert r1.stage_resolved == "deterministic"

    # Regex match with extracted parameters
    g2 = Goal(goal_id="g2", description="echo hello world")
    r2 = matcher.match(g2)
    assert r2 is not None
    assert r2.route_name == "echo"
    assert r2.strategy == ExecutionStrategy.DIRECT_CAPABILITY
    assert r2.extracted_parameters == {"message": "hello world"}

    # No match
    g3 = Goal(goal_id="g3", description="perform text analysis")
    assert matcher.match(g3) is None


def test_deterministic_hash_encoder():
    encoder = DeterministicHashEncoder(dimension=64)
    v1, v2 = encoder.encode(["analyze financial report", "financial quarterly analysis"])
    v3 = encoder.encode(["cook a pizza dinner"])[0]

    # Similar phrases should have higher cosine similarity than unrelated phrases
    def sim(a, b):
        return sum(x * y for x, y in zip(a, b))

    sim_related = sim(v1, v2)
    sim_unrelated = sim(v1, v3)
    assert sim_related > sim_unrelated


def test_semantic_router_offline():
    routes = [
        RouteDefinition(
            name="text_analysis",
            strategy=ExecutionStrategy.PLAN_REQUIRED,
            utterances=[
                "analyze quarterly earnings report",
                "summarize technical document text",
                "extract key metrics from document",
            ],
        ),
        RouteDefinition(
            name="direct_echo",
            strategy=ExecutionStrategy.DIRECT_CAPABILITY,
            target_capability_id="test.echo",
            utterances=["echo this message back", "repeat what I say"],
        ),
    ]
    router = AurelioSemanticRouter(routes=routes, threshold=0.50)

    g1 = Goal(goal_id="g1", description="summarize document text and extract metrics")
    r1 = router.match(g1)
    assert r1 is not None
    assert r1.route_name == "text_analysis"
    assert r1.strategy == ExecutionStrategy.PLAN_REQUIRED
    assert r1.stage_resolved == "semantic_router"


def test_staged_escalation_router():
    routes = [
        RouteDefinition(
            name="system_ping",
            strategy=ExecutionStrategy.DIRECT_DETERMINISTIC,
            metadata={"prefixes": ["ping"]},
        ),
        RouteDefinition(
            name="text_analysis",
            strategy=ExecutionStrategy.PLAN_REQUIRED,
            utterances=["analyze text", "document summary"],
        ),
    ]
    staged = StagedEscalationRouter(routes=routes)

    # Stage 1: Deterministic
    g_det = Goal(goal_id="g1", description="ping server")
    r_det = staged.route(g_det)
    assert r_det.stage_resolved == "deterministic"
    assert r_det.route_name == "system_ping"

    # Stage 2: Semantic Router
    g_sem = Goal(goal_id="g2", description="analyze text and summarize")
    r_sem = staged.route(g_sem)
    assert r_sem.stage_resolved == "semantic_router"
    assert r_sem.route_name == "text_analysis"

    # Fallback: Unrecognized
    g_unknown = Goal(goal_id="g3", description="xyz completely unrelated 123")
    r_fallback = staged.route(g_unknown)
    assert r_fallback.stage_resolved == "fallback"
    assert r_fallback.strategy == ExecutionStrategy.PLAN_REQUIRED


def test_model_selection_policy(tmp_path):
    registry = ModelRegistry(configs_dir=tmp_path, auto_load=False)
    from pathlib import Path
    from core.common.types import ModelFormat

    m_def = ModelDefinition(
        id="qwen-main",
        display_name="Qwen Main",
        format=ModelFormat.GGUF,
        relative_path=Path("dummy/path"),
        supported_providers=["llama_cpp"],
        aliases=["default"],
        capabilities=ModelCapabilities(),
    )
    with registry._lock:
        registry._models["qwen-main"] = m_def
        registry._alias_map["qwen-main"] = "qwen-main"
        registry._alias_map["default"] = "qwen-main"

    policy = ModelSelectionPolicy(
        registry=registry,
        tier_mapping={ModelTier.CAPABLE: "qwen-main"},
    )

    resolved = policy.resolve_model_id(ModelTier.CAPABLE)
    assert resolved == "qwen-main"

    # Fallback to default alias
    resolved_light = policy.resolve_model_id(ModelTier.LIGHTWEIGHT)
    assert resolved_light == "qwen-main"
