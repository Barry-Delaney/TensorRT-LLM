# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the per-param pre-reload coverage debt.

``pre_reload_weights`` opens a per-param/unit coverage debt
(``_reload_outstanding``, via ``init_reload_coverage``) BEFORE replacing
every ``rebuild_tensor_metadata`` param with uninitialized ``empty_like``
storage. Deliveries consume exactly the units they rewrote
(``consume_reload_coverage``); ``post_load_weights`` /
``process_weights_after_loading`` refuse while any unit is outstanding, so
a module destroyed that way can never silently serve garbage -- including
after a weight-only resend that skipped scales, an expert-subset bucket,
or a mid-walk failure. Also covers the mutation-free
``check_reload_capability`` preflight hooks.
"""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.modules.fused_moe.interface import MoE, MoEWeightLoadingMode
from tensorrt_llm._torch.modules.fused_moe.quantization import (
    DeepSeekFP8BlockScalesFusedMoEMethodDeepGemm,
    FusedMoEMethodBase,
    NVFP4FusedMoEMethod,
    UnquantizedFusedMoEMethod,
    W4A8MXFP4MXFP8MegaMoEDeepGemmMethod,
)
from tensorrt_llm._torch.modules.linear import (
    FP8BlockScalesLinearMethod,
    FP8QDQLinearMethod,
    Linear,
    LinearMethodBase,
    NVFP4LinearMethod,
    UnquantizedLinearMethod,
    WeightMode,
    WeightsLoadingConfig,
    init_reload_coverage,
    raise_on_reload_invalidated_module,
)

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="pre_reload re-registers params on CUDA"
)


def _outstanding(module):
    return getattr(module, "_reload_outstanding", None)


def _arm_linear(lin, params=("weight",)):
    """Simulate LinearMethodBase.pre_reload_weights' coverage init without
    touching CUDA (vanilla: one '*' unit per param)."""
    init_reload_coverage(lin, {p: {"*"} for p in params})


def _make_linear_stub(out_features=4, in_features=8):
    """Bare Linear carrying only the reload-coverage surface (the real
    __init__ pulls mapping/allreduce plumbing)."""
    lin = object.__new__(Linear)
    torch.nn.Module.__init__(lin)
    lin.rebuild_tensor_metadata = {}
    lin.quant_config = None
    lin.quant_method = UnquantizedLinearMethod()
    lin.weights_loading_config = WeightsLoadingConfig()
    lin._weights_created = True
    lin._weights_transformed = False
    lin.tp_size = 1
    lin.tp_rank = 0
    lin.tp_mode = None
    lin.weight = torch.nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
    lin.register_parameter("bias", None)
    return lin


def test_linear_pre_reload_empty_metadata_sets_no_marker():
    lin = _make_linear_stub()
    lin.pre_reload_weights()  # nothing registered -> nothing destroyed
    assert not _outstanding(lin)
    lin.post_load_weights()  # and finalize stays legal


def test_linear_post_load_refuses_invalidated_module():
    lin = _make_linear_stub()
    lin.rebuild_tensor_metadata = {"weight": {"meta": torch.empty(4, 8)}}
    _arm_linear(lin)  # as set by pre_reload_weights
    with pytest.raises(RuntimeError, match="uninitialized memory") as excinfo:
        lin.post_load_weights()
    assert "weight" in str(excinfo.value)  # names the outstanding params
    assert lin._weights_transformed is False  # refused BEFORE transforming

    # The RLHF finalize walk runs process_weights_after_loading BEFORE
    # post_load_weights: the veto must fire there too, before any device
    # work touches the garbage bytes.
    with pytest.raises(RuntimeError, match="uninitialized memory"):
        lin.process_weights_after_loading()

    lin._reload_outstanding = {}
    lin.post_load_weights()
    assert lin._weights_transformed is True


def test_linear_marker_survives_quant_load_exception():
    """An exception inside quant_method.load_weights must leave the debt
    outstanding: the empty_like garbage was NOT rewritten, so a later
    finalize has to keep vetoing it (fail-closed, not fail-open)."""

    class _BoomMethod(UnquantizedLinearMethod):
        def load_weights(self, module, weights, weight_mode, allow_partial_loading=False):
            raise RuntimeError("mid-load boom")

    lin = _make_linear_stub()
    lin.quant_method = _BoomMethod()
    lin.rebuild_tensor_metadata = {"weight": {"meta": torch.empty(4, 8)}}
    _arm_linear(lin)  # as set by pre_reload_weights

    with pytest.raises(RuntimeError, match="mid-load boom"):
        lin.load_weights([{"weight": torch.randn(4, 8)}])
    assert _outstanding(lin) == {"weight": {"*"}}
    with pytest.raises(RuntimeError, match="uninitialized memory"):
        lin.post_load_weights()


def test_linear_empty_bucket_call_leaves_marker_set():
    """The generic fused params_map loader calls
    ``load_weights([{}], allow_partial_loading=True)`` on EVERY bucket of a
    sweep; a call that delivered no bytes must not vacuously consume any
    coverage (the most common transformed modules -- qkv_proj/gate_up_proj
    -- would otherwise lose the guarantee on the first bucket)."""
    lin = _make_linear_stub()
    lin.rebuild_tensor_metadata = {"weight": {"meta": torch.empty(4, 8)}}
    _arm_linear(lin)  # as set by pre_reload_weights

    lin.load_weights([{}], allow_partial_loading=True)
    assert _outstanding(lin) == {"weight": {"*"}}
    # The transform guard reset is deliberately NOT delivery-gated
    # (pre-existing contract; the coverage veto precedes any re-transform).
    assert lin._weights_transformed is False
    with pytest.raises(RuntimeError, match="uninitialized memory"):
        lin.post_load_weights()


def test_linear_pre_reload_marker_set_before_destruction(monkeypatch):
    """The debt opens BEFORE the destructive loop: a mid-loop allocation
    failure (second param) must leave BOTH params vetoed, not half-marked
    -- exactly the mid-walk CUDA OOM shape the abort boundary cannot see."""
    lin = _make_linear_stub()
    lin.weight_scale = torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False)
    lin.rebuild_tensor_metadata = {
        "weight": {"meta": torch.empty(4, 8)},
        "weight_scale": {"meta": torch.empty(1, 1)},
    }

    calls = {"n": 0}
    real_empty_like = torch.empty_like

    def flaky_empty_like(t, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("CUDA out of memory (simulated)")
        kwargs.pop("device", None)  # keep the first alloc CPU-side
        return real_empty_like(t, **kwargs)

    monkeypatch.setattr(torch, "empty_like", flaky_empty_like)
    with pytest.raises(RuntimeError, match="out of memory"):
        lin.pre_reload_weights()
    # Full debt recorded despite the mid-loop failure.
    assert set(_outstanding(lin)) == {"weight", "weight_scale"}
    with pytest.raises(RuntimeError, match="uninitialized memory") as excinfo:
        lin.post_load_weights()
    assert "weight_scale" in str(excinfo.value)


@cuda_required
def test_linear_marker_lifecycle_set_by_pre_reload_cleared_by_load():
    lin = _make_linear_stub()
    lin.rebuild_tensor_metadata = {"weight": {"meta": torch.empty(4, 8, dtype=torch.float32)}}

    lin.pre_reload_weights()
    assert _outstanding(lin) == {"weight": {"*"}}
    assert lin.weight.is_cuda  # re-registered as uninitialized CUDA storage
    with pytest.raises(RuntimeError, match="uninitialized memory"):
        lin.post_load_weights()

    fresh = torch.randn(4, 8)
    lin.load_weights([{"weight": fresh}])
    assert not _outstanding(lin)
    assert torch.equal(lin.weight.data.cpu(), fresh)
    lin.post_load_weights()  # no raise once the load rewrote the module


@cuda_required
def test_linear_initial_full_load_never_carries_marker():
    lin = _make_linear_stub()
    lin.load_weights([{"weight": torch.randn(4, 8)}])
    assert not _outstanding(lin)
    lin.post_load_weights()


@cuda_required
def test_linear_partial_weight_delivery_consumes_only_weight_unit():
    """A partial vanilla delivery of 'weight' consumes exactly the weight
    unit and leaves other armed params outstanding."""
    lin = _make_linear_stub()
    lin.weight_scale = torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False)
    lin.rebuild_tensor_metadata = {
        "weight": {"meta": torch.empty(4, 8)},
        "weight_scale": {"meta": torch.empty(1, 1)},
    }
    _arm_linear(lin, params=("weight", "weight_scale"))

    lin.load_weights([{"weight": torch.randn(4, 8)}], allow_partial_loading=True)
    assert _outstanding(lin) == {"weight_scale": {"*"}}
    with pytest.raises(RuntimeError, match="uninitialized memory"):
        lin.post_load_weights()


@cuda_required
def test_linear_fp8_weight_only_resend_leaves_scale_outstanding():
    """The FP8-blockwise transform registers BOTH weight and weight_scale
    for rebuild; a weight-only resend must leave 'weight_scale'
    outstanding (finalize would otherwise resmooth empty_like garbage
    scales into silently wrong weights)."""
    lin = _make_linear_stub(out_features=4, in_features=8)
    lin.quant_method = FP8BlockScalesLinearMethod()
    lin.weight_scale = torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False)
    lin.input_scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
    lin.inv_input_scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
    lin.rebuild_tensor_metadata = {
        "weight": {"meta": torch.empty(4, 8)},
        "weight_scale": {"meta": torch.empty(1, 1)},
    }
    _arm_linear(lin, params=("weight", "weight_scale"))

    lin.load_weights([{"weight": torch.randn(4, 8)}], allow_partial_loading=True)
    assert _outstanding(lin) == {"weight_scale": {"*"}}
    with pytest.raises(RuntimeError, match="uninitialized memory") as excinfo:
        lin.process_weights_after_loading()
    assert "weight_scale" in str(excinfo.value)

    # The covering scale resend (DS 'weight_scale_inv' spelling) clears it.
    lin.load_weights([{"weight_scale_inv": torch.full((1, 1), 0.5)}], allow_partial_loading=True)
    assert not _outstanding(lin)
    raise_on_reload_invalidated_module(lin, "Linear")  # no veto left


@cuda_required
def test_linear_fused_one_sided_buckets_accumulate():
    """One-sided fused gate_up buckets are legal and must ACCUMULATE:
    'weight' clears only after BOTH shard units arrived (guards against a
    per-call intersection regression that would deadlock cross-bucket
    fused deliveries)."""
    lin = _make_linear_stub(out_features=4, in_features=8)
    lin.weights_loading_config = WeightsLoadingConfig(weight_mode=WeightMode.FUSED_GATE_UP_LINEAR)
    lin.fused_weight_shard_indices_mapping = {"gate": (0, 2), "up": (2, 2)}
    lin.rebuild_tensor_metadata = {"weight": {"meta": torch.empty(4, 8)}}
    init_reload_coverage(lin, lin.quant_method.reload_coverage_units(lin))
    assert _outstanding(lin) == {"weight": {"gate", "up"}}

    lin.load_weights([{"weight": torch.randn(2, 8)}, {}], allow_partial_loading=True)
    assert _outstanding(lin) == {"weight": {"up"}}
    with pytest.raises(RuntimeError, match="uninitialized memory"):
        lin.post_load_weights()

    lin.load_weights([{}, {"weight": torch.randn(2, 8)}], allow_partial_loading=True)
    assert not _outstanding(lin)
    lin.post_load_weights()


class _NoPartialLinearMethod(LinearMethodBase):
    def create_weights(self, *args, **kwargs):  # pragma: no cover
        pass

    def apply(self, *args, **kwargs):  # pragma: no cover
        pass

    def load_weights_vanilla(self, *args, **kwargs):  # pragma: no cover
        pass

    def load_weights_fused_qkv_linear(self, *args, **kwargs):  # pragma: no cover
        pass

    def load_weights_fused_gate_up_linear(self, *args, **kwargs):  # pragma: no cover
        pass


def test_linear_check_reload_capability_preflight():
    """Preflight: partial-capable quant methods pass; a non-capable method
    refuses unconditionally BEFORE any mutation; the load-time assert stays
    as backstop."""
    lin = _make_linear_stub()
    lin.rebuild_tensor_metadata = {"weight": {"meta": torch.empty(4, 8)}}
    lin.check_reload_capability()  # Unquantized: capable

    for method_cls in (FP8BlockScalesLinearMethod, NVFP4LinearMethod):
        lin.quant_method = object.__new__(method_cls)
        lin.check_reload_capability()

    lin.quant_method = _NoPartialLinearMethod()
    before_params = dict(lin.named_parameters())
    with pytest.raises(NotImplementedError, match="_NoPartialLinearMethod"):
        lin.check_reload_capability()
    # Side-effect-free: no param touched, no debt opened.
    assert dict(lin.named_parameters()) == before_params
    assert _outstanding(lin) is None

    # Backstop still present: bypassing the preflight hits the load assert.
    with pytest.raises(AssertionError, match="allow_partial_loading"):
        lin.load_weights([{"weight": torch.randn(4, 8)}], allow_partial_loading=True)


# --------------------------------------------------------------------------
# MoE
# --------------------------------------------------------------------------


def _make_moe_module(
    rebuild_params,
    expert_ids=(0, 1),
    mode=MoEWeightLoadingMode.VANILLA,
):
    module = torch.nn.Module()
    module.initial_local_expert_ids = list(expert_ids)
    module.weight_loading_mode = mode
    module.bias = False
    module.w3_w1_weight = torch.nn.Parameter(torch.zeros(1, 2, 2), requires_grad=False)
    module.w2_weight = torch.nn.Parameter(torch.zeros(1, 2, 2), requires_grad=False)
    module._weights_transformed = True
    module.rebuild_tensor_metadata = {
        p: {"meta": torch.empty(1, 2, 2, dtype=torch.float32)} for p in rebuild_params
    }
    return module


class _NoopLoadMixin:
    """Skip real tensor IO: these tests exercise ONLY the coverage
    accounting around FusedMoEMethodBase.load_weights."""

    def load_expert_weights_to_dst(
        self,
        module,
        weights,
        weight_loading_mode,
        expert_ids,
        dst_w3_w1,
        dst_w2,
        dst_w3_w1_bias,
        dst_w2_bias,
        allow_partial_loading=False,
    ):
        pass

    def load_quant_scales(self, module, weights):
        pass

    def need_load_shared_weights(self, module):
        return False


class _NoopMethod(_NoopLoadMixin, UnquantizedFusedMoEMethod):
    pass


class _NoopDeepGemmMethod(_NoopLoadMixin, DeepSeekFP8BlockScalesFusedMoEMethodDeepGemm):
    pass


@cuda_required
def test_fused_moe_method_pre_reload_sets_marker():
    class _Method(FusedMoEMethodBase):
        def setup_quant_scales(self, module):  # pragma: no cover - unused
            pass

    method = _Method.__new__(_Method)
    module = _make_moe_module(["w3_w1_weight"], expert_ids=(0,))
    method.pre_reload_weights(module)
    assert module._reload_outstanding == {"w3_w1_weight": {("w1", 0), ("w3", 0)}}
    assert module.w3_w1_weight.is_cuda

    empty = _make_moe_module([], expert_ids=(0,))
    method.pre_reload_weights(empty)
    assert not _outstanding(empty)


def test_moe_pre_reload_mid_loop_failure_leaves_set_populated(monkeypatch):
    """A mid-loop failure in the MoE pre_reload walk must leave the FULL
    debt recorded (init runs before the destructive loop)."""

    class _Method(FusedMoEMethodBase):
        def setup_quant_scales(self, module):  # pragma: no cover - unused
            pass

    method = _Method.__new__(_Method)
    module = _make_moe_module(["w3_w1_weight", "w2_weight"], expert_ids=(0,))

    calls = {"n": 0}
    real_empty_like = torch.empty_like

    def flaky_empty_like(t, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("CUDA out of memory (simulated)")
        kwargs.pop("device", None)
        return real_empty_like(t, **kwargs)

    monkeypatch.setattr(torch, "empty_like", flaky_empty_like)
    with pytest.raises(RuntimeError, match="out of memory"):
        method.pre_reload_weights(module)
    assert set(_outstanding(module)) == {"w3_w1_weight", "w2_weight"}
    with pytest.raises(RuntimeError, match="uninitialized memory"):
        MoE.post_load_weights(module)


def test_moe_post_load_refuses_invalidated_module():
    stub = torch.nn.Module()
    stub._reload_outstanding = {"w2_weight": {("w2", 0)}}
    calls = []
    stub.transform_weights = lambda: calls.append("transform")
    stub.cache_derived_state = lambda: calls.append("cache")
    with pytest.raises(RuntimeError, match="uninitialized memory") as excinfo:
        MoE.post_load_weights(stub)
    assert "w2_weight" in str(excinfo.value)  # names the outstanding params
    assert calls == []  # refused BEFORE any transform ran

    # PWAL runs first in the RLHF finalize walk: same veto there.
    with pytest.raises(RuntimeError, match="uninitialized memory"):
        MoE.process_weights_after_loading(stub)

    stub._reload_outstanding = {}
    MoE.post_load_weights(stub)  # cleared -> transform/cache run normally
    assert calls == ["transform", "cache"]
    stub.quant_method = SimpleNamespace()  # no PWAL hook -> clean no-op
    MoE.process_weights_after_loading(stub)


def test_fused_moe_empty_bucket_call_leaves_marker_set():
    """A ``{}`` bucket delivered nothing: the base MoE load must neither
    reset the transform guard nor consume any coverage; a real delivery
    consumes exactly its units (mirrors the Linear delivery gate)."""
    module = _make_moe_module(["w2_weight"], expert_ids=(0,))
    init_reload_coverage(module, {"w2_weight": {("w2", 0)}})

    method = _NoopMethod.__new__(_NoopMethod)
    method.load_weights(module, {}, MoEWeightLoadingMode.VANILLA, allow_partial_loading=True)
    assert _outstanding(module) == {"w2_weight": {("w2", 0)}}  # nothing delivered
    assert module._weights_transformed is True

    method.load_weights(
        module,
        {"0.w2.weight": torch.zeros(1)},
        MoEWeightLoadingMode.VANILLA,
        allow_partial_loading=True,
    )
    assert not _outstanding(module)  # real delivery consumed the unit
    assert module._weights_transformed is False


def test_moe_weights_only_resend_leaves_scales_outstanding():
    """DeepGemm-shape transform registers ONLY the two scaling factors
    for rebuild; a pure-weight resend (no *.weight_scale_inv keys) must
    leave BOTH scale params outstanding instead of blessing garbage
    scales."""
    module = _make_moe_module(
        ["w3_w1_weight_scaling_factor", "w2_weight_scaling_factor"], expert_ids=(0, 1)
    )
    method = _NoopDeepGemmMethod.__new__(_NoopDeepGemmMethod)
    init_reload_coverage(module, method.reload_coverage_units(module))

    weights = {f"{e}.w{r}.weight": torch.zeros(1) for e in (0, 1) for r in (1, 2, 3)}
    method.load_weights(module, weights, MoEWeightLoadingMode.VANILLA, allow_partial_loading=True)
    assert set(_outstanding(module)) == {
        "w3_w1_weight_scaling_factor",
        "w2_weight_scaling_factor",
    }
    with pytest.raises(RuntimeError, match="uninitialized memory") as excinfo:
        MoE.process_weights_after_loading(module)
    assert "w3_w1_weight_scaling_factor" in str(excinfo.value)

    scales = {f"{e}.w{r}.weight_scale_inv": torch.zeros(1) for e in (0, 1) for r in (1, 2, 3)}
    method.load_weights(module, scales, MoEWeightLoadingMode.VANILLA, allow_partial_loading=True)
    assert not _outstanding(module)


def test_moe_expert_subset_keeps_rows_outstanding():
    """An expert-subset bucket consumes only the delivered experts' units;
    the missing experts keep the veto until covered."""
    module = _make_moe_module(["w3_w1_weight", "w2_weight"], expert_ids=(0, 1))
    method = _NoopMethod.__new__(_NoopMethod)
    init_reload_coverage(module, method.reload_coverage_units(module))

    expert0 = {f"0.w{r}.weight": torch.zeros(1) for r in (1, 2, 3)}
    method.load_weights(module, expert0, MoEWeightLoadingMode.VANILLA, allow_partial_loading=True)
    assert _outstanding(module) == {
        "w3_w1_weight": {("w1", 1), ("w3", 1)},
        "w2_weight": {("w2", 1)},
    }
    with pytest.raises(RuntimeError, match="uninitialized memory"):
        MoE.post_load_weights(module)

    expert1 = {f"1.w{r}.weight": torch.zeros(1) for r in (1, 2, 3)}
    method.load_weights(module, expert1, MoEWeightLoadingMode.VANILLA, allow_partial_loading=True)
    assert not _outstanding(module)


def test_moe_fused_gate_up_proj_single_key_covers_param():
    """FUSED_GATE_UP_PROJ keys carry every expert's rows: one key covers
    the whole param."""
    module = _make_moe_module(
        ["w3_w1_weight", "w2_weight"],
        expert_ids=(0, 1),
        mode=MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
    )
    method = _NoopMethod.__new__(_NoopMethod)
    init_reload_coverage(module, method.reload_coverage_units(module))
    assert _outstanding(module) == {
        "w3_w1_weight": {"gate_up_proj"},
        "w2_weight": {"down_proj"},
    }

    method.load_weights(
        module,
        {"gate_up_proj": torch.zeros(1)},
        MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
        allow_partial_loading=True,
    )
    assert _outstanding(module) == {"w2_weight": {"down_proj"}}

    method.load_weights(
        module,
        {"down_proj": torch.zeros(1)},
        MoEWeightLoadingMode.FUSED_GATE_UP_PROJ,
        allow_partial_loading=True,
    )
    assert not _outstanding(module)


def test_moe_unknown_rebuild_param_gets_unconsumable_sentinel():
    """A rebuild-registered param with no delivery schema fails closed: its
    sentinel unit survives every partial delivery (only a full load or a
    direct-writing loader clears it)."""
    module = _make_moe_module(["mega_derived_param"], expert_ids=(0,))
    method = _NoopMethod.__new__(_NoopMethod)
    init_reload_coverage(module, method.reload_coverage_units(module))

    everything = {
        f"0.w{r}.{s}": torch.zeros(1) for r in (1, 2, 3) for s in ("weight", "weight_scale_inv")
    }
    method.load_weights(
        module, everything, MoEWeightLoadingMode.VANILLA, allow_partial_loading=True
    )
    assert set(_outstanding(module)) == {"mega_derived_param"}

    # A full (non-partial) load is the sanctioned clear.
    method.load_weights(module, everything, MoEWeightLoadingMode.VANILLA)
    assert not _outstanding(module)


def test_moe_check_reload_capability_preflight():
    """Preflight: EPLB refuses; non-partial-capable quant methods refuse
    unconditionally; both in-path backstops stay present."""
    stub = torch.nn.Module()
    stub.layer_load_balancer = object()
    stub._using_load_balancer = lambda: stub.layer_load_balancer is not None
    stub.rebuild_tensor_metadata = {"w2_weight": {"meta": None}}
    stub.quant_method = UnquantizedFusedMoEMethod.__new__(UnquantizedFusedMoEMethod)

    with pytest.raises(NotImplementedError, match="EPLB"):
        MoE.check_reload_capability(stub)
    # Backstop: the destructive hook still refuses on its own.
    with pytest.raises(NotImplementedError, match="EPLB"):
        MoE.pre_reload_weights(stub)

    stub.layer_load_balancer = None
    MoE.check_reload_capability(stub)  # capable quant method passes

    stub.quant_method = W4A8MXFP4MXFP8MegaMoEDeepGemmMethod.__new__(
        W4A8MXFP4MXFP8MegaMoEDeepGemmMethod
    )
    with pytest.raises(NotImplementedError, match="Partial loading"):
        MoE.check_reload_capability(stub)
    # Backstop: the load-time refusal still fires when preflight is skipped.
    with pytest.raises(NotImplementedError, match="Partial loading"):
        stub.quant_method.load_weights(stub, [{}], allow_partial_loading=True)


def test_moe_gateless_module_has_no_w3_debt_and_clears_on_w1_w2():
    """Gateless models (is_gated_activation False) have no w3 checkpoint
    keys: the debt must not contain phantom ('w3', e) units (they could
    never be consumed -> covering resends vetoed forever), and a
    w1+w2(+bias) delivery fully clears the module. Debt and credit filter
    symmetrically."""
    module = _make_moe_module(
        ["w3_w1_weight", "w2_weight", "w3_w1_bias", "w2_bias"], expert_ids=(0, 1)
    )
    module.is_gated_activation = False
    method = _NoopMethod.__new__(_NoopMethod)
    init_reload_coverage(module, method.reload_coverage_units(module))
    assert _outstanding(module) == {
        "w3_w1_weight": {("w1", 0), ("w1", 1)},
        "w2_weight": {("w2", 0), ("w2", 1)},
        "w3_w1_bias": {("w1", 0), ("w1", 1)},
        "w2_bias": {("w2", 0), ("w2", 1)},
    }

    delivery = {
        f"{e}.w{r}.{s}": torch.zeros(1) for e in (0, 1) for r in (1, 2) for s in ("weight", "bias")
    }
    method.load_weights(module, delivery, MoEWeightLoadingMode.VANILLA, allow_partial_loading=True)
    assert not _outstanding(module)
    # No veto left: the gateless resend covers (finalize hooks would pass).
    raise_on_reload_invalidated_module(module, "MoE")


def test_moe_gated_module_keeps_w3_debt():
    """Control for the gateless filter: a gated module (default) still
    demands the w3 units, so a w1+w2-only resend keeps the veto."""
    module = _make_moe_module(["w3_w1_weight"], expert_ids=(0,))
    method = _NoopMethod.__new__(_NoopMethod)
    init_reload_coverage(module, method.reload_coverage_units(module))

    delivery = {"0.w1.weight": torch.zeros(1), "0.w2.weight": torch.zeros(1)}
    method.load_weights(module, delivery, MoEWeightLoadingMode.VANILLA, allow_partial_loading=True)
    assert _outstanding(module) == {"w3_w1_weight": {("w3", 0)}}


def test_moe_reload_covered_units_asserts_mode_drift():
    """The credit side must key off module.weight_loading_mode (same as
    the debt side); a caller-passed divergent mode is a bug and asserts."""
    module = _make_moe_module(["w2_weight"], expert_ids=(0,))  # VANILLA
    method = _NoopMethod.__new__(_NoopMethod)
    with pytest.raises(AssertionError, match="mode drift"):
        method.reload_covered_units(module, {}, MoEWeightLoadingMode.FUSED_GATE_UP_PROJ)


# --------------------------------------------------------------------------
# Abort/retry: pre_reload must purge the FAILED cycle's transient scale
# accumulators (tmp_* stashes PWAL normally consumes-and-deletes), so a
# recovery cycle's finalize uses ONLY its own deliveries -- no cross-cycle
# list/dict may surface the failed cycle's values.
# --------------------------------------------------------------------------


def test_linear_fp8qdq_abort_then_recover_finalizes_only_new_scales():
    """Failed cycle A stashes input/weight/k/v scales on a fused-QKV FP8QDQ
    Linear; the fresh cycle's pre_reload purges them (incl. the
    has_static_input_scale latch); cycle B's finalize then reflects ONLY
    B's deliveries. Without the purge: tmp_k/v_scales EXTEND, so PWAL's
    max() would pick A's larger kv scales, and A's stale static-input
    latch would resurrect A's input_scale."""
    lin = _make_linear_stub()
    lin.quant_method = FP8QDQLinearMethod()
    lin.weights_loading_config = WeightsLoadingConfig(weight_mode=WeightMode.FUSED_QKV_LINEAR)
    lin.fused_weight_shard_indices_mapping = {"q": (0, 4), "k": (4, 4), "v": (8, 4)}
    lin.dtype = torch.float32
    lin.weight = torch.nn.Parameter(
        torch.zeros(12, 8, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    for name in ("weight_scale", "input_scale", "inv_input_scale"):
        setattr(lin, name, torch.nn.Parameter(torch.tensor(1.0), requires_grad=False))
    lin.kv_scales = torch.nn.Parameter(torch.ones(3), requires_grad=False)
    lin.inv_kv_scales = torch.nn.Parameter(torch.ones(3), requires_grad=False)

    # Cycle A (aborts after this bucket): static input scales + kv scales.
    def _shard_a(extra):
        return {"input_scale": torch.tensor(2.0), "weight_scale": torch.tensor(2.0), **extra}

    lin.load_weights(
        [
            _shard_a({"k_scale": torch.tensor(7.0), "v_scale": torch.tensor(9.0)}),
            _shard_a({}),
            _shard_a({}),
        ],
        allow_partial_loading=True,
    )
    assert [s.item() for s in lin.tmp_k_scales] == [7.0]
    assert hasattr(lin, "has_static_input_scale")

    # Fresh recovery cycle: purge everything the failed cycle stashed.
    lin.pre_reload_weights()
    for attr in FP8QDQLinearMethod._RELOAD_TRANSIENT_ATTRS:
        assert not hasattr(lin, attr), attr

    # Cycle B: dynamic-quant checkpoint (weight_scale only) + smaller kv.
    lin.load_weights(
        [
            {"weight_scale": torch.tensor(0.5)},
            {"weight_scale": torch.tensor(0.5), "k_scale": torch.tensor(0.5)},
            {"weight_scale": torch.tensor(0.5), "v_scale": torch.tensor(0.25)},
        ],
        allow_partial_loading=True,
    )
    lin.process_weights_after_loading()

    assert lin.input_scale is None  # A's static latch must not resurrect
    assert lin.weight_scale.item() == 0.5
    assert torch.equal(lin.kv_scales.data, torch.tensor([1.0, 0.5, 0.25]))  # B only
    for attr in FP8QDQLinearMethod._RELOAD_TRANSIENT_ATTRS:
        assert not hasattr(lin, attr), attr  # clean-cycle PWAL cleanup


@cuda_required
def test_linear_nvfp4_fused_abort_then_recover_finalizes_only_new_scales():
    """Failed cycle A appends input_scale/weight_scale_2 to the NVFP4
    accumulator LISTS and stashes a per-shard block-scale dict entry; the
    fresh cycle's pre_reload purges them; cycle B's finalize then packs and
    computes scales from ONLY B. Without the purge, _finalize_nvfp4_scales
    would see A's 0.25 alongside B's 0.5 and raise its cross-shard allclose
    assert (A surfacing in the recovery cycle)."""
    lin = _make_linear_stub()
    lin.quant_method = NVFP4LinearMethod()
    lin.weights_loading_config = WeightsLoadingConfig(weight_mode=WeightMode.FUSED_GATE_UP_LINEAR)
    lin.fused_weight_shard_indices_mapping = {"gate": (0, 4), "up": (4, 4)}
    lin.scaling_vector_size = 16
    lin.use_cute_dsl_blockscaling_mm = False
    # out_features=8, in_features=32 -> per-shard ckpt weight_scale [4, 2].
    sf_rows, sf_cols = fp4_utils.pad_up(8, 128), fp4_utils.pad_up(32 // 16, 4)
    lin.weight_scale = torch.nn.Parameter(
        torch.zeros(sf_rows * sf_cols, dtype=fp4_utils.float4_sf_dtype), requires_grad=False
    )
    for name in ("input_scale", "inv_input_scale", "alpha", "weight_scale_2"):
        setattr(lin, name, torch.nn.Parameter(torch.zeros(1), requires_grad=False))

    def _shard(ws_value, input_scale, ws2):
        return {
            "weight_scale": torch.full((4, 2), ws_value).to(torch.float8_e4m3fn),
            "input_scale": torch.tensor(input_scale),
            "weight_scale_2": torch.tensor(ws2),
        }

    # Cycle A (aborts): gate-only bucket with input_scale 0.25.
    lin.load_weights([_shard(4.0, 0.25, 4.0), {}], allow_partial_loading=True)
    assert set(lin.tmp_nvfp4_weight_scales) == {"gate"}
    assert len(lin.tmp_nvfp4_input_scales_list) == 1

    lin.pre_reload_weights()
    for attr in NVFP4LinearMethod._RELOAD_TRANSIENT_ATTRS:
        assert not hasattr(lin, attr), attr

    # Cycle B: full covering resend with input_scale 0.5 (!= A's 0.25).
    gate_b, up_b = _shard(2.0, 0.5, 2.0), _shard(3.0, 0.5, 2.0)
    lin.load_weights([gate_b, up_b], allow_partial_loading=True)
    lin.process_weights_after_loading()

    assert lin.input_scale.item() == pytest.approx(2.0)  # 1 / B's 0.5
    assert lin.weight_scale_2.item() == pytest.approx(2.0)
    assert lin.alpha.item() == pytest.approx(0.5 * 2.0)
    # Bit-check the packed block scales against a B-only pack.
    expected = torch.ops.trtllm.block_scale_interleave(
        torch.cat([gate_b["weight_scale"], up_b["weight_scale"]])
        .view(fp4_utils.float4_sf_dtype)
        .cuda()
    )
    assert torch.equal(lin.weight_scale.data.cpu(), expected.cpu().view(fp4_utils.float4_sf_dtype))
    for attr in NVFP4LinearMethod._RELOAD_TRANSIENT_ATTRS:
        assert not hasattr(lin, attr), attr  # clean-cycle PWAL cleanup


class _NVFP4ScaleOnlyMethod(NVFP4FusedMoEMethod):
    """Concrete NVFP4 method for scale-accumulator tests: block-scale
    loaders are never reached (no *.weight_scale keys delivered)."""

    def setup_quant_scales(self, module):
        pass

    def load_expert_w3_w1_weight_scale_nvfp4(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("no weight_scale keys are delivered in this test")

    def load_expert_w2_weight_scale_nvfp4(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("no weight_scale keys are delivered in this test")


def test_moe_nvfp4_abort_then_recover_finalizes_only_new_scales():
    """Failed cycle A stashes expert-1 input scales in
    tmp_raw_input_scales; the fresh cycle's pre_reload purges the dict;
    cycle B (expert-0 scales only) then finalizes global input scales from
    ONLY B. Without the purge, PWAL's max over ALL dict entries would pick
    A's 8.0 (a covering resend does not overwrite A's per-expert keys
    key-for-key), yielding 1/8 instead of B's 1/2."""
    method = _NVFP4ScaleOnlyMethod.__new__(_NVFP4ScaleOnlyMethod)
    module = _make_moe_module([], expert_ids=(0, 1))
    module.num_experts = 2
    for name in ("fc31_input_scale", "fc2_input_scale"):
        setattr(module, name, torch.nn.Parameter(torch.zeros(1), requires_grad=False))
    for name in ("fc31_alpha", "fc2_alpha"):
        setattr(module, name, torch.nn.Parameter(torch.zeros(2), requires_grad=False))
    module.w3_w1_weight_scale = torch.nn.Parameter(torch.zeros(2, 4, 1), requires_grad=False)
    module.w2_weight_scale = torch.nn.Parameter(torch.zeros(2, 4, 1), requires_grad=False)

    # Cycle A (aborts): expert-1 input scales = 8.0.
    method.load_quant_scales(module, {f"1.w{r}.input_scale": torch.tensor(8.0) for r in (1, 2, 3)})
    assert module.tmp_raw_input_scales[1]["w1"].item() == 8.0

    method.pre_reload_weights(module)
    for attr in _NVFP4ScaleOnlyMethod._RELOAD_TRANSIENT_ATTRS:
        assert not hasattr(module, attr), attr

    # Cycle B: expert-0 input scales = 2.0 (expert 1 not re-delivered).
    method.load_quant_scales(module, {f"0.w{r}.input_scale": torch.tensor(2.0) for r in (1, 2, 3)})
    method.process_weights_after_loading(module)

    assert module.fc31_input_scale.item() == pytest.approx(0.5)  # 1 / B's 2.0
    assert module.fc2_input_scale.item() == pytest.approx(0.5)
    for attr in _NVFP4ScaleOnlyMethod._RELOAD_TRANSIENT_ATTRS:
        assert not hasattr(module, attr), attr  # clean-cycle PWAL cleanup
