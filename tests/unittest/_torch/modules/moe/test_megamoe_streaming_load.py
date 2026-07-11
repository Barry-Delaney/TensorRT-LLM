# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Streaming-load lifecycle tests for ``NVFP4MegaMoECuteDslMethod``.

Exercises the load/reload path of the MegaMoE-CuteDSL NVFP4 quant method in
``tensorrt_llm/_torch/modules/fused_moe/quantization.py`` ONLY:

* ``create_weights`` shrinks ``_STREAMED_SOURCE_PARAMS`` to 0-element
  placeholders,
* ``load_weights`` lazily rematerializes them (VANILLA key-suffix check) --
  on CUDA for the initial load, on CPU for partial RELOADS (so
  layer-interleaved bucket orderings keep the GPU peak ordering-independent)
  -- routes aux-only shards through ``load_quant_scales``, and finalizes
  eagerly per module once coverage AND every aux scale family complete,
* ``process_weights_after_loading`` runs the three-way triage, the
  per-component coverage guard, and the aux scale-family checks
  (all-or-nothing per family PLUS the bidirectional
  ``weight_scale_2 <-> input_scale`` dependency: the alphas derive from
  both, so a one-sided refresh is rejected),
* ``pre_reload_weights`` keeps the streamed sources as placeholders.

Everything here is pure torch tensor plumbing: the CuTe DSL kernel is never
imported or launched, so a single CUDA GPU of any architecture is enough and
no model weights are required. The quant method is driven directly against a
lightweight mock module that provides exactly the attributes the method
touches; the weights dict is handed to the quant method as a single Dict --
the same convention as ``MegaMoECuteDsl.load_weights`` (which unwraps the
caller's ``[weights_dict]`` before forwarding).
"""

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs 1 CUDA GPU (initial-load sources are rematerialized on cuda)",
)

NUM_EXPERTS = 4
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 64

_STREAMED_PARAMS = ("w3_w1_weight", "w3_w1_weight_scale", "w2_weight", "w2_weight_scale")
_MEGA_PARAMS = ("mega_fc1_weight", "mega_fc1_weight_sf", "mega_fc2_weight", "mega_fc2_weight_sf")
_SNAPSHOT_PARAMS = _MEGA_PARAMS + (
    "fc1_norm_const",
    "fc31_alpha",
    "fc2_alpha",
    "fc31_input_scale",
    "fc2_input_scale",
)

_WEIGHT_SUFFIXES = (".weight", ".weight_scale")
_AUX_SUFFIXES = (".input_scale", ".weight_scale_2")

# One bucket per module-relative tensor-name suffix, mirroring the in-tree
# RLHF update-weights flow (tests/unittest/_torch/ray_orchestrator/single_gpu/
# test_llm_update_weights.py groups IPC buckets by weight-name suffix across
# ALL layers, in list(set(...)) i.e. ARBITRARY order). This ordering is the
# adversarial one: weight_scale_2 and w2.input_scale land BEFORE the w1/w3
# input scales, so a premature eager finalize (treating "is13 not yet
# arrived" as "omitted") would consume the ws2 stash early and reject the
# trailing is13 buckets. The eager finalize may only fire on the bucket
# where a module's LAST needed family lands (here: .w3.input_scale).
_SUFFIX_ORDER = (
    ".w1.weight",
    ".w3.weight",
    ".w2.weight",
    ".w1.weight_scale",
    ".w3.weight_scale",
    ".w2.weight_scale",
    ".w1.weight_scale_2",
    ".w3.weight_scale_2",
    ".w2.weight_scale_2",
    ".w2.input_scale",
    ".w1.input_scale",
    ".w3.input_scale",
)


def _load_classes():
    """Import lazily so collection works on hosts without tensorrt_llm."""
    from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode
    from tensorrt_llm._torch.modules.fused_moe.quantization import NVFP4MegaMoECuteDslMethod

    return MoEWeightLoadingMode, NVFP4MegaMoECuteDslMethod


class _StreamingMoEModule(nn.Module):
    """Minimal stand-in for the MegaMoECuteDsl backend module.

    Provides only the attributes ``NVFP4MegaMoECuteDslMethod`` (and its
    NVFP4 / base parents) touch on the load/reload path. Single-rank world:
    tp_size == ep_size == 1, every expert is local.
    """

    def __init__(self, weight_loading_mode):
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.hidden_size = HIDDEN_SIZE
        self.intermediate_size_per_partition = INTERMEDIATE_SIZE
        self.expand_intermediate_size_per_partition = 2 * INTERMEDIATE_SIZE
        self.expert_size_per_partition = NUM_EXPERTS
        self.initial_local_expert_ids = list(range(NUM_EXPERTS))
        self.tp_size = 1
        self.tp_rank = 0
        self.ep_size = 1
        self.ep_rank = 0
        self.dtype = torch.bfloat16
        self.bias = False
        self.weight_loading_mode = weight_loading_mode
        # No EPLB in this test: need_load_shared_weights() must be False.
        self.layer_load_balancer = None

    def _add_raw_shared_weights_for_unmap(self, weight_tensors):
        # Only forwards to the dynamic load balancer in production; no-op here.
        del weight_tensors


def _w13_input_scale(expert_id: int) -> float:
    return 0.5 + 0.125 * expert_id


def _w2_input_scale(expert_id: int) -> float:
    return 0.25 + 0.0625 * expert_id


def _make_vanilla_weights(seed: int = 20260708) -> dict:
    """Synthesize a VANILLA-mode NVFP4 checkpoint dict for all experts.

    Byte layouts follow the loader contract (verified against
    ``FusedMoEMethodBase.load_expert_weights_to_dst`` and
    ``NVFP4FusedMoEMethod.load_quant_scales`` / ``load_fp4_weight_block_scales``):

    * ``{e}.w1/w3.weight``      uint8 ``(I, H // 2)``   (viewed as int64 rows)
    * ``{e}.w2.weight``         uint8 ``(H, I // 2)``
    * ``{e}.w1/w3.weight_scale``  fp8 ``(I, H // 16)``  (viewed as int32 rows)
    * ``{e}.w2.weight_scale``     fp8 ``(H, I // 16)``
    * ``{e}.*.input_scale`` / ``{e}.*.weight_scale_2``  fp32 scalars

    The bytes are random -- only the streaming lifecycle is under test, no
    kernel ever consumes them numerically.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand_u8(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=gen)

    def rand_fp8(*shape):
        # Any byte payload works (pure moves/reinterprets); stay clear of the
        # 0x7f/0xff NaN encodings for hygiene.
        raw = torch.randint(1, 120, shape, dtype=torch.uint8, device="cuda", generator=gen)
        return raw.view(torch.float8_e4m3fn)

    weights = {}
    for e in range(NUM_EXPERTS):
        weights[f"{e}.w1.weight"] = rand_u8(INTERMEDIATE_SIZE, HIDDEN_SIZE // 2)
        weights[f"{e}.w3.weight"] = rand_u8(INTERMEDIATE_SIZE, HIDDEN_SIZE // 2)
        weights[f"{e}.w2.weight"] = rand_u8(HIDDEN_SIZE, INTERMEDIATE_SIZE // 2)
        weights[f"{e}.w1.weight_scale"] = rand_fp8(INTERMEDIATE_SIZE, HIDDEN_SIZE // 16)
        weights[f"{e}.w3.weight_scale"] = rand_fp8(INTERMEDIATE_SIZE, HIDDEN_SIZE // 16)
        weights[f"{e}.w2.weight_scale"] = rand_fp8(HIDDEN_SIZE, INTERMEDIATE_SIZE // 16)
        # w1/w3 input scales must match per expert (parent PWAL asserts).
        weights[f"{e}.w1.input_scale"] = torch.tensor(_w13_input_scale(e), dtype=torch.float32)
        weights[f"{e}.w3.input_scale"] = torch.tensor(_w13_input_scale(e), dtype=torch.float32)
        weights[f"{e}.w2.input_scale"] = torch.tensor(_w2_input_scale(e), dtype=torch.float32)
        # w1/w3 weight_scale_2 must match per expert (reconcile warns/maxes).
        ws13 = torch.tensor(0.01 * (e + 1), dtype=torch.float32)
        weights[f"{e}.w1.weight_scale_2"] = ws13
        weights[f"{e}.w3.weight_scale_2"] = ws13.clone()
        weights[f"{e}.w2.weight_scale_2"] = torch.tensor(0.02 * (e + 1), dtype=torch.float32)
    return weights


def _bucket(weights: dict, expert_ids=None, suffixes=None) -> dict:
    """Select a shard bucket by expert id and/or module-relative key suffix."""
    out = {}
    for key, value in weights.items():
        if expert_ids is not None and int(key.split(".")[0]) not in expert_ids:
            continue
        if suffixes is not None and not key.endswith(suffixes):
            continue
        out[key] = value
    return out


def _fresh(seed: int = 20260708):
    """Build (method, module, weights) with weights created on cuda."""
    mode_cls, method_cls = _load_classes()
    module = _StreamingMoEModule(mode_cls.VANILLA)
    method = method_cls()
    with torch.device("cuda"):
        method.create_weights(module)
    return method, module, _make_vanilla_weights(seed)


def _load(method, module, bucket, allow_partial_loading):
    """Same calling convention as ``MegaMoECuteDsl.load_weights`` post-unwrap."""
    mode_cls, _ = _load_classes()
    method.load_weights(
        module, bucket, mode_cls.VANILLA, allow_partial_loading=allow_partial_loading
    )


def _streamed_numels(module) -> dict:
    return {name: getattr(module, name).data.numel() for name in _STREAMED_PARAMS}


def _assert_sources_freed(module, context=""):
    numels = _streamed_numels(module)
    assert all(n == 0 for n in numels.values()), (
        f"streamed source params should be 0-element placeholders {context}: {numels}"
    )


def _assert_sources_materialized(module, context="", device_type=None):
    numels = _streamed_numels(module)
    assert all(n > 0 for n in numels.values()), (
        f"streamed source params should be materialized {context}: {numels}"
    )
    if device_type is not None:
        devices = {name: getattr(module, name).data.device.type for name in _STREAMED_PARAMS}
        assert all(d == device_type for d in devices.values()), (
            f"streamed source params should stage on {device_type} {context}: {devices}"
        )


def _snapshot(module) -> dict:
    torch.cuda.synchronize()
    return {name: getattr(module, name).data.clone() for name in _SNAPSHOT_PARAMS}


def _assert_matches_snapshot(module, snapshot):
    torch.cuda.synchronize()
    for name, expected in snapshot.items():
        actual = getattr(module, name).data
        assert torch.equal(actual, expected), f"{name} bytes changed across reload"


def _assert_finalized(module, context=""):
    """The full PWAL tail ran for this module: sources freed, stashes gone."""
    _assert_sources_freed(module, context)
    for stash in (
        "tmp_cutlass_w3_w1_weights",
        "tmp_cutlass_w3_w1_weight_scales",
        "tmp_weight_scale_2",
        "tmp_raw_input_scales",
    ):
        assert not hasattr(module, stash), (
            f"{stash} still present {context} -- process_weights_after_loading did not run "
            "(a weight-carrying shard may have been misrouted to the aux-only path)"
        )


def _expected_mega_fc2(weights) -> torch.Tensor:
    return torch.stack([weights[f"{e}.w2.weight"] for e in range(NUM_EXPERTS)])


def _expected_mega_fc1(weights) -> torch.Tensor:
    """16-atom gate/up interleave along M: [w1[0:16], w3[0:16], w1[16:32], ...]."""
    per_slot = []
    for e in range(NUM_EXPERTS):
        gate = weights[f"{e}.w1.weight"].view(INTERMEDIATE_SIZE // 16, 16, HIDDEN_SIZE // 2)
        up = weights[f"{e}.w3.weight"].view(INTERMEDIATE_SIZE // 16, 16, HIDDEN_SIZE // 2)
        per_slot.append(
            torch.stack([gate, up], dim=1).reshape(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 2)
        )
    return torch.stack(per_slot)


def _expected_fc1_norm_const() -> torch.Tensor:
    return torch.tensor(
        [1.0 / _w2_input_scale(e) for e in range(NUM_EXPERTS)],
        dtype=torch.float32,
        device="cuda",
    )


def _full_load_and_reset_for_reload(method, module, weights):
    """Initial full (eager) load, snapshot, then pre_reload_weights."""
    _load(method, module, weights, allow_partial_loading=False)
    _assert_finalized(module, "after the initial eager load")
    snapshot = _snapshot(module)
    method.pre_reload_weights(module)
    _assert_sources_freed(module, "after pre_reload_weights (placeholders must be kept)")
    assert module._streamed_finalized_this_cycle is False
    # The streamed sources keep their OWN per-expert-row coverage; the
    # generic per-param debt must stay empty for them (no double
    # accounting; the defensive init covers only non-streamed metadata
    # entries, an empty set today).
    assert not getattr(module, "_reload_outstanding", None)
    return snapshot


def test_initial_load_then_reload_layer_atomic():
    method, module, weights = _fresh()

    # create_weights shrinks all 4 streamed sources to 0-element placeholders
    # and records their full shapes for rematerialization.
    _assert_sources_freed(module, "right after create_weights")
    assert set(module.rebuild_tensor_metadata) == set(_STREAMED_PARAMS)

    # Full eager load: base load_weights runs PWAL inline.
    _load(method, module, weights, allow_partial_loading=False)
    _assert_finalized(module, "after the initial eager load")

    # Derived buffers hold the packed bytes.
    assert torch.equal(module.mega_fc2_weight.data, _expected_mega_fc2(weights))
    assert torch.equal(module.mega_fc1_weight.data, _expected_mega_fc1(weights))
    assert torch.allclose(module.fc1_norm_const.data, _expected_fc1_norm_const())

    snapshot = _snapshot(module)

    # Hot-reload prologue keeps the streamed sources as placeholders.
    method.pre_reload_weights(module)
    _assert_sources_freed(module, "after pre_reload_weights (placeholders must be kept)")
    assert module._streamed_finalized_this_cycle is False
    # Streamed params never enter the generic per-param coverage debt
    # (their own row coverage is the guard).
    assert not getattr(module, "_reload_outstanding", None)

    # One partial call carrying the whole module: the EAGER per-module
    # finalize must pack + free without any manual PWAL sweep call.
    _load(method, module, weights, allow_partial_loading=True)
    assert module._streamed_finalized_this_cycle is True
    _assert_finalized(module, "after the single-bucket partial reload (eager finalize)")
    _assert_matches_snapshot(module, snapshot)


def test_reload_split_buckets():
    method, module, weights = _fresh()
    snapshot = _full_load_and_reset_for_reload(method, module, weights)

    # Bucket 1: experts 0-1 (weights + block scales + aux scales). Partial
    # reload sources stage on CPU (ordering-independent GPU peak).
    _load(method, module, _bucket(weights, expert_ids={0, 1}), allow_partial_loading=True)
    _assert_sources_materialized(module, "after bucket 1 (coverage incomplete)", device_type="cpu")
    assert module._streamed_finalized_this_cycle is False

    # Bucket 2: experts 2-3 completes the module -> eager finalize fires.
    _load(method, module, _bucket(weights, expert_ids={2, 3}), allow_partial_loading=True)
    assert module._streamed_finalized_this_cycle is True
    _assert_finalized(module, "after bucket 2 completed coverage")
    _assert_matches_snapshot(module, snapshot)


def test_reload_weights_then_aux_tail():
    method, module, weights = _fresh()
    snapshot = _full_load_and_reset_for_reload(method, module, weights)

    # Bucket 1: all expert weights + block scales, NO input_scale/weight_scale_2.
    # _streamed_load_complete requires the aux families too, so no finalize yet.
    _load(method, module, _bucket(weights, suffixes=_WEIGHT_SUFFIXES), allow_partial_loading=True)
    _assert_sources_materialized(module, "after the weights-only bucket", device_type="cpu")
    assert module._streamed_finalized_this_cycle is False

    # Bucket 2: all input_scale + weight_scale_2 -> coverage complete -> eager
    # finalize fires on this aux bucket.
    _load(method, module, _bucket(weights, suffixes=_AUX_SUFFIXES), allow_partial_loading=True)
    assert module._streamed_finalized_this_cycle is True
    _assert_finalized(module, "after the aux tail completed the module")
    _assert_matches_snapshot(module, snapshot)

    # Bucket 3: an aux-only refresh AFTER the eager finalize goes through the
    # load_quant_scales-only quick path (sources stay freed). It must ship
    # BOTH aux families together: the alphas derive from input_scale AND
    # weight_scale_2, and the sweep rejects a one-sided refresh (see
    # test_input_scale_only_refresh_raises).
    _load(method, module, _bucket(weights, suffixes=_AUX_SUFFIXES), allow_partial_loading=True)
    _assert_sources_freed(module, "aux-only refresh must not rematerialize sources")
    assert hasattr(module, "tmp_raw_input_scales")

    # ... and the model-wide sweep PWAL lands in the aux-only-refresh triage
    # branch without raising, leaving the packed bytes untouched.
    method.process_weights_after_loading(module)
    _assert_finalized(module, "after the sweep consumed the aux-only refresh")
    _assert_matches_snapshot(module, snapshot)


def test_reload_suffix_ordered_multi_module():
    """Suffix-grouped reload across TWO modules (the RLHF bucket ordering).

    Each bucket carries ONE tensor-name suffix for ALL experts of BOTH
    modules, so every module's sources stay materialized across almost the
    whole cycle. Contract under test: partial-reload sources stage on CPU
    (the GPU peak stays ordering-independent), no module eager-finalizes
    before its LAST needed suffix (.w3.input_scale) lands, and the final
    bucket packs + frees both modules WITHOUT a manual PWAL sweep, with
    bytes bit-exact vs each module's initial load.
    """
    states = [_fresh(seed=20260708 + i) for i in range(2)]
    snapshots = [
        _full_load_and_reset_for_reload(method, module, weights)
        for method, module, weights in states
    ]

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    last_idx = len(_SUFFIX_ORDER) - 1
    for bucket_idx, suffix in enumerate(_SUFFIX_ORDER):
        for method, module, weights in states:
            _load(method, module, _bucket(weights, suffixes=(suffix,)), allow_partial_loading=True)
        if bucket_idx < last_idx:
            for _, module, _ in states:
                # (i) mid-cycle: BOTH modules' sources are CPU-staged.
                _assert_sources_materialized(
                    module, f"mid-cycle after the '{suffix}' bucket", device_type="cpu"
                )
                # (ii) no eager finalize before the last needed suffix.
                assert module._streamed_finalized_this_cycle is False, (
                    f"module eager-finalized early (after the '{suffix}' bucket)"
                )

    # The final bucket (.w3.input_scale) completes every family: both
    # modules eager-finalize WITHOUT any manual PWAL sweep call.
    for (_, module, _), snapshot in zip(states, snapshots):
        assert module._streamed_finalized_this_cycle is True
        _assert_finalized(module, "after the last suffix bucket")
        # (iii) packed bytes bit-exact vs the initial load, per module.
        _assert_matches_snapshot(module, snapshot)

    # Smoke bound on the reload GPU peak: CPU staging keeps the delta at
    # ~one layer of transient pack sources + temps (well under a few MB at
    # these unit dims). Guards against a regression that re-accumulates
    # CUDA source sets across the interleaved cycle.
    peak_delta = torch.cuda.max_memory_allocated() - baseline
    assert peak_delta < 32 * 1024 * 1024, (
        f"reload GPU peak delta {peak_delta} bytes -- expected ~one layer of "
        "transient pack sources with CPU staging"
    )


def test_partial_coverage_raises():
    method, module, weights = _fresh()
    _full_load_and_reset_for_reload(method, module, weights)

    # Only expert 0 arrives; no eager finalize (coverage incomplete).
    _load(method, module, _bucket(weights, expert_ids={0}), allow_partial_loading=True)
    assert module._streamed_finalized_this_cycle is False

    # The model-wide sweep must refuse to pack uninitialized rows.
    with pytest.raises(RuntimeError, match="partially covered"):
        method.process_weights_after_loading(module)


def test_partial_scale_family_raises():
    method, module, weights = _fresh()
    _full_load_and_reset_for_reload(method, module, weights)

    # All weights + block scales + weight_scale_2, but input_scale for only
    # half the experts: the w1/w3/w2 input_scale families are PARTIAL.
    bucket = _bucket(weights, suffixes=_WEIGHT_SUFFIXES + (".weight_scale_2",))
    bucket.update(_bucket(weights, expert_ids={0, 1}, suffixes=(".input_scale",)))
    _load(method, module, bucket, allow_partial_loading=True)
    assert module._streamed_finalized_this_cycle is False

    with pytest.raises(RuntimeError, match="PARTIAL aux scale families"):
        method.process_weights_after_loading(module)


def test_input_scale_only_refresh_raises():
    method, module, weights = _fresh()
    _full_load_and_reset_for_reload(method, module, weights)

    # Complete single-bucket partial reload -> eager finalize.
    _load(method, module, weights, allow_partial_loading=True)
    assert module._streamed_finalized_this_cycle is True

    # Aux tail carrying ONLY input_scale (no weight_scale_2): the alphas
    # derive from BOTH families, so the sweep must reject the one-sided
    # refresh instead of silently mixing fresh input scales with stale
    # alphas. The lazy path still stashes without rematerializing.
    _load(method, module, _bucket(weights, suffixes=(".input_scale",)), allow_partial_loading=True)
    _assert_sources_freed(module, "input_scale-only tail must not rematerialize sources")
    assert hasattr(module, "tmp_raw_input_scales")

    with pytest.raises(RuntimeError, match="weight_scale_2"):
        method.process_weights_after_loading(module)


def test_reload_input_scale_without_ws2_raises():
    method, module, weights = _fresh()
    _full_load_and_reset_for_reload(method, module, weights)

    # Rematerialized path: full weights + block scales + full input_scale but
    # NO weight_scale_2. Coverage completes, yet the eager gate stays closed
    # (every aux family must be complete), and the sweep enforces the
    # bidirectional ws2 <-> input_scale dependency.
    bucket = _bucket(weights, suffixes=_WEIGHT_SUFFIXES + (".input_scale",))
    _load(method, module, bucket, allow_partial_loading=True)
    assert module._streamed_finalized_this_cycle is False
    _assert_sources_materialized(module, "after the ws2-less bucket", device_type="cpu")

    with pytest.raises(RuntimeError, match="weight_scale_2"):
        method.process_weights_after_loading(module)


def test_aux_only_without_weights_raises():
    method, module, weights = _fresh()
    _full_load_and_reset_for_reload(method, module, weights)

    # The reload cycle's ONLY bucket is aux scales: the lazy path stashes them
    # without rematerializing (sources numel == 0, finalize flag False) ...
    _load(method, module, _bucket(weights, suffixes=_AUX_SUFFIXES), allow_partial_loading=True)
    _assert_sources_freed(module, "aux-only bucket must not rematerialize sources")
    assert hasattr(module, "tmp_raw_input_scales")
    assert module._streamed_finalized_this_cycle is False

    # ... so the sweep PWAL hits triage case (c): scales without weights.
    with pytest.raises(RuntimeError, match="no expert weights"):
        method.process_weights_after_loading(module)


def test_pre_reload_purges_aborted_cycle_state():
    """Cycle start is authoritative: pre_reload_weights after a cycle that
    died mid-sweep (transients + CPU-materialized sources left behind) must
    purge the transients and re-shrink the sources, so the next cycle equals
    a from-scratch reload and the coverage guard is re-armed."""
    method, module, weights = _fresh()
    snapshot = _full_load_and_reset_for_reload(method, module, weights)

    # Aborted cycle: one partial bucket lands (stashes + covered-sets +
    # aux stashes exist, sources CPU-staged), then the sweep never runs.
    _load(method, module, _bucket(weights, expert_ids={0, 1}), allow_partial_loading=True)
    _assert_sources_materialized(module, "after the aborted cycle's bucket", device_type="cpu")
    assert hasattr(module, "tmp_cutlass_w3_w1_weights")
    assert hasattr(module, "tmp_raw_input_scales")

    # Idempotent under the double reach (module walk + backend forward).
    method.pre_reload_weights(module)
    method.pre_reload_weights(module)
    # The purge list is the implementation's class constant so this test
    # cannot drift from what pre_reload_weights actually deletes.
    for attr in type(method)._RELOAD_TRANSIENT_ATTRS:
        assert not hasattr(module, attr), f"{attr} survived pre_reload_weights"
    _assert_sources_freed(module, "after pre_reload_weights purged the aborted cycle")
    assert module._streamed_sources_rematerialized is False
    assert module._streamed_finalized_this_cycle is False

    # A clean full reload after the purge equals a from-scratch load.
    _load(method, module, weights, allow_partial_loading=True)
    assert module._streamed_finalized_this_cycle is True
    _assert_finalized(module, "after the clean reload following the purge")
    _assert_matches_snapshot(module, snapshot)


def test_pre_reload_drops_stale_checkpoint_stash():
    """Stale-poisoning regression: leftovers from checkpoint A's aborted
    cycle must not count toward checkpoint B's coverage (an eager finalize
    would bake mixed A/B expert rows), and a full B resend must produce a
    bit-exact pure-B module."""
    # Pure-B reference bytes from an independent from-scratch module.
    method_ref, module_ref, _ = _fresh()
    weights_b = _make_vanilla_weights(seed=31337)
    _load(method_ref, module_ref, weights_b, allow_partial_loading=False)
    snapshot_b = _snapshot(module_ref)

    method, module, weights_a = _fresh()
    _full_load_and_reset_for_reload(method, module, weights_a)

    # Aborted cycle: checkpoint A delivers expert 0 (weights + scales), then
    # the cycle dies before the sweep.
    _load(method, module, _bucket(weights_a, expert_ids={0}), allow_partial_loading=True)
    assert hasattr(module, "tmp_cutlass_w3_w1_weights")

    # Fresh cycle delivering only experts 1-3 of checkpoint B: without the
    # cycle-start purge the stale A expert-0 stash entries complete coverage
    # and the eager finalize bakes a mixed A/B module; with it the cycle is
    # incomplete and the sweep refuses to pack.
    method.pre_reload_weights(module)
    _load(method, module, _bucket(weights_b, expert_ids={1, 2, 3}), allow_partial_loading=True)
    assert module._streamed_finalized_this_cycle is False
    with pytest.raises(RuntimeError, match="partially covered"):
        method.process_weights_after_loading(module)

    # Recovery: a full-B resend in a fresh cycle is bit-exact pure-B.
    method.pre_reload_weights(module)
    _load(method, module, weights_b, allow_partial_loading=True)
    assert module._streamed_finalized_this_cycle is True
    _assert_finalized(module, "after streaming checkpoint B in full")
    _assert_matches_snapshot(module, snapshot_b)
