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
"""CPU-only unit tests for the WorkerExtension.update_weights exception boundary.

No ray, no GPU: a failed update_weights must abort the in-progress
weight-reload cycle — clear the once-per-cycle pre-reload latch and invoke
the model's duck-typed ``abort_reload_cycle`` hook — so the next call
re-runs every module's pre_reload_weights instead of merging the failed
cycle's leftovers into the new cycle.
"""

import base64
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm.llmapi.rlhf_utils as rlhf_utils
from tensorrt_llm.llmapi.rlhf_utils import _PRE_RELOAD_LATCH_ATTR, WorkerExtension

_UUID = "uuid-0"


class _Child(torch.nn.Module):
    """Module with reload-lifecycle hooks, call recorders, and fault taps."""

    def __init__(self):
        super().__init__()
        self.pre_reload_calls = 0
        self.pwal_calls = 0
        self.post_load_calls = 0
        self.pwal_error = None
        self.post_load_error = None
        self.capability_calls = 0
        self.capability_error = None
        self.lifecycle = []  # ordered record: "capability" / "pre_reload"

    def check_reload_capability(self):
        self.capability_calls += 1
        self.lifecycle.append("capability")
        if self.capability_error is not None:
            raise self.capability_error

    def pre_reload_weights(self):
        self.pre_reload_calls += 1
        self.lifecycle.append("pre_reload")

    def process_weights_after_loading(self):
        self.pwal_calls += 1
        if self.pwal_error is not None:
            raise self.pwal_error

    def post_load_weights(self):
        self.post_load_calls += 1
        if self.post_load_error is not None:
            raise self.post_load_error


class _Model(torch.nn.Module):
    """Root model exposing the duck-typed boundary abort hook."""

    def __init__(self):
        super().__init__()
        self.child = _Child()
        self.abort_calls = []
        self.abort_error = None

    def abort_reload_cycle(self, reason):
        self.abort_calls.append(reason)
        if self.abort_error is not None:
            raise self.abort_error


def _make_ext(model, monkeypatch, reload_error=None, reset_error=None, sync_error=None):
    """Real WorkerExtension over a stub engine; returns (ext, events).

    ``events`` records the finalize tail ordering: "reload" for data calls,
    "sync" for torch.cuda.synchronize, and ("reset", <latch still set?>)
    for reset_prefix_cache — pinning that the device sync runs BEFORE the
    prefix-cache reset and the latch delete.
    """
    events = []

    @contextmanager
    def control_action(drain=True):
        yield

    def reload(mdl, weights, allow_partial_loading=False):
        events.append("reload")
        if reload_error is not None:
            raise reload_error

    def reset_prefix_cache():
        events.append(("reset", hasattr(model, _PRE_RELOAD_LATCH_ATTR)))
        if reset_error is not None:
            raise reset_error

    def synchronize():
        events.append("sync")
        if sync_error is not None:
            raise sync_error

    ext = object.__new__(WorkerExtension)
    ext.device_id = 0
    ext.engine = SimpleNamespace(
        control_action=control_action,
        model_engine=SimpleNamespace(model=model, model_loader=SimpleNamespace(reload=reload)),
        reset_prefix_cache=reset_prefix_cache,
    )
    monkeypatch.setattr(rlhf_utils, "get_device_uuid", lambda device_id: _UUID)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return ext, events


def _bucket(**weights):
    handles = []
    for name, tensor in weights.items():

        def _rebuild(*args, _t=tensor):
            # Pins that the device-id rewrite (list_args[6] = device_id, 0
            # here) reached the rebuild func, replacing the placeholder.
            assert args[6] == 0, "device-id rewrite did not reach the rebuild func"
            return _t

        handles.append((name, (_rebuild, (0, 0, 0, 0, 0, 0, "device-placeholder"))))
    return {_UUID: handles}


def test_update_weights_success_path_clears_latch_without_abort(monkeypatch):
    model = _Model()
    ext, events = _make_ext(model, monkeypatch)

    ext.update_weights(_bucket(w=torch.zeros(2)))
    assert model.child.pre_reload_calls == 1
    assert hasattr(model, _PRE_RELOAD_LATCH_ATTR)  # cycle stays open across buckets

    ext.update_weights(None)
    assert model.child.pwal_calls == 1
    assert model.child.post_load_calls == 1
    assert not hasattr(model, _PRE_RELOAD_LATCH_ATTR)
    assert model.abort_calls == []
    # Truthful cycle-complete marker: async finalize errors surface at the
    # sync BEFORE reset_prefix_cache and the latch delete (latch still set
    # when reset runs).
    assert events == ["reload", "sync", ("reset", True)]

    # A new cycle re-runs the pre-hooks (latch was consumed by finalize).
    ext.update_weights(_bucket(w=torch.zeros(2)))
    assert model.child.pre_reload_calls == 2


@pytest.mark.parametrize("failure", ["pwal", "post_load", "reset", "sync", "reload"])
def test_update_weights_failure_aborts_cycle_and_reruns_pre_hooks(monkeypatch, failure):
    model = _Model()
    boom = RuntimeError(f"{failure} boom")
    kwargs = {}
    if failure == "reload":
        kwargs["reload_error"] = boom
    if failure == "reset":
        kwargs["reset_error"] = boom
    if failure == "sync":
        kwargs["sync_error"] = boom
    ext, events = _make_ext(model, monkeypatch, **kwargs)
    if failure == "pwal":
        model.child.pwal_error = boom
    if failure == "post_load":
        model.child.post_load_error = boom

    if failure == "reload":
        with pytest.raises(RuntimeError, match="reload boom"):
            ext.update_weights(_bucket(w=torch.zeros(2)))
    else:
        ext.update_weights(_bucket(w=torch.zeros(2)))
        with pytest.raises(RuntimeError, match=f"{failure} boom"):
            ext.update_weights(None)

    # Boundary abort: latch cleared + duck-typed model hook invoked once
    # with the original cause; the original exception type propagated.
    assert not hasattr(model, _PRE_RELOAD_LATCH_ATTR)
    assert len(model.abort_calls) == 1
    assert f"{failure} boom" in model.abort_calls[0]
    if failure == "sync":
        # The async-error sync precedes reset_prefix_cache and the latch
        # delete, so a sync failure aborts while nothing was committed.
        assert not any(isinstance(ev, tuple) for ev in events)

    # The next call must re-run pre_reload_weights on every module (the
    # stale latch no longer suppresses the walk).
    before = model.child.pre_reload_calls
    ext_ok, _ = _make_ext(model, monkeypatch)
    ext_ok.update_weights(_bucket(w=torch.zeros(2)))
    assert model.child.pre_reload_calls == before + 1


@pytest.mark.parametrize("failure", ["uuid", "deser"])
def test_update_weights_precondition_failure_destroys_and_aborts_nothing(monkeypatch, failure):
    """Precondition failures must destroy nothing and abort nothing.

    UUID lookup and payload deserialization run BEFORE the pre-hook walk
    AND before the try/abort boundary: a failure there must destroy nothing
    (the walk re-registers transformed params as uninitialized storage), set
    no latch, and fire no abort -- the cycle stays coherent and the bucket
    is simply retryable.
    """
    model = _Model()
    ext, events = _make_ext(model, monkeypatch)

    if failure == "uuid":
        with pytest.raises(ValueError, match="not found in ipc_handles"):
            ext.update_weights({"other-uuid": []})
    else:

        def _boom(*args, **kwargs):
            raise RuntimeError("deser boom")

        monkeypatch.setattr(rlhf_utils.serialization, "loads", _boom)
        payload = base64.b64encode(b"payload").decode()
        with pytest.raises(RuntimeError, match="deser boom"):
            ext.update_weights({_UUID: payload})

    assert model.child.pre_reload_calls == 0  # nothing destroyed
    assert not hasattr(model, _PRE_RELOAD_LATCH_ATTR)
    assert model.abort_calls == []  # nothing aborted
    assert events == []  # reload never reached

    # Bucket-level retry is legal: the next good bucket opens the cycle
    # normally and a failing bucket MID-cycle leaves the open cycle intact.
    if failure == "deser":
        monkeypatch.undo()
        ext, events = _make_ext(model, monkeypatch)
    ext.update_weights(_bucket(w=torch.zeros(2)))
    assert model.child.pre_reload_calls == 1
    assert hasattr(model, _PRE_RELOAD_LATCH_ATTR)
    with pytest.raises(ValueError, match="not found in ipc_handles"):
        ext.update_weights({"other-uuid": []})
    assert model.child.pre_reload_calls == 1  # walk not re-run
    assert hasattr(model, _PRE_RELOAD_LATCH_ATTR)  # cycle still open
    assert model.abort_calls == []
    ext.update_weights(None)  # finalize commits the intact cycle
    assert not hasattr(model, _PRE_RELOAD_LATCH_ATTR)
    assert model.abort_calls == []


def test_abort_update_weights_broadcast_converges_cycle_generation(monkeypatch):
    """Broadcast abort converges divergent per-rank cycle generations.

    After a single-rank update_weights failure the ranks diverge (the
    failing rank aborted locally, the healthy rank keeps its latch). The
    trainer-broadcast abort_update_weights must converge every rank back to
    the same fresh cycle generation via the same abort choke point.
    """
    rank0, rank1 = _Model(), _Model()
    ext0, _ = _make_ext(rank0, monkeypatch)
    ext1, _ = _make_ext(rank1, monkeypatch, reload_error=RuntimeError("rank1 boom"))

    ext0.update_weights(_bucket(w=torch.zeros(2)))
    with pytest.raises(RuntimeError, match="rank1 boom"):
        ext1.update_weights(_bucket(w=torch.zeros(2)))

    # Divergence: healthy rank mid-cycle, failed rank aborted.
    assert hasattr(rank0, _PRE_RELOAD_LATCH_ATTR)
    assert not hasattr(rank1, _PRE_RELOAD_LATCH_ATTR)
    assert rank0.abort_calls == []
    assert len(rank1.abort_calls) == 1

    # Trainer broadcast to ALL ranks (the coordinator cannot know which
    # ranks succeeded): routes through the duck-typed model hook with the
    # given reason and clears every latch.
    for ext in (ext0, ext1):
        ext.abort_update_weights("peer failed")
    assert not hasattr(rank0, _PRE_RELOAD_LATCH_ATTR)
    assert not hasattr(rank1, _PRE_RELOAD_LATCH_ATTR)
    assert any("peer failed" in c for c in rank0.abort_calls)
    assert any("peer failed" in c for c in rank1.abort_calls)

    # Converged generation: the next sweep re-runs pre-hooks on the
    # previously-healthy rank too (its stale mid-cycle latch is gone).
    before = rank0.child.pre_reload_calls
    ext0.update_weights(_bucket(w=torch.zeros(2)))
    assert rank0.child.pre_reload_calls == before + 1


def test_abort_update_weights_idempotent_without_cycle():
    """Broadcast abort is safe without a cycle, a hook, or a first call.

    It must be callable on ranks with no cycle in progress and on models
    without the duck-typed hook, and must be repeatable.
    """
    model = torch.nn.Module()  # no abort_reload_cycle hook, no latch
    ext = object.__new__(WorkerExtension)
    ext.engine = SimpleNamespace(model_engine=SimpleNamespace(model=model))

    ext.abort_update_weights()  # default reason; must not raise
    ext.abort_update_weights("again")  # idempotent
    assert not hasattr(model, _PRE_RELOAD_LATCH_ATTR)

    hooked = _Model()
    setattr(hooked, _PRE_RELOAD_LATCH_ATTR, True)
    ext_hooked = object.__new__(WorkerExtension)
    ext_hooked.engine = SimpleNamespace(model_engine=SimpleNamespace(model=hooked))
    ext_hooked.abort_update_weights("cleanup")
    assert not hasattr(hooked, _PRE_RELOAD_LATCH_ATTR)
    assert hooked.abort_calls and "cleanup" in hooked.abort_calls[0]


def test_update_weights_abort_hook_failure_does_not_mask(monkeypatch):
    model = _Model()
    model.abort_error = RuntimeError("hook boom")
    model.child.pwal_error = ValueError("orig boom")
    ext, _ = _make_ext(model, monkeypatch)

    ext.update_weights(_bucket(w=torch.zeros(2)))
    with pytest.raises(ValueError, match="orig boom"):
        ext.update_weights(None)

    # The latch is cleared UNCONDITIONALLY even when the hook raised.
    assert not hasattr(model, _PRE_RELOAD_LATCH_ATTR)
    assert len(model.abort_calls) == 1


def test_update_weights_abort_without_hook_still_clears_latch(monkeypatch):
    model = torch.nn.Module()  # no abort_reload_cycle hook at all
    child = _Child()
    model.child = child
    child.pwal_error = RuntimeError("boom")
    ext, _ = _make_ext(model, monkeypatch)

    ext.update_weights(_bucket(w=torch.zeros(2)))
    with pytest.raises(RuntimeError, match="boom"):
        ext.update_weights(None)
    assert not hasattr(model, _PRE_RELOAD_LATCH_ATTR)

    child.pwal_error = None
    before = child.pre_reload_calls
    ext.update_weights(_bucket(w=torch.zeros(2)))
    assert child.pre_reload_calls == before + 1


def test_abort_weight_reload_cycle_helper_contract(monkeypatch):
    """The abort helper must never raise.

    Non-callable hooks are ignored, the latch delete is idempotent, and an
    unreachable model only logs.
    """
    errors = []
    monkeypatch.setattr(
        rlhf_utils.logger, "error", lambda *msg: errors.append(" ".join(str(m) for m in msg))
    )
    model = torch.nn.Module()
    model.abort_reload_cycle = "not-a-hook"  # non-callable: skipped
    setattr(model, _PRE_RELOAD_LATCH_ATTR, True)
    ext = object.__new__(WorkerExtension)
    ext.engine = SimpleNamespace(model_engine=SimpleNamespace(model=model))

    ext._abort_weight_reload_cycle(RuntimeError("cause"))
    assert not hasattr(model, _PRE_RELOAD_LATCH_ATTR)
    ext._abort_weight_reload_cycle(RuntimeError("cause2"))  # double abort
    # The callable() gate SKIPS a non-callable hook attr; it must not be
    # attempted-and-logged as a hook failure (nor logged as anything else).
    assert errors == []

    class _BrokenEngine:
        @property
        def model_engine(self):
            raise RuntimeError("engine gone")

    ext_broken = object.__new__(WorkerExtension)
    ext_broken.engine = _BrokenEngine()
    ext_broken._abort_weight_reload_cycle(RuntimeError("cause"))  # must not raise
    assert any("model unavailable" in e for e in errors)


# ---------------------------------------------------------------------------
# Tree-wide capability preflight (check_reload_capability)
# ---------------------------------------------------------------------------


def test_update_weights_preflight_refusal_destroys_nothing(monkeypatch):
    """A submodule capability refusal is a PRECONDITION failure.

    Raised before the destructive walk, so nothing is destroyed, no latch
    is set, no abort fires, no marker is opened -- and the same worker
    retries successfully once the incompatibility is removed.
    """
    model = _Model()
    model.child.capability_error = NotImplementedError("EPLB")
    ext, events = _make_ext(model, monkeypatch)

    with pytest.raises(NotImplementedError, match="EPLB"):
        ext.update_weights(_bucket(w=torch.zeros(2)))
    assert model.child.capability_calls == 1
    assert model.child.pre_reload_calls == 0  # nothing destroyed
    assert not hasattr(model, _PRE_RELOAD_LATCH_ATTR)
    assert model.abort_calls == []  # precondition: no abort
    assert events == []  # reload never reached
    assert getattr(model.child, "_reload_outstanding", None) is None

    model.child.capability_error = None
    ext.update_weights(_bucket(w=torch.zeros(2)))  # retryable
    assert model.child.pre_reload_calls == 1
    assert hasattr(model, _PRE_RELOAD_LATCH_ATTR)


def test_update_weights_preflight_runs_once_per_cycle_before_walk(monkeypatch):
    """Preflight shares the once-per-cycle latch gate and precedes the walk.

    Data calls only; a second bucket in the same cycle re-runs neither.
    """
    model = _Model()
    ext, _ = _make_ext(model, monkeypatch)

    ext.update_weights(_bucket(w=torch.zeros(2)))
    assert model.child.lifecycle == ["capability", "pre_reload"]
    ext.update_weights(_bucket(w=torch.zeros(2)))  # same cycle: no re-run
    assert model.child.capability_calls == 1
    ext.update_weights(None)  # finalize closes the cycle
    ext.update_weights(_bucket(w=torch.zeros(2)))  # new cycle re-runs both
    assert model.child.lifecycle == [
        "capability",
        "pre_reload",
        "capability",
        "pre_reload",
    ]


def test_update_weights_preflight_vacuous_without_hook(monkeypatch):
    """Modules without check_reload_capability pass vacuously.

    Duck-typed opt-in: a data call on a hook-less tree behaves exactly as
    before.
    """
    model = torch.nn.Module()
    model.plain_child = torch.nn.Module()  # no capability hook
    ext, _ = _make_ext(model, monkeypatch)

    ext.update_weights(_bucket(w=torch.zeros(2)))
    assert hasattr(model, _PRE_RELOAD_LATCH_ATTR)


def test_update_weights_preflight_skips_weights_removed(monkeypatch):
    """_weights_removed modules are skipped by the preflight.

    Exactly like the destructive walk skips them.
    """
    model = _Model()
    model.child.capability_error = NotImplementedError("EPLB")
    model.child._weights_removed = True
    ext, _ = _make_ext(model, monkeypatch)

    ext.update_weights(_bucket(w=torch.zeros(2)))  # no raise
    assert model.child.capability_calls == 0
    assert model.child.pre_reload_calls == 0  # walk skips it too
    assert hasattr(model, _PRE_RELOAD_LATCH_ATTR)


def test_update_weights_bare_finalize_skips_preflight(monkeypatch):
    """The preflight gates DATA calls only.

    A bare finalize never runs the destructive walk, so it must stay legal
    on a refusing tree.
    """
    model = _Model()
    model.child.capability_error = NotImplementedError("cannot reload")
    ext, _ = _make_ext(model, monkeypatch)

    ext.update_weights(None)  # no raise
    assert model.child.capability_calls == 0


def test_abort_update_weights_keeps_reload_coverage_outstanding(monkeypatch):
    """Sticky veto: the abort must NOT clear per-module coverage debt.

    The abort converges cycle state (latch, stash, taint), but an abort
    followed by a bare finalize would otherwise bless empty_like garbage
    on generic models. The next cycle's pre_reload re-inits the debt.
    """
    model = _Model()
    ext, _ = _make_ext(model, monkeypatch)
    model.child._reload_outstanding = {"weight": {"*"}}
    setattr(model, _PRE_RELOAD_LATCH_ATTR, True)

    ext.abort_update_weights("peer failed")
    assert not hasattr(model, _PRE_RELOAD_LATCH_ATTR)
    assert model.child._reload_outstanding == {"weight": {"*"}}  # untouched
