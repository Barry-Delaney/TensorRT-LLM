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
import base64
import gc
from typing import Optional

import torch

from tensorrt_llm import serialization
from tensorrt_llm._ray_utils import control_action_decorator
from tensorrt_llm._torch.modules.fused_moe.moe_load_balancer import MoeLoadBalancer
from tensorrt_llm._torch.utils import get_device_uuid
from tensorrt_llm.logger import logger

# Once-per-cycle latch: set at the first ``update_weights`` of a reload cycle,
# deleted on finalize or abort. Keep in sync with the copy in
# modeling_deepseekv4.py (llmapi must not import _torch model files).
_PRE_RELOAD_LATCH_ATTR = "first_pre_reload_weights"


class WorkerExtension:
    """Worker extension class for extending TensorRT-LLM Ray workers with custom functionality.

    This class can be injected into tensorrt_llm.LLM() by specifying it via the
    ray_worker_extension_cls parameter in LLMArgs when using orchestrator_type='ray'.
    The extension methods will be available on each Ray worker and can be called via
    the LLM's collective RPC mechanism.

    Examples:
        Creating an LLM with worker extension:

        >>> llm = LLM(
        ...     model=model_dir,
        ...     orchestrator_type="ray",
        ...     ray_worker_extension_cls="rlhf_utils.WorkerExtension",
        ... )

        Calling extension methods via collective RPC:

        >>> llm._collective_rpc("update_weights", args=(ipc_handles,))
    """

    @control_action_decorator
    def update_weights(self, ipc_handles: Optional[dict] = None):
        """Update model weights from IPC (Inter-Process Communication) handles.

        This method receives shared memory handles from another process (typically FSDP training),
        reconstructs tensors from these handles, and loads them into the TensorRT-LLM model.
        Uses the control_action_decorator to ensure all active requests are finished before
        updating weights.

        Sweep atomicity: a multi-bucket sweep is NOT engine-atomic; bracket it
        with ``AsyncLLM.pause_generation()`` / ``resume_generation()``.
        Pre-try failures mutate nothing (the bucket is retryable); in-try
        failures abort the local cycle, and the coordinator must broadcast
        ``abort_update_weights`` to ALL ranks before retrying with a covering
        sweep, or committed weights diverge across ranks.

        Args:
            ipc_handles: Dictionary mapping device UUIDs to lists of (param_name, tensor_handle) tuples.
                        Each tensor_handle is a tuple of (func, args) for reconstructing the tensor.

        Raises:
            NotImplementedError: Capability preflight refused a submodule
                (e.g. EPLB-enabled MoE); nothing mutated, no cycle abort.
            ValueError: Device UUID missing or malformed payload; raised
                before any mutation, the bucket can simply be resent.
            Exception: Re-raised after aborting the local reload cycle (see
                ``_abort_weight_reload_cycle``); recover with a sweep covering
                everything the failed cycle(s) delivered (full resend always
                suffices).
        """
        # Everything before the try/abort boundary mutates no model state, so
        # a failure here must neither destroy weights nor abort the cycle.
        weights = None
        if ipc_handles is not None:
            model = self.engine.model_engine.model
            # Capability preflight (duck-typed, side-effect-free): an
            # incapable submodule must refuse HERE, not mid-way through the
            # destructive pre-hook walk after earlier modules were torn.
            if not hasattr(model, _PRE_RELOAD_LATCH_ATTR):
                for module in model.modules():
                    if hasattr(module, "check_reload_capability") and not getattr(
                        module, "_weights_removed", False
                    ):
                        module.check_reload_capability()
            logger.info("Update weights from IPC handles")
            device_uuid = get_device_uuid(self.device_id)

            if device_uuid not in ipc_handles:
                raise ValueError(f"Device UUID {device_uuid} not found in ipc_handles")

            weights = {}

            serialized_handles = ipc_handles[device_uuid]
            if isinstance(serialized_handles, str):
                # Data is base64-encoded pickled bytes - deserialize it
                # using restricted unpickler from tensorrt_llm.serialization
                logger.info("Deserializing base64-encoded weight handles")
                decoded_data = base64.b64decode(serialized_handles)
                disallowed_imports = {
                    "torch.storage": ["_load_from_bytes"],
                    "torch.hub": ["_load_local"],
                    "torch": ["save"],
                }
                # CUDA IPC tensor handles serialize torch rebuild helpers.
                # Keep deserialization default-deny by allowing only this
                # call site to import torch symbols, with disallowed imports
                # still taking precedence in serialization.Unpickler.
                approved_imports = {
                    "builtins": [
                        "list",
                        "tuple",
                        "str",
                        "int",
                        "float",
                        "bool",
                        "bytes",
                        "dict",
                        "NoneType",
                        "type",
                    ],
                }
                all_handles = serialization.loads(
                    decoded_data,
                    approved_imports=approved_imports,
                    approved_module_patterns=[r"^torch.*"],
                    disallowed_imports=disallowed_imports,
                )

                # Verify the result is a list as expected
                if not isinstance(all_handles, list):
                    raise ValueError(
                        f"Deserialized data must be a list, got {type(all_handles).__name__} instead"
                    )
            else:
                # Data is already in the correct format (backward compatibility)
                all_handles = serialized_handles

            # A mid-loop rebuild failure just drops the partial ``weights``
            # dict; its IPC mappings are freed at the next ipc_collect().
            for param_name, tensor_handle in all_handles:
                func, args = tensor_handle
                list_args = list(args)
                list_args[6] = self.device_id
                tensor = func(*list_args)
                weights[param_name] = tensor

        try:
            if ipc_handles is not None:
                # The pre-hook walk is destructive (params re-registered as
                # uninitialized storage), so it runs only after preconditions
                # passed and inside the abort boundary.
                if not hasattr(self.engine.model_engine.model, _PRE_RELOAD_LATCH_ATTR):
                    for module in self.engine.model_engine.model.modules():
                        if hasattr(module, "pre_reload_weights") and not getattr(
                            module, "_weights_removed", False
                        ):
                            module.pre_reload_weights()
                    setattr(self.engine.model_engine.model, _PRE_RELOAD_LATCH_ATTR, True)
                logger.info(f"weights key size: {len(weights)}")
                self.engine.model_engine.model_loader.reload(
                    self.engine.model_engine.model, weights, allow_partial_loading=True
                )
                del weights
                torch.cuda.ipc_collect()
            else:
                logger.info("Finalize update weights")
                # modules() yields the root model FIRST: the model-level
                # coverage audit must veto a torn cycle before any submodule
                # finalize hook runs.
                for module in self.engine.model_engine.model.modules():
                    if hasattr(module, "process_weights_after_loading") and not getattr(
                        module, "_weights_removed", False
                    ):
                        module.process_weights_after_loading()
                    if hasattr(module, "post_load_weights") and not getattr(
                        module, "_weights_removed", False
                    ):
                        module.post_load_weights()
                moe_load_balancer = getattr(self.engine.model_engine, "moe_load_balancer", None)
                if isinstance(moe_load_balancer, MoeLoadBalancer):
                    moe_load_balancer.register_weight_slots_after_to_cuda()
                    logger.info("moe_load_balancer finalizing model...")
                    moe_load_balancer.finalize_model()
                    logger.info("moe_load_balancer finalize model done")
                # Surface async CUDA errors from the finalize walk BEFORE the
                # latch delete so they hit this cycle's abort boundary.
                torch.cuda.synchronize()
                self.engine.reset_prefix_cache()
                # A bare finalize (no data bucket -> no walk) has no latch;
                # guard so the delete cannot trip the abort boundary.
                if hasattr(self.engine.model_engine.model, _PRE_RELOAD_LATCH_ATTR):
                    delattr(self.engine.model_engine.model, _PRE_RELOAD_LATCH_ATTR)

                # Done once after all buckets to avoid per-bucket cleanup overhead.
                gc.collect()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(
                "Encountered an error in update_weights; aborting the in-progress "
                "weight-reload cycle (recovery requires a fresh weight sweep that "
                "re-covers everything the failed cycle(s) delivered; a full "
                "resend always suffices)"
            )
            self._abort_weight_reload_cycle(e)
            raise

    def _abort_weight_reload_cycle(self, cause: BaseException) -> None:
        """Best-effort invalidation of the current weight-reload cycle.

        The latch must not survive a failure, else the next call skips
        ``pre_reload_weights`` and merges leftovers into the new cycle. The
        optional duck-typed ``abort_reload_cycle(reason)`` hook runs first and
        must not touch the device (CUDA may be poisoned); the latch clear is
        UNCONDITIONAL. Cleanup failures are logged, never raised.
        """
        try:
            model = self.engine.model_engine.model
        except Exception as cleanup_exc:
            logger.error(
                f"update_weights abort: model unavailable; skipping cleanup: {cleanup_exc!r}"
            )
            return
        try:
            abort_hook = getattr(model, "abort_reload_cycle", None)
            if callable(abort_hook):
                abort_hook(repr(cause))
        except Exception as cleanup_exc:
            logger.error(f"update_weights abort: abort_reload_cycle hook failed: {cleanup_exc!r}")
        try:
            if hasattr(model, _PRE_RELOAD_LATCH_ATTR):
                delattr(model, _PRE_RELOAD_LATCH_ATTR)
        except Exception as cleanup_exc:
            logger.error(f"update_weights abort: failed to clear pre-reload latch: {cleanup_exc!r}")

    def abort_update_weights(self, reason: str = "trainer-initiated abort") -> None:
        """Coordinator-broadcast abort of the in-progress weight-reload cycle.

        After ANY rank's ``update_weights`` raises, per-rank state diverges;
        the coordinator MUST broadcast this to ALL ranks before retrying so no
        rank commits a later finalize while another refuses. CPU-only;
        idempotent; safe to call when no cycle is in progress.

        Sticky veto (deliberate): the per-module ``_reload_outstanding``
        coverage debt is NOT cleared here -- abort + bare finalize must not
        bless empty_like garbage. Only a covering delivery or full load
        clears it.
        """
        # Deliberately NOT @control_action_decorator: only Python attrs the
        # forward never reads are mutated, and a drain could hang behind live
        # rollouts. Ordering vs a running update_weights relies on the worker
        # being a default SYNC Ray actor; revisit if it becomes concurrent.
        self._abort_weight_reload_cycle(RuntimeError(reason))

    def reset_prefix_cache(self) -> None:
        """Invalidate the KV cache prefix reuse state after weight updates."""
        self.engine.reset_prefix_cache()

    @control_action_decorator
    def wait_for_engine_idle(self) -> None:
        """Block until the engine has no active or queued requests."""
        pass

    def check_weights_updated(self) -> bool:
        """Check if the weights are updated to 0."""
        weights_updated = True
        for name, p in self.engine.model_engine.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(p, torch.zeros_like(p))
        return weights_updated

    def start_profile(self):
        torch.cuda.profiler.start()

    def stop_profile(self):
        torch.cuda.profiler.stop()
