# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import unittest

# isort: off
import torch
import torch.cuda.nvtx as nvtx
# isort: on

import tensorrt_llm
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod


def woq_assert_near_eq(ref, act, wTypeId):
    # match the scale in cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp
    if wTypeId == 1:
        bits_in_type = 8
    else:
        bits_in_type = 4
    quant_range_scale = 1.0 / float(1 << (bits_in_type - 1))

    max_val = torch.max(abs(ref)).item()
    atol = (max_val * quant_range_scale) * 1.5  # allow for rounding
    torch.testing.assert_close(ref, act, atol=atol, rtol=1e-7)


class TestMoEWeightOnlyGroupWiseQuantMatmul(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        tensorrt_llm.logger.set_level('error')

    def _woq_moe_groupwise_matmul(self,
                                  m,
                                  n,
                                  k,
                                  num_experts,
                                  dtype_str,
                                  has_pre_quant,
                                  has_alpha,
                                  top_k=2,
                                  group_size=128):

        dtype = tensorrt_llm._utils.str_dtype_to_torch(dtype_str)
        activation = torch.randn(m, k, dtype=dtype, device="cuda") * 0.1
        router = torch.randn((num_experts, k),
                             dtype=torch.float32,
                             device="cuda")

        num_weights_in_32_bits = 8
        assert n % num_weights_in_32_bits == 0, f"n must be a multiple of {num_weights_in_32_bits}"
        unprocessed_int_weight_1 = torch.randint(
            -2**31,
            2**31, (num_experts, k, n * 2 // num_weights_in_32_bits),
            dtype=torch.int32,
            device="cuda")
        unprocessed_int_weight_2 = torch.randint(
            -2**31,
            2**31, (num_experts, n, k // num_weights_in_32_bits),
            dtype=torch.int32,
            device="cuda")
        pre_quant_scale_1 = torch.randn(1, k, dtype=dtype, device="cuda")
        pre_quant_scale_2 = torch.randn(1, n, dtype=dtype, device="cuda")
        scale_1 = torch.randn(num_experts,
                              k // group_size,
                              n * 2,
                              dtype=torch.bfloat16,
                              device="cuda") * 0.1
        scale_2 = torch.randn(num_experts,
                              n // group_size,
                              k,
                              dtype=torch.bfloat16,
                              device="cuda") * 0.1
        alpha_1 = torch.randn(
            num_experts, 1, dtype=torch.float32, device="cuda") * 0.1
        alpha_2 = torch.randn(
            num_experts, 1, dtype=torch.float32, device="cuda") * 0.1

        unprocessed_weight_1 = unprocessed_int_weight_1.view(torch.int8)
        unprocessed_weight_2 = unprocessed_int_weight_2.view(torch.int8)

        unpacker = torch.ops.trtllm.unpack_int4_packed_tensor_to_int8
        packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4

        ref_q_weight_1 = unpacker(unprocessed_weight_1.cpu()).cuda()
        ref_q_weight_2 = unpacker(unprocessed_weight_2.cpu()).cuda()
        ref_weight_1 = ref_q_weight_1 * scale_1.repeat_interleave(group_size,
                                                                  dim=1)
        ref_weight_2 = ref_q_weight_2 * scale_2.repeat_interleave(group_size,
                                                                  dim=1)

        ###############################################################
        # scale interleave
        # scale_1 [E, K, N]
        scale_1 = scale_1.permute(0, 2, 1)  # [E, N, K]
        scale_1_interleaved = scale_1.reshape(scale_1.shape[0],
                                              scale_1.shape[1],
                                              (scale_1.shape[2] // 4),
                                              4)  # [E, N, K/4, 4]
        scale_1_interleaved = scale_1_interleaved.permute(0, 2, 1,
                                                          3)  # [E, K/4, N, 4]
        scale_1_interleaved = scale_1_interleaved.reshape(
            scale_1.shape[0], scale_1.shape[2] // 4,
            scale_1.shape[1] * 4)  # [E, K/4, N*4]
        scale_1_interleaved.contiguous()

        scale_2 = scale_2.permute(0, 2, 1)  # [E, N, K]
        scale_2_interleaved = scale_2.reshape(scale_2.shape[0],
                                              scale_2.shape[1],
                                              (scale_2.shape[2] // 4),
                                              4)  # [E, N, K/4, 4]
        scale_2_interleaved = scale_2_interleaved.permute(0, 2, 1,
                                                          3)  # [E, K/4, N, 4]
        scale_2_interleaved = scale_2_interleaved.reshape(
            scale_2.shape[0], scale_2.shape[2] // 4,
            scale_2.shape[1] * 4)  # [E, K/4, N*4]
        scale_2_interleaved.contiguous()
        ###############################################################

        inputs = activation.cuda().float()
        inputs_merged = inputs.view(-1, inputs.shape[-1])
        routing = torch.matmul(inputs_merged, router.T.float())
        router_probs = torch.softmax(routing, 1, dtype=inputs.dtype)
        topk = torch.topk(router_probs, top_k)

        weight_1 = ref_q_weight_1.permute(0, 2, 1).contiguous()
        weight_2 = ref_q_weight_2.permute(0, 2, 1).contiguous()
        weight_1 = packer(weight_1.cpu()).cuda()
        weight_2 = packer(weight_2.cpu()).cuda()
        weight_1 = weight_1.view(
            (num_experts, n * 2, k // 2)).view(torch.quint4x2)
        weight_2 = weight_2.view((num_experts, k, n // 2)).view(torch.quint4x2)

        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)
        selected_experts, final_scales = routing_method.apply(routing)

        # ref
        results = torch.zeros_like(inputs_merged)
        for i, (scales, experts) in enumerate(zip(topk.values, topk.indices)):
            scales /= sum(scales)
            input = inputs_merged[i, :]
            for scale, expert in zip(scales, experts):
                input = inputs_merged[i, :]
                fc1_qd = ref_weight_1[expert].cuda().float()
                if has_pre_quant:
                    input = input * pre_quant_scale_1.squeeze()
                if has_alpha:
                    input = input.to(torch.float8_e4m3fn).float()
                    fc1_qd = fc1_qd.to(torch.float8_e4m3fn).float()
                    fc1 = torch.matmul(input, fc1_qd) * alpha_1[expert]
                else:
                    fc1 = torch.matmul(input, fc1_qd)
                fc1, gate = fc1.chunk(2, dim=-1)
                fc1 = fc1 * torch.nn.functional.silu(gate)
                fc2_qd = ref_weight_2[expert].cuda().float()
                if has_pre_quant:
                    fc1 = fc1 * pre_quant_scale_2.squeeze()
                if has_alpha:
                    fc1 = fc1.to(torch.float8_e4m3fn).float()
                    fc2_qd = fc2_qd.to(torch.float8_e4m3fn).float()
                    final = torch.matmul(fc1, fc2_qd) * alpha_2[expert]
                else:
                    final = torch.matmul(fc1, fc2_qd)
                results[i] += scale * final
        ref = results.view(*inputs.shape).to(dtype)

        nvtx.range_push("profiler")
        profiler = torch.classes.trtllm.FusedMoeProfiler.get_instance(
            dtype, torch.quint4x2, dtype, False, False, has_alpha)
        profiler.run_profile(
            weight_2,
            top_k,
            1,
            0,
            1,
            0,
            [2, 4, 8, 16, 32, 64, 128]  # num_tokens_buckets
        )
        profile_ids = profiler.get_profile_ids(m, weight_2, top_k, num_experts)

        nvtx.range_pop()

        def run(profile_ids=None):
            nvtx.range_push("fused_moe op")
            output = torch.ops.trtllm.fused_moe(
                activation,
                selected_experts,
                final_scales,
                weight_1,
                weight_2,
                dtype,
                quant_scales=[
                    scale_1_interleaved, scale_2_interleaved,
                    pre_quant_scale_1 if has_pre_quant else torch.Tensor(),
                    pre_quant_scale_2 if has_pre_quant else torch.Tensor(),
                    torch.Tensor(),
                    torch.Tensor(), alpha_1 if has_alpha else torch.Tensor(),
                    alpha_2 if has_alpha else torch.Tensor()
                ],
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                profile_ids=profile_ids,
                use_w4afp8=has_alpha,
            )
            nvtx.range_pop()
            return output

        with torch.inference_mode():
            for _ in range(2):
                out = run()
            import triton
            nvtx.range_push("unprofiled")
            t = triton.testing.do_bench(lambda: run()) * 1000
            nvtx.range_pop()
            print(f"TRTLLM: {t:.3f} us")

        assert len(profile_ids) == 2
        with torch.inference_mode():
            nvtx.range_push("profiled single")
            output_with_profile = run(profile_ids)
            nvtx.range_pop()
            import triton
            nvtx.range_push("profiled")
            t = triton.testing.do_bench(lambda: run(profile_ids)) * 1000
            nvtx.range_pop()
            print(f"TRTLLM-optimized: {t:.3f} us")

        woq_assert_near_eq(ref, out, 2)
        woq_assert_near_eq(ref, output_with_profile, 2)
        print("==========================")
        print(ref)
        print(out)
        print(output_with_profile)

    def test_moe_w4a8(self, m, n, k, experts, dtype, topk):
        self._woq_moe_groupwise_matmul(m, n, k, experts, dtype, True, True,
                                       topk)


if __name__ == '__main__':
    t = TestMoEWeightOnlyGroupWiseQuantMatmul()
    ss = [
        (1, 14336, 4096, 8, "float16", 1), (1, 14336, 4096, 8, "bfloat16", 1),
        (64, 2048, 7168, 64, "float16", 1), (64, 2048, 7168, 64, "bfloat16", 1),
        (64, 7168, 2048, 64, "float16", 1), (64, 7168, 2048, 64, "bfloat16", 1)
    ]
    for s in ss:
        t.test_moe_w4a8(s[0], s[1], s[2], s[3], s[4], s[5])
