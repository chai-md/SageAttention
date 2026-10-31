/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/extension.h>

void quant_per_block_int8_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                float sm_scale,
                int block_size,
                int tensor_layout);

void quant_per_block_int8_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int block_size,
                int tensor_layout);

void quant_per_block_int8_fuse_sub_mean_cuda(
                torch::Tensor input,
                torch::Tensor mean,
                torch::Tensor output,
                torch::Tensor scale,
                int block_size,
                int tensor_layout);

void quant_per_warp_int8_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int block_size,
                int warp_block_size,
                int tensor_layout);

void sub_mean_cuda(
                torch::Tensor input,
                torch::Tensor mean,
                torch::Tensor output,
                int tensor_layout);

void transpose_pad_permute_cuda(
                torch::Tensor input,
                torch::Tensor output,
                int tensor_layout);

// GB10: per-(batch, head, channel) amax of V; one DRAM pass, coalesced.
void channel_amax_cuda(
                torch::Tensor input,
                torch::Tensor amax,
                int tensor_layout);

// GB10: transpose/pad/permute + per-channel FP8 quantization in one kernel,
// using the amax from channel_amax_cuda (bit-identical to the two-step path).
void transpose_pad_permute_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor amax,
                double scale_max,
                int tensor_layout);

void scale_fuse_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int num_tokens,
                float scale_max,
                int tensor_layout);

void mean_scale_fuse_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor mean,
                torch::Tensor scale,
                int num_tokens,
                float scale_max,
                int tensor_layout);

// GB10: one launch, two phases - column-sum the K mean, then per-64-token int8
// quantize with that mean subtracted.  The quantize pass re-reads the head from
// L2; identical output to "k.mean() + quant_per_block_int8_fuse_sub_mean".
// lag: 1 overlaps reduce(h+1) with quantize(h), 0 is strictly per head.
void mean_quant_per_block_int8_cuda(
                torch::Tensor input,
                torch::Tensor km_out,
                torch::Tensor output,
                torch::Tensor scale,
                int tensor_layout,
                int lag,
                int red_tokens);

// GB10: one launch - per-channel amax of V, then transpose/pad/permute + FP8.
void amax_transpose_pad_permute_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor amax,
                double scale_max,
                int tensor_layout,
                int lag,
                int red_tokens);
