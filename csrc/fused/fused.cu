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

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "../dispatch_utils.h"
#include "../utils.cuh"
#include "../reduction_utils.cuh"
#include "../numeric_conversion.cuh"
#include "../cp_async.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// GB10 (sm_121) additions, see GB10_PORTING.md §2.4/2.7:
//   ChannelAmaxKernel + TransposePadPermuteQuantKernel
//       V quantization in two DRAM passes (no FP16 transpose round trip).
//   L2ReduceQuantKernel
//       Fuses "reduce then quantize" (K: mean + int8, V: amax + fp8) into one
//       launch and re-reads each head from L2 (one head fits in GB10's 24 MB).
//       CTAs take tickets from an atomic counter, so a quantize CTA only ever
//       waits on reduce CTAs that already hold smaller tickets (no deadlock).
//   Everything here is bit-identical to the separate-kernel path; the Python
//   side selects between them with SAGEATTN_L2_FUSED.

enum class QuantType
{
  kInt8,
  kInt4,
};

template <typename T>
__device__ __forceinline__ float convert_to_float(T val)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  if constexpr (std::is_same<T, half>::value)
  {
    return __half2float(val);
  }
  else if constexpr (std::is_same<T, nv_bfloat16>::value)
  {
    return __bfloat162float(val);
  }
}

template <typename T>
__device__ __forceinline__ T convert_from_float(float val)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  if constexpr (std::is_same<T, half>::value)
  {
    return __float2half_rn(val);
  }
  else if constexpr (std::is_same<T, nv_bfloat16>::value)
  {
    return __float2bfloat16_rn(val);
  }
}

template <uint32_t head_dim, uint32_t BLOCK_SIZE, uint32_t num_pack_per_thread = 1, bool has_sm_scale = false, bool sub_mean = false, typename T>
__global__ void QuantInt8Kernel(T *__restrict__ input, T *__restrict__ mean, int8_t *__restrict__ output, float *__restrict__ scale, float sm_scale, const uint32_t num_tokens, 
                            const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                            const uint32_t stride_bz_output, const uint32_t stride_seq_output, const uint32_t stride_h_output,
                            const uint32_t stride_bz_scale, const uint32_t stride_h_scale)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");
  static_assert(num_pack_per_thread > 0, "The number of pack per thread must be greater than 0");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16
  constexpr uint32_t num_threads_per_token = head_dim / pack_size;

  static_assert(num_threads_per_token <= 32, "The number of threads per token must be less than or equal to warp size");

  T x_val[num_pack_per_thread][8];
  T mean_val[8];
  float x_val_float[num_pack_per_thread][8];
  float mean_val_float[8];

  uint32_t bx = blockIdx.x;
  uint32_t head_id = blockIdx.y;
  uint32_t batch_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t thread_base_token = bx * BLOCK_SIZE + thread_id / num_threads_per_token;
  T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
  T *mean_ptr_base = mean + batch_id * stride_bz_mean + head_id * stride_h_mean + thread_id % num_threads_per_token * pack_size;
  int8_t *output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + thread_base_token * stride_seq_output + thread_id % num_threads_per_token * pack_size;
  float *scale_ptr_base = scale + batch_id * stride_bz_scale + head_id * stride_h_scale + bx;

  if constexpr (sub_mean)
  {
    *(float4*)(&mean_val[0]) = *(float4*)(mean_ptr_base);
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      mean_val_float[j] = convert_to_float(mean_val[j]);
    }
  }

  constexpr uint32_t iter_stride = BLOCK_SIZE / num_pack_per_thread;

  // load the data
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    if (thread_base_token + i * iter_stride < num_tokens)
    {
      *(float4*)(&x_val[i][0]) = *(float4*)(input_ptr_base + i * iter_stride * stride_seq_input);
#pragma unroll
      for (uint32_t j = 0; j < 8; j++)
      {
        x_val_float[i][j] = convert_to_float(x_val[i][j]);
      }

      if constexpr (sub_mean)
      {
#pragma unroll
        for (uint32_t j = 0; j < 8; j++)
        {
          x_val_float[i][j] -= mean_val_float[j];
        }
      }

      if constexpr (has_sm_scale)
      {
#pragma unroll
        for (uint32_t j = 0; j < 8; j++)
        {
          x_val_float[i][j] *= sm_scale;
        }
      }
    }
    else
    {
#pragma unroll
      for (uint32_t j = 0; j < 8; j++)
      {
        x_val_float[i][j] = 0.0f;
      }
    }
  }

  float amax_val = 0.0000001f; // prevent from dividing by zero

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      amax_val = fmaxf(amax_val, fabsf(x_val_float[i][j]));
    }
  }

  __shared__ float s_amax;
  const float block_amax_val = vllm::blockReduceMax(amax_val);
  if (thread_id == 0)
  {
    s_amax = block_amax_val;
    scale_ptr_base[0] = s_amax / 127.0f;
  }

  __syncthreads();

  float tmp_scale = 127.0f / s_amax;

  char4 o_val[num_pack_per_thread][2];

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < 2; j += 1)
    {
      o_val[i][j] = make_char4(
        float_to_int8_rn(x_val_float[i][j * 4 + 0] * tmp_scale),
        float_to_int8_rn(x_val_float[i][j * 4 + 1] * tmp_scale),
        float_to_int8_rn(x_val_float[i][j * 4 + 2] * tmp_scale),
        float_to_int8_rn(x_val_float[i][j * 4 + 3] * tmp_scale)
      );
    }
  }

  // int8 result
#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    
    if (thread_base_token + i * iter_stride < num_tokens)
    {
      *reinterpret_cast<float2*>(output_ptr_base + i * iter_stride * stride_seq_output) = *reinterpret_cast<float2*>(&o_val[i][0]);
    }
  }
}

template <uint32_t head_dim, uint32_t CTA_SIZE, typename T>
__global__ void ChannelAmaxKernel(T *__restrict__ input, float *__restrict__ amax,
                                  const uint32_t num_tokens, const uint32_t num_heads,
                                  const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input)
{
  // One thread per feature column: the reads of a warp are consecutive halves
  // along head_dim, so each (b, h, d) column is streamed with fully coalesced
  // sectors.  Blocks split the sequence; the per-column max is folded with an
  // atomic max (values are non-negative, so the int bit pattern orders them).
  const uint32_t c = threadIdx.x;
  const uint32_t bh = blockIdx.y;
  const uint32_t b = bh / num_heads;
  const uint32_t h = bh - b * num_heads;

  const uint32_t start = blockIdx.x * CTA_SIZE;
  const uint32_t end = min(start + CTA_SIZE, num_tokens);
  const T *p = input + b * stride_bz_input + h * stride_h_input + c;

  float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
  uint32_t i = start;
  for (; i + 3 < end; i += 4)
  {
    m0 = fmaxf(m0, fabsf(convert_to_float(p[(i + 0) * stride_seq_input])));
    m1 = fmaxf(m1, fabsf(convert_to_float(p[(i + 1) * stride_seq_input])));
    m2 = fmaxf(m2, fabsf(convert_to_float(p[(i + 2) * stride_seq_input])));
    m3 = fmaxf(m3, fabsf(convert_to_float(p[(i + 3) * stride_seq_input])));
  }
  for (; i < end; i++)
  {
    m0 = fmaxf(m0, fabsf(convert_to_float(p[i * stride_seq_input])));
  }

  const float m = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
  atomicMax(reinterpret_cast<unsigned int *>(amax + (b * num_heads + h) * head_dim + c), __float_as_uint(m));
}

// Same layout work as TransposePadPermuteKernel, but the transposed tile is
// scaled by the per-channel factor and written straight out as FP8.  This
// removes the FP16 round trip (write + re-read of the whole tensor) that the
// two-pass version needs on memory-bound devices.
template <uint32_t head_dim, uint32_t CTA_SIZE, bool pad_zero = false, typename T>
__global__ void TransposePadPermuteQuantKernel(T *__restrict__ input, int8_t *__restrict__ output, float *__restrict__ amax,
                            const float scale_max, const uint32_t num_tokens,
                            const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_output, const uint32_t stride_d_output, const uint32_t stride_h_output)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16
  uint32_t num_threads_per_token = head_dim / pack_size;
  uint32_t num_threads_per_cta = CTA_SIZE / pack_size;

  uint32_t bx = blockIdx.x;
  uint32_t head_id = blockIdx.y;
  uint32_t batch_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t thread_base_token = bx * CTA_SIZE + thread_id / num_threads_per_token;

  T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
  int8_t *output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + bx * CTA_SIZE + thread_id % num_threads_per_cta * pack_size + thread_id / num_threads_per_cta * stride_d_output;

  __shared__ T shared_load[CTA_SIZE][head_dim];
  __shared__ T shared_store[head_dim][CTA_SIZE];

  // 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
  // permute on the seq dimension for fp8 mma
  uint32_t smem_load_row_base = ((thread_id / num_threads_per_token) / 16) * 16;
  uint32_t smem_load_row_mod = (thread_id / num_threads_per_token) % 16;
  uint32_t smem_load_row = smem_load_row_base + (smem_load_row_mod / 8) * 2 + ((smem_load_row_mod / 2) % 4) * 4 + (smem_load_row_mod % 2);

  constexpr cp_async::SharedMemFillMode fill_mode = pad_zero ? cp_async::SharedMemFillMode::kFillZero : cp_async::SharedMemFillMode::kNoFill;
  cp_async::pred_load_128b<cp_async::PrefetchMode::kNoPrefetch, fill_mode>(shared_load[smem_load_row] + thread_id % num_threads_per_token * pack_size, input_ptr_base, thread_base_token < num_tokens);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  uint32_t smem_row_base = thread_id % CTA_SIZE;
  uint32_t smem_col_base = thread_id / CTA_SIZE;
  uint32_t smem_col_stride = head_dim / 8;

  // TODO: use ldmatrix to do permutation
#pragma unroll
  for (uint32_t i = 0; i < 8; i++)
  {
    shared_store[smem_col_base + i * smem_col_stride][smem_row_base] = shared_load[smem_row_base][smem_col_base + i * smem_col_stride];
  }

  __syncthreads();

  const uint32_t d_idx = thread_id / num_threads_per_cta;
  const float amax_val = amax[(batch_id * gridDim.y + head_id) * head_dim + d_idx];
  const float recp_scale = (amax_val > 0.0f) ? (scale_max / amax_val) : 0.0f;

  T *src = &shared_store[d_idx][thread_id % num_threads_per_cta * pack_size];
  float vals[8];
#pragma unroll
  for (uint32_t j = 0; j < 8; j++)
  {
    vals[j] = convert_to_float(src[j]) * recp_scale;
  }
  uint32_t packed[2];
  floatx4_to_e4m3x4(packed, vals, vals + 2);
  floatx4_to_e4m3x4(packed + 1, vals + 4, vals + 6);

  *reinterpret_cast<uint2 *>(output_ptr_base) = *reinterpret_cast<uint2 *>(&packed[0]);
}

// ---------------------------------------------------------------------------
// L2-resident "reduce then quantize" kernels (GB10 / bandwidth-bound parts).
//
// Both K smoothing (mean over tokens, then per-block int8 with the mean
// subtracted) and V quantization (per-channel amax, then transpose + fp8) need
// a full reduction over the sequence before the quantize pass can start, so
// the input is streamed from DRAM twice.  One head of K or V is only
// kv_len * head_dim * 2 bytes (8.4 MB at 32k x 128), which fits in L2 (24 MB
// on GB10), so if the quantize pass for a head runs right after that head's
// reduction, its re-read is served by L2 instead of DRAM.
//
// A single launch does both phases.  Each CTA grabs a ticket from an atomic
// counter (so tickets follow the order in which CTAs actually start) and the
// ticket stream is laid out as
//   lag 1: reduce(0) | reduce(1) quant(0) | reduce(2) quant(1) | ... | quant(BH-1)
//   lag 0: reduce(0) quant(0) | reduce(1) quant(1) | ...
// A quant CTA of head h only waits for the reduce CTAs of head h, which all
// hold smaller tickets and never wait themselves, so the spin cannot deadlock.
// lag 1 overlaps the reduce of head h+1 with the quant of head h (no idle
// SMs) but keeps ~2.5 heads live in L2; lag 0 keeps ~1 head live, for long
// sequences where a head is a large fraction of L2.
//
// The cross-CTA reduction is deterministic: every reduce CTA writes its
// partial to a workspace slot and the last one to finish folds the slots in
// index order.
// ---------------------------------------------------------------------------
constexpr uint32_t kL2QuantTokens = 64;  // tokens per quant CTA (= BLKK for K, CTA_SIZE for V)

struct L2PhaseInfo
{
  bool is_reduce;
  uint32_t bh;
  uint32_t idx;
};

__device__ __forceinline__ L2PhaseInfo l2_phase_from_ticket(uint32_t t, uint32_t num_bh, uint32_t R, uint32_t Qn, uint32_t lag)
{
  L2PhaseInfo info;
  if (lag == 0)
  {
    const uint32_t g = t / (R + Qn);
    const uint32_t r = t - g * (R + Qn);
    info.bh = g;
    info.is_reduce = r < R;
    info.idx = info.is_reduce ? r : r - R;
    return info;
  }
  if (t < R)
  {
    info.is_reduce = true; info.bh = 0; info.idx = t;
    return info;
  }
  const uint32_t u = t - R;
  const uint32_t g = 1 + u / (R + Qn);
  const uint32_t r = u - (g - 1) * (R + Qn);
  if (g < num_bh && r < R)
  {
    info.is_reduce = true; info.bh = g; info.idx = r;
  }
  else
  {
    info.is_reduce = false; info.bh = g - 1; info.idx = (g < num_bh) ? r - R : r;
  }
  return info;
}

// IS_V == false: K path.  reduce = column sum -> km (T, like torch.mean),
//                         quant  = per-64-token-block int8 of (k - km).
// IS_V == true : V path.  reduce = column abs-max -> amax (fp32),
//                         quant  = transpose/pad/permute + per-channel fp8.
template <uint32_t head_dim, bool IS_V, typename T>
__global__ void __launch_bounds__(head_dim * 8) L2ReduceQuantKernel(
                            const T *__restrict__ input, int8_t *__restrict__ output,
                            T *__restrict__ km_out, float *__restrict__ amax_out,
                            float *__restrict__ k_scale,
                            float *__restrict__ partial, int *__restrict__ ws,
                            const float scale_max, const uint32_t num_tokens, const uint32_t num_heads,
                            const uint32_t R, const uint32_t Qn, const uint32_t lag, const uint32_t red_tokens,
                            const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_output, const uint32_t stride_x_output, const uint32_t stride_h_output,
                            const uint32_t stride_bz_scale, const uint32_t stride_h_scale)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");
  constexpr uint32_t pack_size = 8;
  constexpr uint32_t threads_per_token = head_dim / pack_size;
  constexpr uint32_t num_threads = head_dim * 8;
  constexpr uint32_t rows_per_iter = num_threads / threads_per_token;  // = 64
  static_assert(rows_per_iter == kL2QuantTokens, "tile mismatch");

  constexpr uint32_t smem_bytes = rows_per_iter * head_dim * sizeof(float);  // >= 2 * 64 * head_dim * sizeof(T)
  __shared__ __align__(16) uint8_t smem[smem_bytes];
  __shared__ uint32_t s_ticket;
  __shared__ bool s_is_last;
  __shared__ float s_amax;

  const uint32_t tid = threadIdx.x;
  const uint32_t num_bh = gridDim.x / (R + Qn);
  int *ticket_ctr = ws;
  int *done_ctr = ws + 1;
  int *ready = ws + 1 + num_bh;

  if (tid == 0)
    s_ticket = atomicAdd(ticket_ctr, 1);
  __syncthreads();
  const L2PhaseInfo info = l2_phase_from_ticket(s_ticket, num_bh, R, Qn, lag);
  const uint32_t b = info.bh / num_heads;
  const uint32_t h = info.bh - b * num_heads;
  const T *in_base = input + b * stride_bz_input + h * stride_h_input;
  const uint32_t row = tid / threads_per_token;
  const uint32_t col = (tid % threads_per_token) * pack_size;

  if (info.is_reduce)
  {
    // ---- phase 1: column sum / abs-max over kL2RedTokens tokens ----
    float acc[8];
#pragma unroll
    for (uint32_t j = 0; j < 8; j++) acc[j] = 0.0f;

    const uint32_t start = info.idx * red_tokens;
#pragma unroll 4
    for (uint32_t it = 0; it < red_tokens / rows_per_iter; it++)
    {
      const uint32_t token = start + it * rows_per_iter + row;
      if (token < num_tokens)
      {
        T x[8];
        *(float4 *)(&x[0]) = __ldg((const float4 *)(in_base + token * stride_seq_input + col));
#pragma unroll
        for (uint32_t j = 0; j < 8; j++)
        {
          const float f = convert_to_float(x[j]);
          if constexpr (IS_V) acc[j] = fmaxf(acc[j], fabsf(f));
          else acc[j] += f;
        }
      }
    }

    float (*red)[head_dim] = reinterpret_cast<float (*)[head_dim]>(smem);
#pragma unroll
    for (uint32_t j = 0; j < 8; j++) red[row][col + j] = acc[j];
    __syncthreads();

    float *my_partial = partial + ((size_t)info.bh * R + info.idx) * head_dim;
    if (tid < head_dim)
    {
      float v = 0.0f;
      for (uint32_t r = 0; r < rows_per_iter; r++)
      {
        if constexpr (IS_V) v = fmaxf(v, red[r][tid]);
        else v += red[r][tid];
      }
      my_partial[tid] = v;
      __threadfence();
    }
    __syncthreads();
    if (tid == 0)
      s_is_last = (atomicAdd(done_ctr + info.bh, 1) == (int)R - 1);
    __syncthreads();

    if (s_is_last)
    {
      __threadfence();
      if (tid < head_dim)
      {
        const float *p = partial + (size_t)info.bh * R * head_dim + tid;
        float v = 0.0f;
        for (uint32_t r = 0; r < R; r++)
        {
          const float x = __ldcg(p + (size_t)r * head_dim);
          if constexpr (IS_V) v = fmaxf(v, x);
          else v += x;
        }
        if constexpr (IS_V) amax_out[info.bh * head_dim + tid] = v;
        else km_out[info.bh * head_dim + tid] = convert_from_float<T>(v * (1.0f / (float)num_tokens));
        __threadfence();
      }
      __syncthreads();
      if (tid == 0)
        atomicExch(ready + info.bh, 1);
    }
    return;
  }

  // ---- phase 2: wait for this head's reduction, then quantize a 64-token tile ----
  if (tid == 0)
  {
    volatile int *flag = ready + info.bh;
    while (*flag == 0) __nanosleep(64);
    __threadfence();
  }
  __syncthreads();

  if constexpr (!IS_V)
  {
    // same math as QuantInt8Kernel<head_dim, 64, 1, false, /*sub_mean=*/true>
    const uint32_t token = info.idx * kL2QuantTokens + row;
    T mean_val[8];
    *(int4 *)(&mean_val[0]) = __ldcg((const int4 *)(km_out + info.bh * head_dim + col));

    float xf[8];
    if (token < num_tokens)
    {
      T x[8];
      *(float4 *)(&x[0]) = *(const float4 *)(in_base + token * stride_seq_input + col);
#pragma unroll
      for (uint32_t j = 0; j < 8; j++) xf[j] = convert_to_float(x[j]) - convert_to_float(mean_val[j]);
    }
    else
    {
#pragma unroll
      for (uint32_t j = 0; j < 8; j++) xf[j] = 0.0f;
    }

    float amax_val = 0.0000001f;
#pragma unroll
    for (uint32_t j = 0; j < 8; j++) amax_val = fmaxf(amax_val, fabsf(xf[j]));
    const float block_amax_val = vllm::blockReduceMax(amax_val);
    if (tid == 0)
    {
      s_amax = block_amax_val;
      k_scale[b * stride_bz_scale + h * stride_h_scale + info.idx] = s_amax / 127.0f;
    }
    __syncthreads();
    const float tmp_scale = 127.0f / s_amax;

    char4 o_val[2];
#pragma unroll
    for (uint32_t j = 0; j < 2; j++)
    {
      o_val[j] = make_char4(
        float_to_int8_rn(xf[j * 4 + 0] * tmp_scale),
        float_to_int8_rn(xf[j * 4 + 1] * tmp_scale),
        float_to_int8_rn(xf[j * 4 + 2] * tmp_scale),
        float_to_int8_rn(xf[j * 4 + 3] * tmp_scale));
    }
    if (token < num_tokens)
    {
      int8_t *out = output + b * stride_bz_output + h * stride_h_output + token * stride_x_output + col;
      *reinterpret_cast<float2 *>(out) = *reinterpret_cast<float2 *>(&o_val[0]);
    }
  }
  else
  {
    // same math as TransposePadPermuteQuantKernel<head_dim, 64, /*pad_zero=*/true>
    constexpr uint32_t CTA_SIZE = kL2QuantTokens;
    constexpr uint32_t num_threads_per_cta = CTA_SIZE / pack_size;
    T (*shared_load)[head_dim] = reinterpret_cast<T (*)[head_dim]>(smem);
    T (*shared_store)[CTA_SIZE] = reinterpret_cast<T (*)[CTA_SIZE]>(smem + CTA_SIZE * head_dim * sizeof(T));

    const uint32_t token = info.idx * CTA_SIZE + row;
    const uint32_t smem_load_row = (row / 16) * 16 + ((row % 16) / 8) * 2 + (((row % 16) / 2) % 4) * 4 + (row % 2);
    cp_async::pred_load_128b<cp_async::PrefetchMode::kNoPrefetch, cp_async::SharedMemFillMode::kFillZero>(
        shared_load[smem_load_row] + col, in_base + token * stride_seq_input + col, token < num_tokens);
    cp_async::commit_group();
    cp_async::wait_group<0>();
    __syncthreads();

    const uint32_t smem_row_base = tid % CTA_SIZE;
    const uint32_t smem_col_base = tid / CTA_SIZE;
    constexpr uint32_t smem_col_stride = head_dim / 8;
#pragma unroll
    for (uint32_t i = 0; i < 8; i++)
    {
      shared_store[smem_col_base + i * smem_col_stride][smem_row_base] = shared_load[smem_row_base][smem_col_base + i * smem_col_stride];
    }
    __syncthreads();

    const uint32_t d_idx = tid / num_threads_per_cta;
    const float amax_val = __ldcg(amax_out + info.bh * head_dim + d_idx);
    const float recp_scale = (amax_val > 0.0f) ? (scale_max / amax_val) : 0.0f;

    const T *src = &shared_store[d_idx][tid % num_threads_per_cta * pack_size];
    float vals[8];
#pragma unroll
    for (uint32_t j = 0; j < 8; j++) vals[j] = convert_to_float(src[j]) * recp_scale;
    uint32_t packed[2];
    floatx4_to_e4m3x4(packed, vals, vals + 2);
    floatx4_to_e4m3x4(packed + 1, vals + 4, vals + 6);

    int8_t *out = output + b * stride_bz_output + h * stride_h_output + info.idx * CTA_SIZE
                  + tid % num_threads_per_cta * pack_size + d_idx * stride_x_output;
    *reinterpret_cast<uint2 *>(out) = *reinterpret_cast<uint2 *>(&packed[0]);
  }
}

template <uint32_t head_dim, uint32_t BLOCK_SIZE, uint32_t num_pack_per_thread = 1, typename T>
__global__ void SubMeanKernel(T *__restrict__ input, T *__restrict__ mean, half *__restrict__ output, const uint32_t num_tokens, 
                            const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                            const uint32_t stride_bz_output, const uint32_t stride_seq_output, const uint32_t stride_h_output)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");
  static_assert(num_pack_per_thread > 0, "The number of pack per thread must be greater than 0");

  using T2 = typename std::conditional<std::is_same<T, half>::value, half2, nv_bfloat162>::type;

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16
  constexpr uint32_t num_threads_per_token = head_dim / pack_size;

  static_assert(num_threads_per_token <= 32, "The number of threads per token must be less than or equal to warp size");

  T2 x_val[num_pack_per_thread][4];
  T2 mean_val[4];

  uint32_t bx = blockIdx.x;
  uint32_t head_id = blockIdx.y;
  uint32_t batch_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t thread_base_token = bx * BLOCK_SIZE + thread_id / num_threads_per_token;
  T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
  T *mean_ptr_base = mean + batch_id * stride_bz_mean + head_id * stride_h_mean + thread_id % num_threads_per_token * pack_size;
  half *output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + thread_base_token * stride_seq_output + thread_id % num_threads_per_token * pack_size;

  *(float4*)(&mean_val[0]) = *(float4*)(mean_ptr_base);

  constexpr uint32_t iter_stride = BLOCK_SIZE / num_pack_per_thread;

  // load the data
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    if (thread_base_token + i * iter_stride < num_tokens)
    {
      *(float4*)(&x_val[i][0]) = *(float4*)(input_ptr_base + i * iter_stride * stride_seq_input);
#pragma unroll
      for (uint32_t j = 0; j < 4; j++)
      {
        x_val[i][j] = __hsub2(x_val[i][j], mean_val[j]);

        if constexpr (std::is_same<T, nv_bfloat16>::value)
        {
          ((half2*)x_val[i])[j] = __float22half2_rn(__bfloat1622float2(x_val[i][j])); 
        }
      }
    }
  }

#pragma unroll
  for (uint32_t i = 0; i < num_pack_per_thread; i++)
  {
    if (thread_base_token + i * iter_stride < num_tokens)
    {
      *reinterpret_cast<float4*>(output_ptr_base + i * iter_stride * stride_seq_output) = *reinterpret_cast<float4*>(&x_val[i][0]);
    }
  }
}

template <uint32_t head_dim, uint32_t CTA_SIZE, bool pad_zero=false, typename T>
__global__ void TransposePadPermuteKernel(T *__restrict__ input, T *__restrict__ output, const uint32_t num_tokens,
                            const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_output, const uint32_t stride_d_output, const uint32_t stride_h_output)
{

  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "Only half and bfloat16 are supported");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16
  uint32_t num_threads_per_token = head_dim / pack_size;
  uint32_t num_threads_per_cta = CTA_SIZE / pack_size;

  uint32_t bx = blockIdx.x;
  uint32_t head_id = blockIdx.y;
  uint32_t batch_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t thread_base_token = bx * CTA_SIZE + thread_id / num_threads_per_token;

  T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
  T* output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + bx * CTA_SIZE + thread_id % num_threads_per_cta * pack_size + thread_id / num_threads_per_cta * stride_d_output;

  __shared__ T shared_load[CTA_SIZE][head_dim];
  __shared__ T shared_store[head_dim][CTA_SIZE];

  // 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
  // permute on the seq dimension for fp8 mma
  uint32_t smem_load_row_base = ((thread_id / num_threads_per_token) / 16) * 16;
  uint32_t smem_load_row_mod = (thread_id / num_threads_per_token) % 16;
  uint32_t smem_load_row = smem_load_row_base + (smem_load_row_mod  / 8) * 2 + ((smem_load_row_mod / 2) % 4) * 4 + (smem_load_row_mod % 2);

  constexpr cp_async::SharedMemFillMode fill_mode = pad_zero ? cp_async::SharedMemFillMode::kFillZero : cp_async::SharedMemFillMode::kNoFill;
  cp_async::pred_load_128b<cp_async::PrefetchMode::kNoPrefetch, fill_mode>(shared_load[smem_load_row] + thread_id % num_threads_per_token * pack_size, input_ptr_base, thread_base_token < num_tokens);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  uint32_t smem_row_base = thread_id % CTA_SIZE;
  uint32_t smem_col_base = thread_id / CTA_SIZE;
  uint32_t smem_col_stride = head_dim / 8;

  // TODO: use ldmatrix to do permutation
#pragma unroll
  for (uint32_t i = 0; i < 8; i++)
  {
    shared_store[smem_col_base + i * smem_col_stride][smem_row_base] = shared_load[smem_row_base][smem_col_base + i * smem_col_stride];
  }

  __syncthreads();

  *(float4*)(output_ptr_base) = *(float4*)(&shared_store[thread_id / num_threads_per_cta][thread_id % num_threads_per_cta * pack_size]);
}


template<uint32_t pad_size, bool sub_mean = false, typename T>
__global__ void MeanScaleKernel(T *__restrict__ input, int8_t *__restrict__ output, float *__restrict__ mean, float *__restrict__ scale, const float scale_max, const uint32_t num_tokens,
                            const uint32_t stride_bz_input, const uint32_t stride_d_input, const uint32_t stride_h_input,
                            const uint32_t stride_bz_output, const uint32_t stride_d_output, const uint32_t stride_h_output,
                            const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                            const uint32_t stride_bz_scale, const uint32_t stride_h_scale)
{
  static_assert(std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value, "Only half and bfloat16 are supported");

  constexpr uint32_t pack_size = 8; // float4 contains 8 half or 8 bfloat16

  uint32_t head_id = blockIdx.x;
  uint32_t batch_id = blockIdx.y;
  uint32_t d_id = blockIdx.z;
  uint32_t thread_id = threadIdx.x;

  uint32_t num_threads = blockDim.x;
  uint32_t gmem_stride = num_threads * pack_size;
  // pad the number of tokens to 16 to deal with fp8 permute in previous kernel
  uint32_t fp8_padded_num_tokens = (num_tokens + 15) / 16 * 16;
  uint32_t num_iters = fp8_padded_num_tokens / gmem_stride + ((fp8_padded_num_tokens % gmem_stride) > thread_id * pack_size);

  T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + d_id * stride_d_input + thread_id * pack_size;
  int8_t *output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + d_id * stride_d_output + thread_id * pack_size;

  T x_val[8];
  float x_val_float[8];
  uint32_t x_val_fp8[2];

  float max_val = - 1000000.0f;
  float min_val = 1000000.0f;
  float sum_val = 0.0f;

  for (int i = 0; i < num_iters; i++)
  {
    *(float4*)(&x_val[0]) = *(float4*)(input_ptr_base + i * gmem_stride);
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      float x_temp = convert_to_float(x_val[j]);
      max_val = fmaxf(max_val, x_temp);
      min_val = fminf(min_val, x_temp);

      if constexpr (sub_mean)
      {
        sum_val += x_temp;
      }
    }
  }

  // reduce
  __shared__ float s_amax_val;
  __shared__ float s_mean_val;

  float block_max_val = vllm::blockReduceMax(max_val);
  float block_min_val = vllm::blockReduceMin(min_val);
  float block_sum_val;

  if constexpr (sub_mean)
  {
    block_sum_val = vllm::blockReduceSum(sum_val);
  }

  if (thread_id == 0)
  {
    s_mean_val = block_sum_val / fp8_padded_num_tokens;

    if constexpr (sub_mean)
    {
      s_amax_val = fmaxf(fabsf(block_max_val - s_mean_val), fabsf(block_min_val - s_mean_val));
      mean[batch_id * stride_bz_mean + head_id * stride_h_mean + d_id] = s_mean_val;
    }
    else
    {
      s_amax_val = fmaxf(fabsf(block_max_val), fabsf(block_min_val));
    }

    scale[batch_id * stride_bz_scale + head_id * stride_h_scale + d_id] = s_amax_val / scale_max;
  }

  __syncthreads();

  float mean_val = s_mean_val;
  float recp_scale = scale_max / s_amax_val;

  // recalculate num_iters to cover all fp8 output tokens to prevent nan in random initialization
  uint32_t padded_num_tokens = (num_tokens + pad_size - 1) / pad_size * pad_size;
  num_iters = padded_num_tokens / gmem_stride + ((padded_num_tokens % gmem_stride) > thread_id * pack_size);

  for (int i = 0; i < num_iters; i++)
  {
    *(float4*)(&x_val[0]) = *(float4*)(input_ptr_base + i * gmem_stride);
#pragma unroll
    for (uint32_t j = 0; j < 8; j++)
    {
      x_val_float[j] = convert_to_float(x_val[j]);
      if constexpr (sub_mean)
      {
        x_val_float[j] = (x_val_float[j] - mean_val) * recp_scale;
      }
      else
      {
        x_val_float[j] *= recp_scale;
      }
    }

    floatx4_to_e4m3x4(x_val_fp8, x_val_float, x_val_float + 2);
    floatx4_to_e4m3x4(x_val_fp8 + 1, x_val_float + 4, x_val_float + 6);

    *(uint2*)(output_ptr_base + i * gmem_stride) = *(uint2*)(&x_val_fp8[0]);
  }
}

void quant_per_block_int8_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                float sm_scale,
                int block_size,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);
  
  CHECK_DTYPE(output, torch::kInt8);
  CHECK_DTYPE(scale, torch::kFloat);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  int stride_bz_input = input.stride(0);
  int stride_bz_output = output.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_seq_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_seq_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  auto input_dtype = input.scalar_type();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {

        CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
        CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);

        dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);

        constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

        dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

        QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, true, false, c_type><<<grid, block>>>(
          reinterpret_cast<c_type*>(input.data_ptr()),
          nullptr,
          output.data_ptr<int8_t>(),
          reinterpret_cast<float*>(scale.data_ptr()),
          sm_scale,
          num_tokens,
          stride_bz_input, stride_seq_input, stride_h_input,
          0, 0,
          stride_bz_output, stride_seq_output, stride_h_output,
          scale.stride(0), scale.stride(1)
        );
      });
    });
  });
}

void quant_per_block_int8_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int block_size,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);
  
  CHECK_DTYPE(output, torch::kInt8);
  CHECK_DTYPE(scale, torch::kFloat);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  int stride_bz_input = input.stride(0);
  int stride_bz_output = output.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_seq_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_seq_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  auto input_dtype = input.scalar_type();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {

        CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
        CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);

        dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);

        constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

        dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

        QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, false, c_type><<<grid, block>>>(
          reinterpret_cast<c_type*>(input.data_ptr()),
          nullptr,
          output.data_ptr<int8_t>(),
          reinterpret_cast<float*>(scale.data_ptr()),
          0.0f,
          num_tokens,
          stride_bz_input, stride_seq_input, stride_h_input,
          0, 0,
          stride_bz_output, stride_seq_output, stride_h_output,
          scale.stride(0), scale.stride(1)
        );
      });
    });
  });
}

void quant_per_block_int8_fuse_sub_mean_cuda(
                torch::Tensor input,
                torch::Tensor mean,
                torch::Tensor output,
                torch::Tensor scale,
                int block_size,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(mean);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);
  
  CHECK_DTYPE(output, torch::kInt8);
  CHECK_DTYPE(scale, torch::kFloat);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(mean, 3);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  int stride_bz_input = input.stride(0);
  int stride_bz_output = output.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_seq_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_seq_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  auto input_dtype = input.scalar_type();
  auto mean_dtype = mean.scalar_type();

  TORCH_CHECK(input_dtype == mean_dtype, "Input and mean must have the same data type");

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {

        CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
        CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
        CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);

        dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);

        constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

        dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

        QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, true, c_type><<<grid, block>>>(
          reinterpret_cast<c_type*>(input.data_ptr()),
          reinterpret_cast<c_type*>(mean.data_ptr()),
          output.data_ptr<int8_t>(),
          reinterpret_cast<float*>(scale.data_ptr()),
          0.0f,
          num_tokens,
          stride_bz_input, stride_seq_input, stride_h_input,
          mean.stride(0), mean.stride(1),
          stride_bz_output, stride_seq_output, stride_h_output,
          scale.stride(0), scale.stride(1)
        );
      });
    });
  });
}

// use block size 128 and warp_block size 32
void quant_per_warp_int8_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int block_size,
                int warp_block_size,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);
  
  CHECK_DTYPE(output, torch::kInt8);
  CHECK_DTYPE(scale, torch::kFloat);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  int stride_bz_input = input.stride(0);
  int stride_bz_output = output.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_seq_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_seq_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  auto input_dtype = input.scalar_type();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
      DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_BLOCK_SIZE, {
        DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {

          CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
          CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE));

          dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE), num_heads, batch_size);

          constexpr int num_pack_per_thread = (WARP_BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

          dim3 block(WARP_BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

          QuantInt8Kernel<HEAD_DIM, WARP_BLOCK_SIZE, num_pack_per_thread, false, false, c_type><<<grid, block>>>(
            reinterpret_cast<c_type*>(input.data_ptr()),
            nullptr,
            output.data_ptr<int8_t>(),
            reinterpret_cast<float*>(scale.data_ptr()),
            0.0,
            num_tokens,
            stride_bz_input, stride_seq_input, stride_h_input,
            0, 0,
            stride_bz_output, stride_seq_output, stride_h_output,
            scale.stride(0), scale.stride(1)
          );
        });
      });
    });
  });
}

void sub_mean_cuda(
                torch::Tensor input,
                torch::Tensor mean,
                torch::Tensor output,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(mean);
  CHECK_CUDA(output);
  
  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(output);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(mean, 3);
  CHECK_DIMS(output, 4);

  CHECK_DTYPE(output, torch::kHalf);

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  int stride_bz_input = input.stride(0);
  int stride_bz_output = output.stride(0);

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_seq_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_seq_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  auto input_dtype = input.scalar_type();
  auto mean_dtype = mean.scalar_type();

  TORCH_CHECK(input_dtype == mean_dtype, "Input and mean must have the same data type");

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        
        CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
        CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
  
        constexpr int BLOCK_SIZE = (HEAD_DIM == 128) ? 64 : 128;

        dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);

        constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / 8) + 1023) / 1024;

        dim3 block(BLOCK_SIZE * (HEAD_DIM / 8) / num_pack_per_thread);

        SubMeanKernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread><<<grid, block>>>(
          reinterpret_cast<c_type*>(input.data_ptr()),
          reinterpret_cast<c_type*>(mean.data_ptr()),
          reinterpret_cast<half*>(output.data_ptr()),
          num_tokens,
          stride_bz_input, stride_seq_input, stride_h_input,
          mean.stride(0), mean.stride(1),
          stride_bz_output, stride_seq_output, stride_h_output
        );
    });
  });
}

void transpose_pad_permute_cuda(
                torch::Tensor input,
                torch::Tensor output,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);

  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);

  constexpr int CTA_SIZE = 64;

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  int stride_bz_input = input.stride(0);
  int stride_bz_output = output.stride(0);

  int num_tokens, padded_num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_d_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_d_output = output.stride(1);
    stride_h_output = output.stride(2);

    padded_num_tokens = (num_tokens + CTA_SIZE - 1) / CTA_SIZE * CTA_SIZE;

    CHECK_SHAPE(output, batch_size, head_dim, num_heads, padded_num_tokens);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_d_output = output.stride(2);
    stride_h_output = output.stride(1);

    padded_num_tokens = (num_tokens + CTA_SIZE - 1) / CTA_SIZE * CTA_SIZE;
    CHECK_SHAPE(output, batch_size, num_heads, head_dim, padded_num_tokens);
  }

  auto input_dtype = input.scalar_type();
  auto output_dtype = output.scalar_type();

  TORCH_CHECK(input_dtype == output_dtype, "Input and output must have the same data type");

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 grid(padded_num_tokens / CTA_SIZE, num_heads, batch_size);

      static_assert(CTA_SIZE * HEAD_DIM <= 8192);

      dim3 block(CTA_SIZE * (HEAD_DIM / 8));

      TransposePadPermuteKernel<HEAD_DIM, CTA_SIZE, true, c_type><<<grid, block>>>(
        reinterpret_cast<c_type*>(input.data_ptr()),
        reinterpret_cast<c_type*>(output.data_ptr()),
        num_tokens,
        stride_bz_input, stride_seq_input, stride_h_input,
        stride_bz_output, stride_d_output, stride_h_output
      );
    });
  });
}

void channel_amax_cuda(
                torch::Tensor input,
                torch::Tensor amax,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(amax);
  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(amax);
  CHECK_DIMS(input, 4);
  CHECK_DIMS(amax, 3);

  constexpr int CTA_SIZE = 1024;

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  int num_tokens, num_heads;
  int stride_seq_input, stride_h_input;
  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
  }

  CHECK_SHAPE(amax, batch_size, num_heads, head_dim);

  auto input_dtype = input.scalar_type();
  TORCH_CHECK(amax.scalar_type() == torch::kFloat, "amax must be float32");

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 grid((num_tokens + CTA_SIZE - 1) / CTA_SIZE, num_heads * batch_size);
      dim3 block(HEAD_DIM);
      ChannelAmaxKernel<HEAD_DIM, CTA_SIZE, c_type><<<grid, block>>>(
        reinterpret_cast<c_type*>(input.data_ptr()),
        reinterpret_cast<float*>(amax.data_ptr()),
        num_tokens, num_heads,
        input.stride(0), stride_seq_input, stride_h_input
      );
    });
  });
}

void transpose_pad_permute_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor amax,
                double scale_max,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(amax);
  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(amax);
  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(amax, 3);

  constexpr int CTA_SIZE = 64;

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);

  int num_tokens, padded_num_tokens, num_heads;
  int stride_seq_input, stride_h_input, stride_d_output, stride_h_output;
  if (tensor_layout == 0)
  {
    num_tokens = input.size(1);
    num_heads = input.size(2);
    stride_seq_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_d_output = output.stride(1);
    stride_h_output = output.stride(2);
    padded_num_tokens = (num_tokens + CTA_SIZE - 1) / CTA_SIZE * CTA_SIZE;
    CHECK_SHAPE(output, batch_size, head_dim, num_heads, padded_num_tokens);
  }
  else
  {
    num_tokens = input.size(2);
    num_heads = input.size(1);
    stride_seq_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_d_output = output.stride(2);
    stride_h_output = output.stride(1);
    padded_num_tokens = (num_tokens + CTA_SIZE - 1) / CTA_SIZE * CTA_SIZE;
    CHECK_SHAPE(output, batch_size, num_heads, head_dim, padded_num_tokens);
  }

  auto input_dtype = input.scalar_type();
  TORCH_CHECK(output.scalar_type() == torch::kFloat8_e4m3fn ||
              output.scalar_type() == torch::kChar ||
              output.scalar_type() == torch::kByte, "output must be fp8(e4m3) or int8/uint8 storage");

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 grid(padded_num_tokens / CTA_SIZE, num_heads, batch_size);
      dim3 block(CTA_SIZE * (HEAD_DIM / 8));
      TransposePadPermuteQuantKernel<HEAD_DIM, CTA_SIZE, true, c_type><<<grid, block>>>(
        reinterpret_cast<c_type*>(input.data_ptr()),
        reinterpret_cast<int8_t*>(output.data_ptr()),
        reinterpret_cast<float*>(amax.data_ptr()),
        static_cast<float>(scale_max),
        num_tokens,
        input.stride(0), stride_seq_input, stride_h_input,
        output.stride(0), stride_d_output, stride_h_output
      );
    });
  });
}

void scale_fuse_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int num_tokens,
                float scale_max,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(scale);

  // CHECK_DTYPE(output, torch::kInt8);
  CHECK_DTYPE(scale, torch::kFloat);

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.size(0);
  const int num_tokens_padded = input.size(3);

  int stride_bz_input = input.stride(0);
  int stride_bz_output = output.stride(0);

  int num_heads, head_dim;
  int stride_d_input, stride_h_input, stride_d_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_heads = input.size(2);
    head_dim = input.size(1);
    stride_d_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_d_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_heads = input.size(1);
    head_dim = input.size(2);
    stride_d_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_d_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
  CHECK_SHAPE(scale, batch_size, num_heads, head_dim);

  constexpr int CTA_SIZE = 256;

  dim3 grid(num_heads, batch_size, head_dim);
  dim3 block(CTA_SIZE);

  auto input_dtype = input.scalar_type();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    MeanScaleKernel<64, false, c_type><<<grid, block>>>(
      reinterpret_cast<c_type*>(input.data_ptr()),
      reinterpret_cast<int8_t*>(output.data_ptr()),
      nullptr,
      reinterpret_cast<float*>(scale.data_ptr()),
      scale_max,
      num_tokens,
      stride_bz_input, stride_d_input, stride_h_input,
      stride_bz_output, stride_d_output, stride_h_output,
      0, 0,
      scale.stride(0), scale.stride(1)
    );
  });
}

void mean_scale_fuse_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor mean,
                torch::Tensor scale,
                int num_tokens,
                float scale_max,
                int tensor_layout)
{
  CHECK_CUDA(input);
  CHECK_CUDA(output);
  CHECK_CUDA(mean);
  CHECK_CUDA(scale);

  // CHECK_DTYPE(output, torch::kInt8);
  CHECK_DTYPE(mean, torch::kFloat);
  CHECK_DTYPE(scale, torch::kFloat);

  CHECK_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(mean);
  CHECK_CONTIGUOUS(scale);

  CHECK_DIMS(input, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(mean, 3);
  CHECK_DIMS(scale, 3);

  const int batch_size = input.size(0);
  const int num_tokens_padded = input.size(3);

  int stride_bz_input = input.stride(0);
  int stride_bz_output = output.stride(0);

  int num_heads, head_dim;
  int stride_d_input, stride_h_input, stride_d_output, stride_h_output;

  if (tensor_layout == 0)
  {
    num_heads = input.size(2);
    head_dim = input.size(1);
    stride_d_input = input.stride(1);
    stride_h_input = input.stride(2);
    stride_d_output = output.stride(1);
    stride_h_output = output.stride(2);
  }
  else
  {
    num_heads = input.size(1);
    head_dim = input.size(2);
    stride_d_input = input.stride(2);
    stride_h_input = input.stride(1);
    stride_d_output = output.stride(2);
    stride_h_output = output.stride(1);
  }

  CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
  CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
  CHECK_SHAPE(scale, batch_size, num_heads, head_dim);

  constexpr int CTA_SIZE = 256;

  dim3 grid(num_heads, batch_size, head_dim);
  dim3 block(CTA_SIZE);

  auto input_dtype = input.scalar_type();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
    MeanScaleKernel<64, true, c_type><<<grid, block>>>(
      reinterpret_cast<c_type*>(input.data_ptr()),
      reinterpret_cast<int8_t*>(output.data_ptr()),
      reinterpret_cast<float*>(mean.data_ptr()),
      reinterpret_cast<float*>(scale.data_ptr()),
      scale_max,
      num_tokens,
      stride_bz_input, stride_d_input, stride_h_input,
      stride_bz_output, stride_d_output, stride_h_output,
      mean.stride(0), mean.stride(1),
      scale.stride(0), scale.stride(1)
    );
  });
}

// Launch helper shared by the K and V entry points of L2ReduceQuantKernel.
template <bool IS_V>
static void launch_l2_reduce_quant(torch::Tensor input, torch::Tensor output,
                                   torch::Tensor km_out, torch::Tensor amax_out, torch::Tensor scale,
                                   float scale_max, int tensor_layout, int lag, int red_tokens,
                                   int64_t stride_bz_output, int64_t stride_x_output, int64_t stride_h_output)
{
  const int batch_size = input.size(0);
  const int head_dim = input.size(3);
  const int num_tokens = tensor_layout == 0 ? input.size(1) : input.size(2);
  const int num_heads = tensor_layout == 0 ? input.size(2) : input.size(1);
  const int stride_seq_input = tensor_layout == 0 ? input.stride(1) : input.stride(2);
  const int stride_h_input = tensor_layout == 0 ? input.stride(2) : input.stride(1);

  const uint32_t num_bh = batch_size * num_heads;
  if (red_tokens <= 0) red_tokens = 1024;
  TORCH_CHECK(red_tokens % 64 == 0, "red_tokens must be a multiple of 64");
  // lag must be explicit: a "pick by L2 size" heuristic was measured wrong on GB10
  // (it selected lag 0 at 32k, which is the slower schedule).  Python passes the
  // measured value (see sageattention/quant.py: L2_LAG).
  TORCH_CHECK(lag == 0 || lag == 1, "lag must be 0 or 1");
  const uint32_t R = (num_tokens + red_tokens - 1) / red_tokens;
  const uint32_t Qn = (num_tokens + kL2QuantTokens - 1) / kL2QuantTokens;
  TORCH_CHECK((uint64_t)num_bh * (R + Qn) < (1ull << 31), "grid too large");

  auto ws = torch::zeros({1 + 2 * (int64_t)num_bh}, input.options().dtype(torch::kInt));
  auto partial = torch::empty({(int64_t)num_bh, (int64_t)R, head_dim}, input.options().dtype(torch::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      dim3 grid(num_bh * (R + Qn));
      dim3 block(HEAD_DIM * 8);
      L2ReduceQuantKernel<HEAD_DIM, IS_V, c_type><<<grid, block, 0, stream>>>(
        reinterpret_cast<const c_type*>(input.data_ptr()),
        reinterpret_cast<int8_t*>(output.data_ptr()),
        IS_V ? nullptr : reinterpret_cast<c_type*>(km_out.data_ptr()),
        IS_V ? reinterpret_cast<float*>(amax_out.data_ptr()) : nullptr,
        IS_V ? nullptr : reinterpret_cast<float*>(scale.data_ptr()),
        partial.data_ptr<float>(), ws.data_ptr<int>(),
        scale_max, num_tokens, num_heads, R, Qn, (uint32_t)lag, (uint32_t)red_tokens,
        input.stride(0), stride_seq_input, stride_h_input,
        stride_bz_output, stride_x_output, stride_h_output,
        IS_V ? 0 : scale.stride(0), IS_V ? 0 : scale.stride(1)
      );
    });
  });
}

// K smoothing + per-block (BLKK = 64) int8 quantization in one launch.
// km_out: [b, h_kv, head_dim] in the dtype of `input` (== k.mean(seq)).
void mean_quant_per_block_int8_cuda(
                torch::Tensor input,
                torch::Tensor km_out,
                torch::Tensor output,
                torch::Tensor scale,
                int tensor_layout,
                int lag,
                int red_tokens)
{
  CHECK_CUDA(input); CHECK_CUDA(km_out); CHECK_CUDA(output); CHECK_CUDA(scale);
  CHECK_DTYPE(output, torch::kInt8);
  CHECK_DTYPE(scale, torch::kFloat);
  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(km_out); CHECK_CONTIGUOUS(output); CHECK_CONTIGUOUS(scale);
  CHECK_DIMS(input, 4); CHECK_DIMS(km_out, 3); CHECK_DIMS(output, 4); CHECK_DIMS(scale, 3);
  TORCH_CHECK(km_out.scalar_type() == input.scalar_type(), "km_out must have the dtype of input");

  const int num_tokens = tensor_layout == 0 ? input.size(1) : input.size(2);
  const int num_heads = tensor_layout == 0 ? input.size(2) : input.size(1);
  CHECK_SHAPE(km_out, input.size(0), num_heads, input.size(3));
  CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
  CHECK_SHAPE(scale, input.size(0), num_heads, (num_tokens + kL2QuantTokens - 1) / kL2QuantTokens);

  launch_l2_reduce_quant<false>(input, output, km_out, torch::Tensor(), scale, 0.0f, tensor_layout, lag, red_tokens,
                                output.stride(0),
                                tensor_layout == 0 ? output.stride(1) : output.stride(2),
                                tensor_layout == 0 ? output.stride(2) : output.stride(1));
}

// Per-channel amax + transpose/pad/permute + fp8 quantization in one launch.
// Output identical to channel_amax_cuda + transpose_pad_permute_quant_cuda.
void amax_transpose_pad_permute_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor amax,
                double scale_max,
                int tensor_layout,
                int lag,
                int red_tokens)
{
  CHECK_CUDA(input); CHECK_CUDA(output); CHECK_CUDA(amax);
  CHECK_LASTDIM_CONTIGUOUS(input);
  CHECK_CONTIGUOUS(output); CHECK_CONTIGUOUS(amax);
  CHECK_DIMS(input, 4); CHECK_DIMS(output, 4); CHECK_DIMS(amax, 3);
  TORCH_CHECK(amax.scalar_type() == torch::kFloat, "amax must be float32");
  TORCH_CHECK(output.scalar_type() == torch::kFloat8_e4m3fn ||
              output.scalar_type() == torch::kChar ||
              output.scalar_type() == torch::kByte, "output must be fp8(e4m3) or int8/uint8 storage");

  const int batch_size = input.size(0);
  const int head_dim = input.size(3);
  const int num_tokens = tensor_layout == 0 ? input.size(1) : input.size(2);
  const int num_heads = tensor_layout == 0 ? input.size(2) : input.size(1);
  const int padded = (num_tokens + kL2QuantTokens - 1) / kL2QuantTokens * kL2QuantTokens;
  CHECK_SHAPE(amax, batch_size, num_heads, head_dim);
  if (tensor_layout == 0) {
    CHECK_SHAPE(output, batch_size, head_dim, num_heads, padded);
  } else {
    CHECK_SHAPE(output, batch_size, num_heads, head_dim, padded);
  }

  launch_l2_reduce_quant<true>(input, output, torch::Tensor(), amax, torch::Tensor(),
                               static_cast<float>(scale_max), tensor_layout, lag, red_tokens,
                               output.stride(0),
                               tensor_layout == 0 ? output.stride(1) : output.stride(2),
                               tensor_layout == 0 ? output.stride(2) : output.stride(1));
}
