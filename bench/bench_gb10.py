#!/usr/bin/env python3
"""GB10 (sm_121) benchmark / correctness harness for SageAttention 2.2.0 + SageAttention3.

Run from anywhere; do NOT run with CWD=/home/chai/SageAttention or the source tree
shadows the installed extension modules.

  python bench/bench_gb10.py --suite sa2
  python bench/bench_gb10.py --suite sa3
  python bench/bench_gb10.py --suite sa2 --shape 1,24,32768,128 --dtype bf16
"""
import argparse
import os
import sys
import time

import torch


def cuda_time(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def sdpa_time(q, k, v, is_causal=False, dtype=torch.float16):
    # HND layout -> SDPA expects B,H,L,D
    return cuda_time(lambda: torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=is_causal))


def reference(q, k, v, is_causal=False):
    qf, kf, vf = q.float(), k.float(), v.float()
    return torch.nn.functional.scaled_dot_product_attention(qf, kf, vf, is_causal=is_causal)


def err(o, ref):
    o = o.float()
    diff = (o - ref).abs()
    cos = torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
    return diff.max().item(), diff.mean().item(), cos


def parse_shape(s):
    return tuple(int(x) for x in s.split(","))


def bench_sa2(shapes, dtype, is_causal, stages):
    import sageattention
    from sageattention import sageattn, sageattn_qk_int8_pv_fp8_cuda
    print(f"\n### SageAttention 2.2.0  dtype={dtype} causal={is_causal}")
    print(f"{'shape':>28} {'SDPA ms':>9} {'Sage ms':>9} {'speedup':>8} {'maxerr':>8} {'mearr':>8} {'cos':>8}")
    rows = []
    for B, H, L, D in shapes:
        q = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        ref = reference(q, k, v, is_causal) if L <= 16384 else None
        t_sdpa = sdpa_time(q[:, :, :L].contiguous(), k.contiguous(), v.contiguous(), is_causal)
        o = sageattn(q, k, v, tensor_layout="HND", is_causal=is_causal)
        t_sage = cuda_time(lambda: sageattn(q, k, v, tensor_layout="HND", is_causal=is_causal))
        if ref is not None:
            me, mae, cos = err(o, ref)
            print(f"{str((B,H,L,D)):>28} {t_sdpa:9.3f} {t_sage:9.3f} {t_sdpa/t_sage:8.2f}x {me:8.4f} {mae:8.5f} {cos:8.5f}")
        else:
            print(f"{str((B,H,L,D)):>28} {t_sdpa:9.3f} {t_sage:9.3f} {t_sdpa/t_sage:8.2f}x        -        -        -")
        rows.append((B, H, L, D, t_sdpa, t_sage))
        if stages:
            stage_breakdown(q, k, v, dtype, is_causal)
        del q, k, v, ref, o
        torch.cuda.empty_cache()
    return rows


def stage_breakdown(q, k, v, dtype, is_causal):
    """Time each preprocessing stage + kernel for the sm121 dispatch path.

    Follows the real dispatch: with SAGEATTN_L2_FUSED=1 (default) the K mean is
    computed inside the K quantizer and V's amax inside the transpose/quantize
    kernel, so the separate `k.mean` pass does not happen.
    """
    import sageattention.quant as _q
    from sageattention.quant import per_channel_fp8
    from sageattention import sm89_compile
    B, H, L, D = q.shape
    fused = getattr(_q, "L2_FUSED", False)
    if fused:
        smooth_k = _q.per_warp_int8_smooth_k
        t_k = cuda_time(lambda: smooth_k(q, k, tensor_layout="HND", BLKQ=128, WARPQ=32))
        q_int8, q_scale, k_int8, k_scale, km = smooth_k(q, k, tensor_layout="HND", BLKQ=128, WARPQ=32)
        t_mean = 0.0
        kv_label = "quantK+mean(fused)"
    else:
        from sageattention.quant import per_warp_int8 as per_warp_int8_cuda
        km = k.mean(dim=2, keepdim=True)
        t_mean = cuda_time(lambda: k.mean(dim=2, keepdim=True))
        t_k = cuda_time(lambda: per_warp_int8_cuda(q, k, km, tensor_layout="HND", BLKQ=128, WARPQ=32, BLKK=64))
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout="HND", BLKQ=128, WARPQ=32, BLKK=64)
        kv_label = "per_warp_int8(q,k)"
    t_v = cuda_time(lambda: per_channel_fp8(v, tensor_layout="HND", scale_max=2.25, smooth_v=False))
    v_fp8, v_scale, vm = per_channel_fp8(v, tensor_layout="HND", scale_max=2.25, smooth_v=False)
    o = torch.empty_like(q)
    t_kernel = cuda_time(lambda: sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(
        q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, 1, int(is_causal), 2, D ** -0.5, 0))
    total = t_mean + t_k + t_v + t_kernel
    print(f"    stages: k.mean {t_mean*1e3:7.1f}us | {kv_label} {t_k*1e3:7.1f}us | "
          f"per_channel_fp8(v) {t_v*1e3:7.1f}us | attn kernel {t_kernel*1e3:8.1f}us | sum {total*1e3:8.1f}us")


def bench_sa3(shapes, dtype, is_causal):
    from sageattn3 import sageattn3_blackwell
    print(f"\n### SageAttention3 (FP4 Blackwell)  dtype={dtype} causal={is_causal}")
    print(f"{'shape':>28} {'SDPA ms':>9} {'SA3 ms':>9} {'speedup':>8} {'maxerr':>8} {'mearr':>8} {'cos':>8}")
    for B, H, L, D in shapes:
        if D > 128:
            print(f"{str((B,H,L,D)):>28}   skipped (head_dim>128 unsupported)")
            continue
        q = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        ref = reference(q, k, v, is_causal) if L <= 16384 else None
        t_sdpa = sdpa_time(q, k, v, is_causal)
        try:
            o = sageattn3_blackwell(q.clone(), k.clone(), v.clone(), is_causal=is_causal)
            t3 = cuda_time(lambda: sageattn3_blackwell(q, k.clone(), v, is_causal=is_causal))
        except Exception as e:
            print(f"{str((B,H,L,D)):>28}   FAILED: {type(e).__name__}: {e}")
            del q, k, v
            torch.cuda.empty_cache()
            continue
        if ref is not None:
            me, mae, cos = err(o, ref)
            print(f"{str((B,H,L,D)):>28} {t_sdpa:9.3f} {t3:9.3f} {t_sdpa/t3:8.2f}x {me:8.4f} {mae:8.5f} {cos:8.5f}")
        else:
            print(f"{str((B,H,L,D)):>28} {t_sdpa:9.3f} {t3:9.3f} {t_sdpa/t3:8.2f}x        -        -        -")
        del q, k, v, ref, o
        torch.cuda.empty_cache()


DEFAULT_SA2 = [(1, 8, 4096, 128), (1, 24, 8192, 128), (1, 24, 16384, 128), (1, 24, 32768, 128)]
DEFAULT_SA3 = [(1, 8, 4096, 128), (1, 24, 8192, 128), (1, 24, 16384, 128)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["sa2", "sa3"], default="sa2")
    ap.add_argument("--shape", action="append", default=None)
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--stages", action="store_true")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}  SMs={props.multi_processor_count}  cc={props.major}.{props.minor}  "
          f"L2={props.L2_cache_size/2**20:.0f}MB  torch={torch.__version__}")

    if args.suite == "sa2":
        shapes = [parse_shape(s) for s in args.shape] if args.shape else DEFAULT_SA2
        bench_sa2(shapes, dtype, args.causal, args.stages)
    else:
        shapes = [parse_shape(s) for s in args.shape] if args.shape else DEFAULT_SA3
        bench_sa3(shapes, dtype, args.causal)


if __name__ == "__main__":
    main()
