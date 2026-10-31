#!/usr/bin/env python3
"""Compare SageAttention 2.2.0 back-ends (sm80 pv-fp16 vs sm89 pv-fp8) on this GPU."""
import sys

import torch

sys.path.insert(0, "/home/chai/SageAttention/bench")
from bench_gb10 import cuda_time, reference, err  # noqa: E402

from sageattention import (  # noqa: E402
    sageattn,
    sageattn_qk_int8_pv_fp16_cuda,
    sageattn_qk_int8_pv_fp8_cuda,
)


def main():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    print(f"{'shape':>22} {'backend':>26} {'ms':>8} {'maxerr':>8} {'cos':>8}")
    for B, H, L, D in [(1, 24, 8192, 128), (1, 24, 16384, 128)]:
        q = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        ref = reference(q, k, v, False)
        variants = {
            "auto": lambda: sageattn(q, k, v),
            "pv_fp16_sm80(fp32acc)": lambda: sageattn_qk_int8_pv_fp16_cuda(q, k, v, pv_accum_dtype="fp32"),
            "pv_fp16_sm80(fp16acc)": lambda: sageattn_qk_int8_pv_fp16_cuda(q, k, v, pv_accum_dtype="fp16"),
            "pv_fp8(per_warp,fp32)": lambda: sageattn_qk_int8_pv_fp8_cuda(q, k, v, qk_quant_gran="per_warp", pv_accum_dtype="fp32"),
            "pv_fp8(per_warp,fp32+fp16)": lambda: sageattn_qk_int8_pv_fp8_cuda(q, k, v, qk_quant_gran="per_warp", pv_accum_dtype="fp32+fp16"),
        }
        for name, fn in variants.items():
            try:
                o = fn()
                t = cuda_time(fn)
                me, mae, cos = err(o, ref)
                print(f"{str((B,H,L,D)):>22} {name:>26} {t:8.3f} {me:8.4f} {cos:8.5f}")
            except Exception as e:
                print(f"{str((B,H,L,D)):>22} {name:>26}   FAILED {type(e).__name__}: {str(e)[:50]}")
        del q, k, v, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
