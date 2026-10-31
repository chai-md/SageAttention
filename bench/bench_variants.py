#!/usr/bin/env python3
"""Compare SageAttention 2.2.0 kernel variants on the current GPU (sm121 tuning)."""
import itertools
import sys

import torch

sys.path.insert(0, "/home/chai/SageAttention/bench")
from bench_gb10 import cuda_time, reference, err  # noqa: E402

from sageattention import sageattn_qk_int8_pv_fp8_cuda  # noqa: E402


def main():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    shapes = [(1, 24, 8192, 128), (1, 24, 16384, 128)]
    print(f"{'shape':>22} {'gran':>10} {'pv_accum':>10} {'ms':>8} {'maxerr':>8} {'cos':>8}")
    for shape in shapes:
        B, H, L, D = shape
        q = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        ref = reference(q, k, v, False)
        for gran, pv in itertools.product(["per_warp", "per_thread"], ["fp32", "fp32+fp32", "fp32+fp16"]):
            try:
                o = sageattn_qk_int8_pv_fp8_cuda(q, k, v, qk_quant_gran=gran, pv_accum_dtype=pv)
                t = cuda_time(lambda: sageattn_qk_int8_pv_fp8_cuda(q, k, v, qk_quant_gran=gran, pv_accum_dtype=pv))
                me, mae, cos = err(o, ref)
                print(f"{str(shape):>22} {gran:>10} {pv:>10} {t:8.3f} {me:8.4f} {cos:8.5f}")
            except Exception as e:
                print(f"{str(shape):>22} {gran:>10} {pv:>10}   FAILED {type(e).__name__}: {str(e)[:60]}")
        del q, k, v, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
