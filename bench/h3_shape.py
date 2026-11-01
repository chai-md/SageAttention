#!/usr/bin/env python3
"""SageAttention cost at MiniMax-H3's real shape.

H3 (minimax_h3_fl2va_pruned_int8): 50 DiT blocks, dim 5376, 56 heads x 128,
patch 2 over a 16x/4x VAE -> tokens = ceil(frames/4) * (w/32) * (h/32).
0.4 MP = 864x480 -> 27*15 = 405 tokens/frame.
"""
import sys

import torch

sys.path.insert(0, "/home/chai/SageAttention/bench")
from bench_gb10 import cuda_time  # noqa: E402

from sageattention import sageattn  # noqa: E402
from sageattn3 import sageattn3_blackwell  # noqa: E402

BLOCKS = 50
H, D = 56, 128


def main():
    torch.manual_seed(0)
    for frames in (121, 241):
        L = ((frames + 3) // 4) * 405
        q = torch.randn(1, H, L, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        flops = 4.0 * L * L * H * D  # QK + PV

        t2 = cuda_time(lambda: sageattn(q, k, v), warmup=2, iters=5)
        t3 = cuda_time(lambda: sageattn3_blackwell(q, k, v), warmup=2, iters=5)
        f2 = flops / (t2 / 1e3) / 1e12
        f3 = flops / (t3 / 1e3) / 1e12
        print(f"frames={frames:4d}  L={L:6d}  per layer: SA2 {t2:7.3f} ms  SA3 {t3:7.3f} ms"
              f"   ({f2:5.0f} / {f3:5.0f} TFLOPS)")
        print(f"             x{BLOCKS} blocks: SA2 {t2*BLOCKS:7.1f} ms  SA3 {t3*BLOCKS:7.1f} ms"
              f"   per forward ({flops*BLOCKS/1e12:.1f} TFLOP attention total)")
        del q, k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
