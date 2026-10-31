#!/usr/bin/env python3
"""SA3 `per_block_mean` knob on GB10: speed vs accuracy.

per_block_mean=False replaces the per-128-token Q mean with a single global mean,
which shrinks `delta_s` from [B,H,L/128,L] fp32 to [B,H,1,L] and removes its
`.float()` conversion + the big k^T matmul output.  On a bandwidth-starved part
that is ~3.5 ms of 21.9 ms at L=16384 - but it changes the numerics, so measure
the accuracy cost on both iid Gaussian and DC-heavy inputs (real attention
activations carry a large per-channel offset, which is what the block mean is
there to remove).
"""
import sys

import torch

sys.path.insert(0, "/home/chai/SageAttention/bench")
from bench_gb10 import cuda_time  # noqa: E402

from sageattn3 import sageattn3_blackwell  # noqa: E402


def reference(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())


def make(B, H, L, D, dc, dtype=torch.bfloat16):
    q = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
    k = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
    v = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
    if dc:
        # per-channel offset with a slow drift along the sequence: this is what
        # "smooth_k / block mean" exists for.
        off = (torch.randn(1, H, 1, D, device="cuda", dtype=torch.float32) * dc).bfloat16()
        drift = (torch.linspace(-1, 1, L, device="cuda").view(1, 1, L, 1) * dc * 0.5).bfloat16()
        k = k + off + drift
        q = q + off * 0.5
    return q, k, v


def main():
    for dc in (0.0, 5.0):
        print(f"\n=== per-channel DC offset = {dc} ===")
        print(f"{'shape':>22} {'mode':>14} {'ms':>8} {'maxerr':>8} {'cos':>8}")
        for B, H, L, D in [(1, 24, 8192, 128), (1, 24, 16384, 128)]:
            q, k, v = make(B, H, L, D, dc)
            ref = reference(q, k, v)
            for mode in (True, False):
                fn = lambda: sageattn3_blackwell(q, k, v, per_block_mean=mode)
                o = fn()
                t = cuda_time(fn)
                me = (o.float() - ref).abs().max().item()
                cos = torch.nn.functional.cosine_similarity(o.float().flatten(), ref.flatten(), dim=0).item()
                print(f"{str((B,H,L,D)):>22} {'per_block' if mode else 'global':>14} {t:8.3f} {me:8.4f} {cos:8.5f}")
            del q, k, v, ref
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
