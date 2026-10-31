#!/usr/bin/env python3
"""Regression test for the SA3 key masks (causal diagonal and key padding).

K is stored permuted inside every 32-token group by the FP4 K quantizer; the masks
in mainloop_tma_ws.h must index by key *token*, not by S-tile column.  Before the
fix: causal cos ~0.71-0.79, 8 keys cos ~0.47, L=1000 cos ~0.979 (vs ~0.982).

usage (from outside the repo):
  PYTHONPATH=<worktree>/sageattention3_blackwell python <worktree>/bench/sa3_mask_test.py
"""
import torch

from sageattn3 import sageattn3_blackwell

THRESH = 0.978


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def main():
    torch.manual_seed(0)
    worst = 1.0
    cases = []
    for dt in (torch.bfloat16, torch.float16):
        for D in (64, 128):
            for L in (1000, 2048, 3001):
                for causal in (False, True):
                    cases.append((dt, D, L, L, causal))
            for nk in (4, 8, 16, 40, 136):  # few / odd number of keys, non-causal
                cases.append((dt, D, 256, nk, False))
    for dt, D, LQ, LK, causal in cases:
        q = torch.randn(1, 8, LQ, D, device="cuda", dtype=dt)
        k = torch.randn(1, 8, LK, D, device="cuda", dtype=dt)
        v = torch.randn(1, 8, LK, D, device="cuda", dtype=dt)
        o = sageattn3_blackwell(q, k, v, is_causal=causal)
        r = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)
        c = cos(o, r)
        worst = min(worst, c)
        flag = "" if c >= THRESH else "  <-- LOW"
        print(f"{str(dt)[6:]:>8} D{D:3d} Lq{LQ:5d} Lk{LK:5d} causal={int(causal)}  cos {c:.5f}{flag}")
    print(f"worst cos {worst:.5f} over {len(cases)} cases")
    if worst < THRESH:
        raise SystemExit("REGRESSION: SA3 mask")
    print("sa3 mask ok")


if __name__ == "__main__":
    main()
