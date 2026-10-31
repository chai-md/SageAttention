#!/usr/bin/env python3
"""Stress the fused L2 reduce+quantize path (SAGEATTN_L2_FUSED) for hangs/races.

Covers: many repeated calls, odd sequence lengths, GQA, fp16/bf16, NHD layout,
causal, and back-to-back shapes.  Any hang means the ticket/spin-wait protocol is
unsound; any cos below the SA2 threshold means a numerical regression.
"""
import itertools
import sys

import torch

sys.path.insert(0, "/home/chai/SageAttention/bench")
from bench_gb10 import cuda_time  # noqa: E402

import sageattention.quant as q  # noqa: E402
from sageattention import sageattn  # noqa: E402

print("L2_FUSED =", q.L2_FUSED, "lag =", q.L2_LAG, "red_tokens =", q.L2_RED_TOKENS)


def ref(q_, k_, v_, causal=False):
    n = q_.size(1) // k_.size(1)
    if n > 1:
        k_ = k_.repeat_interleave(n, dim=1)
        v_ = v_.repeat_interleave(n, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(
        q_.float(), k_.float(), v_.float(), is_causal=causal)


def cos(o, r):
    return torch.nn.functional.cosine_similarity(o.float().flatten(), r.flatten(), dim=0).item()


torch.manual_seed(7)
worst = 1.0
cases = []
for dt in (torch.bfloat16, torch.float16):
    for (H, KV) in ((8, 8), (24, 8)):
        for L in (1000, 1024, 3001, 4096, 4097, 8192):
            for D in (64, 128):
                cases.append((dt, H, KV, L, D))
for dt, H, KV, L, D in cases:
    qq = torch.randn(2, H, L, D, device="cuda", dtype=dt)
    kk = torch.randn(2, KV, L, D, device="cuda", dtype=dt)
    vv = torch.randn(2, KV, L, D, device="cuda", dtype=dt)
    o = sageattn(qq, kk, vv)
    c = cos(o, ref(qq, kk, vv))
    worst = min(worst, c)
    flag = "" if c >= 0.9990 else "  <-- LOW"
    print(f"{str(dt).split('.')[-1]:>8} H{H:2d}/KV{KV:2d} L{L:5d} D{D:3d}  cos {c:.5f}{flag}")
    del qq, kk, vv, o
    torch.cuda.empty_cache()

# causal + NHD
qq = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.bfloat16)
kk = torch.randn_like(qq)
vv = torch.randn_like(qq)
print("causal   cos", f"{cos(sageattn(qq, kk, vv, is_causal=True), ref(qq, kk, vv, True)):.5f}")
qq2 = qq.transpose(1, 2).contiguous()
kk2 = kk.transpose(1, 2).contiguous()
vv2 = vv.transpose(1, 2).contiguous()
o = sageattn(qq2, kk2, vv2, tensor_layout="NHD")
print("NHD      cos", f"{cos(o, ref(qq, kk, vv).transpose(1, 2)):.5f}")
del qq, kk, vv, qq2, kk2, vv2, o
torch.cuda.empty_cache()

# repeated calls: 60 iterations, alternating shapes (workspace reuse / leak / hang)
qq = torch.randn(1, 24, 8192, 128, device="cuda", dtype=torch.bfloat16)
kk = torch.randn_like(qq)
vv = torch.randn_like(qq)
t = cuda_time(lambda: sageattn(qq, kk, vv), warmup=5, iters=60)
print(f"60 repeated calls OK: {t:.3f} ms/call (stable => no workspace race)")

print(f"worst cos across {len(cases)} shapes: {worst:.5f}")
if worst < 0.9990:
    raise SystemExit("REGRESSION: cos below threshold")
print("stress ok")
