#!/usr/bin/env python3
"""Smoke tests: fp16, GQA, causal, head_dim 64/128, NHD layout, SA3 too."""
import sys

import torch

sys.path.insert(0, "/home/chai/SageAttention/bench")
from bench_gb10 import cuda_time  # noqa: E402

from sageattention import sageattn  # noqa: E402
from sageattn3 import sageattn3_blackwell  # noqa: E402


def ref(q, k, v, causal=False):
    n = q.size(1) // k.size(1)
    if n > 1:
        k = k.repeat_interleave(n, dim=1)
        v = v.repeat_interleave(n, dim=1)
    return torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)


def check(name, o, r):
    cos = torch.nn.functional.cosine_similarity(o.float().flatten(), r.flatten(), dim=0).item()
    print(f"{name:<48} max {(o.float()-r).abs().max().item():.4f}  cos {cos:.5f}")
    return cos


torch.manual_seed(1)
for dt in (torch.float16, torch.bfloat16):
    for (B, H, KV, L, D) in [(1, 8, 8, 4096, 128), (2, 24, 8, 2048, 128), (1, 12, 12, 2048, 64)]:
        q = torch.randn(B, H, L, D, device="cuda", dtype=dt)
        k = torch.randn(B, KV, L, D, device="cuda", dtype=dt)
        v = torch.randn(B, KV, L, D, device="cuda", dtype=dt)
        o = sageattn(q, k, v)
        check(f"SA2 {dt} H{H}/KV{KV} L{L} D{D}", o, ref(q, k, v))
        del q, k, v, o
        torch.cuda.empty_cache()

# NHD layout
q = torch.randn(1, 4096, 8, 128, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
o = sageattn(q, k, v, tensor_layout="NHD")
check("SA2 NHD L4096 D128", o, ref(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2))
del q, k, v, o
torch.cuda.empty_cache()

# causal
q = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
o = sageattn(q, k, v, is_causal=True)
check("SA2 causal L4096", o, ref(q, k, v, True))
del q, k, v, o
torch.cuda.empty_cache()

# SA3 fp16 / bf16, 64 and 128 head dim
for dt in (torch.float16, torch.bfloat16):
    for D in (64, 128):
        q = torch.randn(1, 12, 2048, D, device="cuda", dtype=dt)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        o = sageattn3_blackwell(q, k, v)
        check(f"SA3 {dt} L2048 D{D}", o, ref(q, k, v))
        del q, k, v, o
        torch.cuda.empty_cache()
print("smoke ok")
