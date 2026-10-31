#!/usr/bin/env python3
"""Dump SA3 outputs for a set of shapes so a rebuild can be compared bit-for-bit."""
import sys

import torch

from sageattn3 import sageattn3_blackwell

torch.manual_seed(1234)
out = {}
cases = []
for dt in (torch.bfloat16, torch.float16):
    for D in (64, 128):
        for L in (4096, 8192):
            cases.append((dt, D, L))
for dt, D, L in cases:
    q = torch.randn(1, 8, L, D, device="cuda", dtype=dt)
    k = torch.randn(1, 8, L, D, device="cuda", dtype=dt)
    v = torch.randn(1, 8, L, D, device="cuda", dtype=dt)
    o = sageattn3_blackwell(q, k, v)
    out[f"{str(dt).split('.')[-1]}_{D}_{L}"] = o.clone()
    del q, k, v, o
    torch.cuda.empty_cache()

path = sys.argv[1]
torch.save(out, path)
print("saved", len(out), "tensors ->", path)
for k_, t in out.items():
    print(f"  {k_}: {tuple(t.shape)} checksum={t.float().sum().item():.6f}")
