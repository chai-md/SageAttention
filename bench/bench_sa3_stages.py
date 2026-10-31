#!/usr/bin/env python3
"""Stage breakdown for SageAttention3 on GB10, following the *real* dispatch.

Mirrors `sageattn3_blackwell()` with per_block_mean=True (the default):
  km = k.mean -> pad -> Q quantizer (also emits qm) -> delta_s = qm @ k^T
  -> K quantizer (subtracts km) -> V quantizer -> fp4 attention kernel
"""
import sys

import torch

sys.path.insert(0, "/home/chai/SageAttention/bench")
from bench_gb10 import cuda_time  # noqa: E402

import sageattn3.api as api  # noqa: E402


def main():
    torch.manual_seed(0)
    dtype = torch.bfloat16
    for B, H, L, D in [(1, 24, 8192, 128), (1, 24, 16384, 128)]:
        q = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, L, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, L, D, device="cuda", dtype=dtype)

        t_km = cuda_time(lambda: k.mean(dim=-2, keepdim=True))
        km = k.mean(dim=-2, keepdim=True)
        t_pad = cuda_time(lambda: api._pad_128(q))
        qp, kp, vp = map(api._pad_128, [q, k, v])
        t_q = cuda_time(lambda: api.scale_and_quant_fp4_block_mean(qp))
        q_fp4, q_sf, qm = api.scale_and_quant_fp4_block_mean(qp)
        t_ds = cuda_time(lambda: torch.matmul(qm, kp.transpose(-2, -1)))
        delta_s = torch.matmul(qm, kp.transpose(-2, -1))
        t_k = cuda_time(lambda: api.scale_and_quant_fp4_permute(kp, km))
        klist = api.scale_and_quant_fp4_permute(kp, km)
        t_v = cuda_time(lambda: api.scale_and_quant_fp4_transpose(vp))
        vlist = api.scale_and_quant_fp4_transpose(vp)
        t_attn = cuda_time(lambda: api.blockscaled_fp4_attn((q_fp4, q_sf), klist, vlist, delta_s, L, False, True, True))
        total = t_km + t_pad + t_q + t_ds + t_k + t_v + t_attn
        print(f"shape {(B,H,L,D)}  total {total:.2f} ms  (attn kernel {t_attn:.2f} ms = {100*t_attn/total:.0f}%)")
        print(f"  k.mean {t_km*1e3:6.1f}us | pad {t_pad*1e3:5.1f}us | quantQ+qm {t_q*1e3:6.1f}us | "
              f"delta_s mm {t_ds*1e3:6.1f}us | quantK {t_k*1e3:6.1f}us | quantV {t_v*1e3:6.1f}us")
        del q, k, v, qp, kp, vp, q_fp4, q_sf, qm, delta_s, klist, vlist
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
