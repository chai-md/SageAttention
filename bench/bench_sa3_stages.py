#!/usr/bin/env python3
"""Stage breakdown for SageAttention3 on GB10 (post mean-fusion)."""
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

        t_pre = cuda_time(lambda: api.preprocess_qkv(q, k, v, True))
        qq, kk, vv, qm, km, ds = api.preprocess_qkv(q, k, v, True)
        t_q = cuda_time(lambda: api.scale_and_quant_fp4(qq, qm))
        ql = api.scale_and_quant_fp4(qq, qm)
        t_k = cuda_time(lambda: api.scale_and_quant_fp4_permute(kk, km))
        kl = api.scale_and_quant_fp4_permute(kk, km)
        t_v = cuda_time(lambda: api.scale_and_quant_fp4_transpose(vv))
        vl = api.scale_and_quant_fp4_transpose(vv)
        t_attn = cuda_time(lambda: api.blockscaled_fp4_attn(ql, kl, vl, ds, L, False, True, True))
        total = t_pre + t_q + t_k + t_v + t_attn
        print(f"shape {(B,H,L,D)}  total {total:.2f} ms  (attn kernel {t_attn:.2f} ms = {100*t_attn/total:.0f}%)")
        print(f"  preprocess {t_pre*1e3:8.1f}us  quantQ {t_q*1e3:7.1f}us  quantK {t_k*1e3:7.1f}us  quantV {t_v*1e3:7.1f}us")
        t_mean = cuda_time(lambda: k.mean(dim=-2, keepdim=True))
        t_ds = cuda_time(lambda: torch.matmul(qm, k.transpose(-2, -1)))
        t_dsf = cuda_time(lambda: ds.to(torch.float32).contiguous())
        t_pad = cuda_time(lambda: q.contiguous())
        print(f"  [parts] k.mean {t_mean*1e3:.1f}us | delta_s matmul {t_ds*1e3:.1f}us | "
              f"delta_s->f32 {t_dsf*1e3:.1f}us | q.contiguous {t_pad*1e3:.1f}us")
        del q, k, v, qq, kk, vv
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
