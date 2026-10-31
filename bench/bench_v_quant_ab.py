#!/usr/bin/env python3
"""A/B the V quantization stage: legacy (2 kernels, fp16 intermediate) vs fused."""
import sys

import torch

sys.path.insert(0, "/home/chai/SageAttention/bench")
from bench_gb10 import cuda_time  # noqa: E402
from sageattention import _fused  # noqa: E402
from sageattention.quant import per_channel_fp8  # noqa: E402


def legacy(v, scale_max=2.25):
    b, h, L, D = v.shape
    shape = (b, h, D, (L + 63) // 64 * 64)
    vt = torch.empty(shape, dtype=v.dtype, device=v.device)
    _fused.transpose_pad_permute_cuda(v, vt, 1)
    vf = torch.empty(shape, dtype=torch.float8_e4m3fn, device=v.device)
    vs = torch.empty((b, h, D), dtype=torch.float32, device=v.device)
    _fused.scale_fuse_quant_cuda(vt, vf, vs, L, scale_max, 1)


def main():
    for L in (4096, 8192, 16384, 32768):
        v = torch.randn(1, 24, L, D_ := 128, device="cuda", dtype=torch.bfloat16)
        t_old = cuda_time(lambda: legacy(v), warmup=5, iters=20)
        t_new = cuda_time(lambda: per_channel_fp8(v, tensor_layout="HND", scale_max=2.25, smooth_v=False),
                          warmup=5, iters=20)
        gb = v.numel() * 2 / 2**30
        print(f"L={L:6d}  legacy {t_old*1e3:8.1f}us ({3*gb/(t_old/1e3):6.0f} GB/s eff) | "
              f"fused {t_new*1e3:8.1f}us ({2*gb/(t_new/1e3):6.0f} GB/s eff) | x{t_old/t_new:.2f}")
        del v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
