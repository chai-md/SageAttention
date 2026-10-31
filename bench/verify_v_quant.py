#!/usr/bin/env python3
"""Verify the fused V-quantize path is numerically identical to the 2-kernel path."""
import torch
from sageattention import _fused
from sageattention.quant import per_channel_fp8

torch.manual_seed(0)


def legacy(v, scale_max=2.25, layout="HND"):
    """Old path: transpose_pad_permute (fp16) -> scale_fuse_quant (fp8)."""
    tl = 0 if layout == "NHD" else 1
    if layout == "HND":
        b, h, L, D = v.shape
        shape = (b, h, D, (L + 63) // 64 * 64)
    else:
        b, L, h, D = v.shape
        shape = (b, D, h, (L + 63) // 64 * 64)
    vt = torch.empty(shape, dtype=v.dtype, device=v.device)
    _fused.transpose_pad_permute_cuda(v, vt, tl)
    vf = torch.empty(shape, dtype=torch.float8_e4m3fn, device=v.device)
    vs = torch.empty((b, h, D), dtype=torch.float32, device=v.device)
    _fused.scale_fuse_quant_cuda(vt, vf, vs, L, scale_max, tl)
    return vf, vs


for layout, v in [
    ("HND", torch.randn(1, 3, 4096, 128, device="cuda", dtype=torch.bfloat16)),
    ("HND", torch.randn(2, 2, 3001, 64, device="cuda", dtype=torch.float16)),
    ("NHD", torch.randn(1, 4096, 3, 128, device="cuda", dtype=torch.bfloat16)),
]:
    v = (v * 3.0).contiguous()  # widen range so scales are meaningful
    new_f, new_s, _ = per_channel_fp8(v, tensor_layout=layout, scale_max=2.25, smooth_v=False)
    old_f, old_s = legacy(v, 2.25, layout)
    sdiff = (new_s - old_s).abs().max().item()
    fdiff = (new_f.float() - old_f.float()).abs().max().item()
    neq = (new_f.view(torch.uint8) != old_f.view(torch.uint8)).sum().item()
    total = new_f.numel()
    print(f"{layout} {tuple(v.shape)}: scale maxdiff {sdiff:.3e} | fp8 maxdiff {fdiff:.3e} | "
          f"byte-mismatches {neq}/{total} ({100.0*neq/total:.4f}%)")
