#!/usr/bin/env python3
"""SA2 GB10 round 2: check + time the single-launch L2 reduce/quantize kernels.

1. correctness: new K path (mean fused into K quant) vs k.mean + per_warp_int8,
   new V path vs channel_amax + transpose_pad_permute_quant (must be bit-equal).
2. stage timing A/B at H=24, D=128 for several L.
"""
import sys
import torch
from sageattention import _fused
from sageattention.quant import per_warp_int8, per_warp_int8_smooth_k

torch.manual_seed(0)


def cuda_time(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def v_old(v, layout, scale_max=2.25):
    tl = 0 if layout == "NHD" else 1
    if layout == "HND":
        b, h, L, D = v.shape
        shape = (b, h, D, (L + 63) // 64 * 64)
    else:
        b, L, h, D = v.shape
        shape = (b, D, h, (L + 63) // 64 * 64)
    amax = torch.zeros((b, h, D), dtype=torch.float32, device=v.device)
    _fused.channel_amax_cuda(v, amax, tl)
    out = torch.empty(shape, dtype=torch.float8_e4m3fn, device=v.device)
    _fused.transpose_pad_permute_quant_cuda(v, out, amax, scale_max, tl)
    return out, amax


def v_new(v, layout, scale_max=2.25):
    tl = 0 if layout == "NHD" else 1
    if layout == "HND":
        b, h, L, D = v.shape
        shape = (b, h, D, (L + 63) // 64 * 64)
    else:
        b, L, h, D = v.shape
        shape = (b, D, h, (L + 63) // 64 * 64)
    amax = torch.empty((b, h, D), dtype=torch.float32, device=v.device)
    out = torch.empty(shape, dtype=torch.float8_e4m3fn, device=v.device)
    from sageattention import quant
    _fused.amax_transpose_pad_permute_quant_cuda(v, out, amax, scale_max, tl, quant.L2_LAG, quant._l2_red_tokens(v.shape[2] if layout == "HND" else v.shape[1]))
    return out, amax


def check():
    ok = True
    cases = [
        ("HND", (1, 24, 4096, 128), torch.bfloat16),
        ("HND", (2, 3, 3001, 64), torch.float16),
        ("NHD", (1, 4096, 3, 128), torch.bfloat16),
        ("NHD", (2, 1000, 5, 64), torch.float16),
        ("HND", (1, 2, 100, 128), torch.bfloat16),
        ("HND", (1, 4, 32768, 128), torch.bfloat16),
    ]
    for layout, shape, dt in cases:
        q = torch.randn(shape, device="cuda", dtype=dt)
        k = (torch.randn(shape, device="cuda", dtype=dt) * 2 + 0.5).contiguous()
        v = (torch.randn(shape, device="cuda", dtype=dt) * 3).contiguous()
        seq = 2 if layout == "HND" else 1
        km_ref = k.mean(dim=seq, keepdim=True)
        qi0, qs0, ki0, ks0 = per_warp_int8(q, k, km_ref, BLKQ=128, WARPQ=32, BLKK=64, tensor_layout=layout)
        qi1, qs1, ki1, ks1, km1 = per_warp_int8_smooth_k(q, k, BLKQ=128, WARPQ=32, tensor_layout=layout)
        km_eq = (km1.view(torch.int16) == km_ref.view(torch.int16)).float().mean().item()
        kmis = (ki0 != ki1).sum().item()
        ks_diff = ((ks0 - ks1).abs() / ks0.abs()).max().item()
        q_ok = torch.equal(qi0, qi1) and torch.equal(qs0, qs1)
        vo, ao = v_old(v, layout)
        vn, an = v_new(v, layout)
        v_ok = torch.equal(vo.view(torch.uint8), vn.view(torch.uint8)) and torch.equal(ao, an)
        good = q_ok and v_ok and km_eq > 0.99 and kmis <= 1e-4 * ki0.numel()
        ok &= good
        print(f"{layout} {str(shape):>22} {str(dt):>15}: Q equal {q_ok} | km bit-equal {100*km_eq:6.2f}% | "
              f"k_int8 mismatches {kmis}/{ki0.numel()} | k_scale rel {ks_diff:.1e} | V bit-equal {v_ok}  "
              f"{'OK' if good else 'FAIL'}")
    return ok


def timing():
    print("\nstage timing, bf16 HND H=24 D=128 (ms)")
    print(f"{'L':>6} {'K old (mean+quant)':>19} {'K new':>8} {'V old (2 launches)':>19} {'V new':>8} {'saved':>7}")
    for L in (4096, 8192, 16384, 32768):
        k = torch.randn(1, 24, L, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        q = torch.randn_like(k)

        def k_old():
            km = k.mean(dim=2, keepdim=True)
            per_warp_int8(q, k, km, tensor_layout="HND")

        def k_new():
            per_warp_int8_smooth_k(q, k, tensor_layout="HND")

        tko, tkn = cuda_time(k_old), cuda_time(k_new)
        tvo, tvn = cuda_time(lambda: v_old(v, "HND")), cuda_time(lambda: v_new(v, "HND"))
        print(f"{L:>6} {tko:19.3f} {tkn:8.3f} {tvo:19.3f} {tvn:8.3f} {tko + tvo - tkn - tvn:7.3f}")
        del q, k, v
        torch.cuda.empty_cache()


if __name__ == "__main__":
    good = check()
    if "--no-time" not in sys.argv:
        timing()
    sys.exit(0 if good else 1)
