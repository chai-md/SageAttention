#!/usr/bin/env python3
"""SA3 extra checks (review-claude): fused qm correctness + bf16 delta_s paths + new stage split."""
import torch
import torch.nn.functional as F
import sageattn3.api as api
from sageattn3 import sageattn3_blackwell


def cuda_time(fn, iters=20, warmup=3):
    for _ in range(warmup):
        fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


torch.manual_seed(0)
# 1) fused qm == torch mean, and quantized bits identical to quant(q, qm)
for dt in (torch.bfloat16, torch.float16):
    q = torch.randn(2, 4, 1024, 128, device="cuda", dtype=dt) + 0.5
    qf, sf, qm = api.scale_and_quant_fp4_block_mean(q)
    qm_ref = q.view(2, 4, 8, 128, 128).mean(-2)
    qf2, sf2 = api.scale_and_quant_fp4(q, qm)
    print(f"{dt}: qm maxdiff vs torch {(qm.float()-qm_ref.float()).abs().max().item():.2e}  "
          f"fp4 bits equal={torch.equal(qf, qf2)} sf equal={torch.equal(sf.view(torch.uint8), sf2.view(torch.uint8))}")
# 2) end-to-end cos over dtypes / per_block_mean / ragged L
for dt in (torch.bfloat16, torch.float16):
    for pbm in (True, False):
        for L in (1000, 4096):
            q, k, v = (torch.randn(1, 8, L, 128, device="cuda", dtype=dt) for _ in range(3))
            ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
            qc, kc, vc = q.clone(), k.clone(), v.clone()
            o = sageattn3_blackwell(q, k, v, per_block_mean=pbm)
            nd = torch.equal(q, qc) and torch.equal(k, kc) and torch.equal(v, vc)
            print(f"{str(dt):15s} pbm={pbm!s:5s} L={L:5d} cos={cos(o, ref):.5f} inputs_untouched={nd}")
# 3) new pipeline stage split at 16k
B, H, L, D = 1, 24, 16384, 128
q, k, v = (torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
_, _, qm = api.scale_and_quant_fp4_block_mean(q)
print(f"16k: k.mean {cuda_time(lambda: k.mean(dim=-2, keepdim=True)):.3f} | "
      f"quantQ+qm {cuda_time(lambda: api.scale_and_quant_fp4_block_mean(q)):.3f} | "
      f"quantQ(mean in) {cuda_time(lambda: api.scale_and_quant_fp4(q, qm)):.3f} | "
      f"qm torch {cuda_time(lambda: q.view(B, H, L // 128, 128, D).mean(-2)):.3f} | "
      f"delta_s mm {cuda_time(lambda: torch.matmul(qm, k.transpose(-2, -1))):.3f} | "
      f"total {cuda_time(lambda: sageattn3_blackwell(q, k, v)):.3f} ms")
