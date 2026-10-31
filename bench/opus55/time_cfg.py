"""attn-kernel-only timing for one build across configs. usage: time_cfg.py VAR"""
import sys
var = sys.argv[1]
WT = "/home/chai/worktrees/sa-opus55/sageattention3_blackwell"
if var != "prod": sys.path.insert(0, f"/tmp/opus55/abl/{var}")
sys.path.insert(1 if var != "prod" else 0, WT)
import torch, fp4attn_cuda, sageattn3.api as api
assert (var == "prod") == fp4attn_cuda.__file__.startswith(WT)
def ct(fn, warmup=3, iters=3, trials=7):
    for _ in range(warmup): fn()
    best = 1e9
    for _ in range(trials):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize(); best = min(best, s.elapsed_time(e) / iters)
    return best
for (dt, D, L, causal) in [(torch.bfloat16, 128, 8192, False), (torch.bfloat16, 128, 16384, False), (torch.bfloat16, 128, 32768, False),
                            (torch.bfloat16, 128, 16384, True), (torch.bfloat16, 64, 16384, False), (torch.float16, 128, 16384, False)]:
    H = 24
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, H, L, D, device="cuda", dtype=dt) for _ in range(3))
    km = k.mean(dim=-2, keepdim=True)
    q_fp4, q_sf, qm = api.scale_and_quant_fp4_block_mean(q)
    ds = torch.matmul(qm, k.transpose(-2, -1))
    kl = api.scale_and_quant_fp4_permute(k, km); vl = api.scale_and_quant_fp4_transpose(v)
    f = lambda: api.blockscaled_fp4_attn((q_fp4, q_sf), kl, vl, ds, L, causal, True, dt == torch.bfloat16)
    t = ct(f)
    fl = 4 * L * L * D * H / (2 if causal else 1)
    print(f"{var:>5} {str(dt)[6:]:>8} D{D:3d} L{L:6d} causal={int(causal)}  attn {t:8.3f} ms  {fl/t/1e9:6.1f} TFLOPS", flush=True)
    del q, k, v, q_fp4, q_sf, qm, ds, kl, vl; torch.cuda.empty_cache()
