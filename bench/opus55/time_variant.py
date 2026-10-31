import sys, os
var = sys.argv[1]; Ls = [int(x) for x in sys.argv[2].split(",")]
WT = "/home/chai/worktrees/sa-opus55/sageattention3_blackwell"
if var != "prod":
    sys.path.insert(0, f"/tmp/opus55/abl/{var}")
sys.path.insert(1 if var != "prod" else 0, WT)
import torch
import fp4attn_cuda
assert (var == "prod") == fp4attn_cuda.__file__.startswith(WT), fp4attn_cuda.__file__
import sageattn3.api as api
def ct(fn, warmup=5, iters=20):
    for _ in range(warmup): fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters
torch.manual_seed(0)
res = []
for L in Ls:
    H, D = 24, 128
    q, k, v = (torch.randn(1, H, L, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    km = k.mean(dim=-2, keepdim=True)
    q_fp4, q_sf, qm = api.scale_and_quant_fp4_block_mean(q)
    ds = torch.matmul(qm, k.transpose(-2, -1))
    kl = api.scale_and_quant_fp4_permute(k, km); vl = api.scale_and_quant_fp4_transpose(v)
    f = lambda: api.blockscaled_fp4_attn((q_fp4, q_sf), kl, vl, ds, L, False, True, True)
    t = ct(f)
    tf = 4 * L * L * D * H / t / 1e9
    extra = ""
    if var in ("prod", "v0", "s2", "p512", "z256", "a16", "vpre"):
        o = f()[0].float(); r = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
        extra = f" cos={torch.nn.functional.cosine_similarity(o.flatten(), r.flatten(), dim=0).item():.5f}"
    res.append(f"{var:>5} L={L:5d} attn {t:7.3f} ms  {tf:6.1f} TFLOPS{extra}")
    del q, k, v, q_fp4, q_sf, qm, ds, kl, vl
print("\n".join(res), flush=True)
