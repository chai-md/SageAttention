"""Deadlock check for L2ReduceQuantKernel under minimal co-residency: launch a persistent
SA3 attention grid on SAGE3_NUM_SM=47 SMs (1 CTA/SM, full RF), then the L2 kernel on a
second stream. It can use at most ~1 SM until attention ends, so tickets are handed out
to one CTA at a time; it must still complete and produce the same bits."""
import os, sys, time, torch
sys.path.insert(0, "/home/chai/worktrees/sa-opus55"); sys.path.insert(1, "/home/chai/worktrees/sa-opus55/sageattention3_blackwell")
from sageattention import _fused
import sageattn3.api as api
L = 32768
q, k, v = (torch.randn(1, 24, L, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
km = k.mean(dim=-2, keepdim=True)
qf, qs, qm = api.scale_and_quant_fp4_block_mean(q); ds = torch.matmul(qm, k.transpose(-2, -1))
kl = api.scale_and_quant_fp4_permute(k, km); vl = api.scale_and_quant_fp4_transpose(v)
x = torch.randn(1, 24, 8192, 128, device="cuda", dtype=torch.bfloat16)
def run_k():
    km_ = torch.empty(1, 24, 128, device="cuda", dtype=x.dtype); ki = torch.empty(x.shape, dtype=torch.int8, device="cuda")
    ks = torch.empty(1, 24, 128, device="cuda"); _fused.mean_quant_per_block_int8_cuda(x, km_, ki, ks, 1, 1, 1024); return km_, ki, ks
ref = run_k(); torch.cuda.synchronize()
sa, sb = torch.cuda.Stream(), torch.cuda.Stream()
ea, eb0, eb1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
with torch.cuda.stream(sa):
    api.blockscaled_fp4_attn((qf, qs), kl, vl, ds, L, False, True, True); ea.record()
time.sleep(0.005)
with torch.cuda.stream(sb):
    eb0.record(); out = run_k(); eb1.record()
torch.cuda.synchronize()
ok = all(torch.equal(a, b) for a, b in zip(out, ref))
print(f"SAGE3_NUM_SM={os.environ.get('SAGE3_NUM_SM')} L2 kernel under co-running attention: "
      f"{eb0.elapsed_time(eb1):.2f} ms, attn finished at {eb0.elapsed_time(ea):.2f} ms rel., bits equal={ok}")
