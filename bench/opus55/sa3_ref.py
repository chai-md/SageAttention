"""save (mode=save) or compare (mode=cmp) SA3 outputs bit-for-bit across builds"""
import sys, torch
mode, path = sys.argv[1], sys.argv[2]
import sageattn3.api as api, fp4attn_cuda
print("using", fp4attn_cuda.__file__)
outs = {}
torch.manual_seed(123)
for dt in (torch.bfloat16, torch.float16):
    for D in (64, 128):
        for L in (1000, 4096, 8192):
            for causal in (False, True):
                for pbm in (True, False):
                    q, k, v = (torch.randn(1, 8, L, D, device="cuda", dtype=dt) for _ in range(3))
                    q0, k0, v0 = q.clone(), k.clone(), v.clone()
                    o = api.sageattn3_blackwell(q, k, v, is_causal=causal, per_block_mean=pbm)
                    assert torch.equal(q, q0) and torch.equal(k, k0) and torch.equal(v, v0), "input modified"
                    r = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)
                    c = torch.nn.functional.cosine_similarity(o.float().flatten(), r.flatten(), dim=0).item()
                    key = f"{str(dt)[6:]}_D{D}_L{L}_c{int(causal)}_pbm{int(pbm)}"
                    outs[key] = (o.cpu(), c)
if mode == "save":
    torch.save(outs, path); print("saved", len(outs))
else:
    ref = torch.load(path); bad = 0
    for kk, (o, c) in outs.items():
        ro, rc = ref[kk]
        eq = torch.equal(o, ro)
        bad += not eq
        print(f"{kk:28s} cos {c:.5f} (ref {rc:.5f}) {'bit-identical' if eq else 'DIFFERENT max|d|=%.3g' % (o.float()-ro.float()).abs().max().item()}")
    print("DIFFERENT:", bad, "of", len(outs))
