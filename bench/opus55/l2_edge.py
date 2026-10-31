"""Edge cases for L2ReduceQuantKernel: bit-equality vs the separate kernels,
lag 0/1, R=1, num_bh=1, tiny L, red_tokens=64, multi-stream, CUDA graph."""
import itertools, torch
from sageattention import _fused
import sageattention as _sa; print("using", _sa.__file__)

def old_k(k, layout):
    sd = 1 if layout == 0 else 2
    km = k.mean(dim=sd)
    b = k.size(0); h = k.size(2) if layout == 0 else k.size(1); L = k.size(sd)
    ki = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    ks = torch.empty((b, h, (L + 63) // 64), dtype=torch.float32, device=k.device)
    _fused.quant_per_block_int8_fuse_sub_mean_cuda(k, km, ki, ks, 64, layout)
    return km, ki, ks

def new_k(k, layout, lag, rt):
    sd = 1 if layout == 0 else 2
    b = k.size(0); h = k.size(2) if layout == 0 else k.size(1); L = k.size(sd); D = k.size(3)
    km = torch.empty((b, h, D), dtype=k.dtype, device=k.device)
    ki = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    ks = torch.empty((b, h, (L + 63) // 64), dtype=torch.float32, device=k.device)
    _fused.mean_quant_per_block_int8_cuda(k, km, ki, ks, layout, lag, rt)
    return km, ki, ks

def vshape(v, layout):
    if layout == 1:
        b, h, L, D = v.shape; return (b, h, D, (L + 63) // 64 * 64), (b, h, D)
    b, L, h, D = v.shape; return (b, D, h, (L + 63) // 64 * 64), (b, h, D)

def old_v(v, layout):
    os_, as_ = vshape(v, layout)
    out = torch.empty(os_, dtype=torch.float8_e4m3fn, device=v.device)
    am = torch.zeros(as_, dtype=torch.float32, device=v.device)
    _fused.channel_amax_cuda(v, am, layout)
    _fused.transpose_pad_permute_quant_cuda(v, out, am, 448.0, layout)
    return am, out

def new_v(v, layout, lag, rt):
    os_, as_ = vshape(v, layout)
    out = torch.empty(os_, dtype=torch.float8_e4m3fn, device=v.device)
    am = torch.empty(as_, dtype=torch.float32, device=v.device)
    _fused.amax_transpose_pad_permute_quant_cuda(v, out, am, 448.0, layout, lag, rt)
    return am, out

def mk(b, h, L, D, dt, layout):
    shp = (b, h, L, D) if layout == 1 else (b, L, h, D)
    return (torch.randn(shp, device="cuda", dtype=dt) * 3 + 0.5)

torch.manual_seed(0)
bad = 0; n = 0; kmd = 0.0
for (b, h, L, D, dt, layout, lag, rt) in itertools.product(
        (1, 2), (1, 3), (1, 63, 64, 65, 511, 1000, 2048, 4097), (64, 128),
        (torch.bfloat16, torch.float16), (0, 1), (0, 1), (64, 512, 2048)):
    x = mk(b, h, L, D, dt, layout)
    km0, ki0, ks0 = old_k(x, layout); km1, ki1, ks1 = new_k(x, layout, lag, rt)
    am0, vo0 = old_v(x, layout); am1, vo1 = new_v(x, layout, lag, rt)
    # km: fp32-sum order differs from torch.mean -> allow 1 ulp, then quant must match
    # given the same km; check the quant bits with the kernel's own km
    okm = torch.equal(km0, km1)
    ok = okm and torch.equal(ki0, ki1) and torch.equal(ks0, ks1) and torch.equal(am0, am1) and \
        torch.equal(vo0.view(torch.uint8), vo1.view(torch.uint8))
    n += 1
    if not ok:
        bad += 1
        kmd = max(kmd, (km0.float() - km1.float()).abs().max().item())
        if bad <= 8:
            print("MISMATCH", b, h, L, D, dt, layout, lag, rt, "km", okm,
                  "ki", torch.equal(ki0, ki1), "ks", torch.equal(ks0, ks1), "am", torch.equal(am0, am1),
                  "v", torch.equal(vo0.view(torch.uint8), vo1.view(torch.uint8)))
print(f"bit-equality: {n - bad}/{n} configs identical (max |km diff| on mismatches {kmd:g})")

# multi-stream: 4 streams launching concurrently, each with its own workspace
x = mk(1, 24, 8192, 128, torch.bfloat16, 1)
ref = new_k(x, 1, 1, 1024)
streams = [torch.cuda.Stream() for _ in range(4)]
res = []
torch.cuda.synchronize()
for it in range(20):
    for s in streams:
        with torch.cuda.stream(s):
            res.append((s, new_k(x, 1, 1, 1024)))
torch.cuda.synchronize()
ok = all(torch.equal(r[0], ref[0]) and torch.equal(r[1], ref[1]) and torch.equal(r[2], ref[2]) for _, r in res)
print("multi-stream x4 x20:", "OK" if ok else "FAIL")

# CUDA graph capture + replay (workspace zeroing must be captured)
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(2): new_k(x, 1, 1, 1024)
torch.cuda.synchronize()
with torch.cuda.graph(g):
    out = new_k(x, 1, 1, 1024)
    outv = new_v(x, 1, 1, 1024)
okg = True
for _ in range(10):
    g.replay(); torch.cuda.synchronize()
    okg &= torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1]) and torch.equal(out[2], ref[2])
am0, vo0 = new_v(x, 1, 1, 1024)
okg &= torch.equal(outv[1].view(torch.uint8), vo0.view(torch.uint8))
print("CUDA graph replay x10:", "OK" if okg else "FAIL")

# determinism: 50 runs, km/amax bitwise identical
x = mk(2, 24, 16384, 128, torch.bfloat16, 1)
r0 = new_k(x, 1, 1, 2048)[0]; a0 = new_v(x, 1, 1, 2048)[0]
det = all(torch.equal(new_k(x, 1, 1, 2048)[0], r0) and torch.equal(new_v(x, 1, 1, 2048)[0], a0) for _ in range(50))
print("determinism x50:", "OK" if det else "FAIL")
