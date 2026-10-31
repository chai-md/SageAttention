#!/usr/bin/env python3
"""Roofline probe for GB10: measure achievable tensor-core and DRAM rates."""
import torch

def t(fn, iters=20, warmup=5):
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


print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).multi_processor_count, "SMs")

# ---- DRAM bandwidth: large copy
for mb in (256, 1024, 2048):
    x = torch.empty(mb * 1024 * 1024 // 2, dtype=torch.float16, device="cuda")
    y = torch.empty_like(x)
    ms = t(lambda: y.copy_(x))
    print(f"copy {mb}MB: {ms:.3f} ms -> {2*mb/1024/(ms/1000):.0f} GB/s")

# ---- fp16/bf16 matmul (tf32 off)
M = N = 8192; K = 8192
a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
ms = t(lambda: a @ b)
print(f"bf16 matmul {M}x{K}x{N}: {ms:.3f} ms -> {2*M*N*K/ms/1e9:.1f} TFLOPS")

af = a.float(); bf = b.float()
torch.backends.cuda.matmul.allow_tf32 = True
ms = t(lambda: af @ bf)
print(f"tf32 matmul: {ms:.3f} ms -> {2*M*N*K/ms/1e9:.1f} TFLOPS")

# ---- int8 matmul (_int_mm)
ai = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8)
bi = torch.randint(-127, 127, (K, N), device="cuda", dtype=torch.int8)
try:
    ms = t(lambda: torch._int_mm(ai, bi))
    print(f"int8 _int_mm: {ms:.3f} ms -> {2*M*N*K/ms/1e9:.1f} TOPS")
except Exception as ex:
    print("int8 _int_mm failed:", type(ex).__name__, str(ex)[:100])

# ---- fp8 matmul
try:
    a8 = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    b8 = torch.randn(K, N, device="cuda").to(torch.float8_e4m3fn).t().contiguous().t()
    ms = t(lambda: torch._scaled_mm(a8, b8, out_dtype=torch.bfloat16))
    print(f"fp8 _scaled_mm: {ms:.3f} ms -> {2*M*N*K/ms/1e9:.1f} TFLOPS")
except Exception as ex:
    print("fp8 _scaled_mm failed:", type(ex).__name__, str(ex)[:150])

# ---- clock
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit", "--format=csv"], capture_output=True, text=True).stdout)
