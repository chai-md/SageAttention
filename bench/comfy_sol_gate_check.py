"""Verify the COMFY_SOL_ATTN gate on the exact MiniMax-H3 call signature."""
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, "/home/chai/ComfyUI")
from comfy.ldm.modules.attention import (  # noqa: E402
    SOL_ATTN_FORCED, SOL_ATTN_TAU, AttentionTensorContainer, ComfyAttention,
    attention_sage, optimized_attention,
)

H, D, L = 56, 128, 12555


def containers(t):
    return [AttentionTensorContainer(x.clone()) for x in t]


def timeit(fn, iters=3):
    fn()  # warmup
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(iters)]
    out = None
    for s, e in ev:
        s.record()
        out = fn()
        e.record()
    torch.cuda.synchronize()
    return statistics.median([s.elapsed_time(e) for s, e in ev]), out


def main():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, H, L, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    attn = ComfyAttention()
    attn.load_state_dict({}, strict=False)
    print(f"COMFY_SOL_ATTN={SOL_ATTN_FORCED} tau={SOL_ATTN_TAU} -> method={attn.method} "
          f"function={getattr(attn.function, 'func', attn.function)}")

    t_sage, o_sage = timeit(lambda: attention_sage(*containers((q, k, v)), H, mask=None, skip_reshape=True))
    print(f"  sage           {t_sage:7.2f} ms  out {tuple(o_sage.shape)} {o_sage.dtype}")

    if attn.function is not None:
        t_sol, o_sol = timeit(lambda: optimized_attention(
            *containers((q, k, v)), H, preferred_attention=attn, mask=None,
            skip_reshape=True, transformer_options={}))
        cos = torch.nn.functional.cosine_similarity(
            o_sol.float().flatten(), o_sage.float().flatten(), dim=0).item()
        print(f"  sol (tau={SOL_ATTN_TAU})  {t_sol:7.2f} ms  out {tuple(o_sol.shape)} {o_sol.dtype}"
              f"  speedup {t_sage/t_sol:4.2f}x  cos(vs sage) {cos:.6f}")

    # head_dim 64 must fall back to the dense path instead of raising
    q2, k2, v2 = (torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    o2 = optimized_attention(*containers((q2, k2, v2)), 8, preferred_attention=attn,
                             mask=None, skip_reshape=True, transformer_options={})
    ref = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2)
    print(f"  head_dim 64 fallback: out {tuple(o2.shape)} cos(vs sdpa) "
          f"{torch.nn.functional.cosine_similarity(o2.reshape(-1).float(), ref.transpose(1, 2).reshape(-1).float(), dim=0).item():.6f}")


if __name__ == "__main__":
    main()
