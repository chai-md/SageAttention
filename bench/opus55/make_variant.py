"""Create an ablated copy of the SA3 attention kernel in /tmp and build it as a
standalone fp4attn_cuda module (bf16, D=128, non-causal, per_block_mean only).
usage: make_variant.py NAME ABL_MASK [STAGES]
ABL bits: 1 QK gemm, 2 softmax, 4 P->FP4 quantize, 8 PV gemm, 16 add_delta_s,
          32 K smem->reg copy, 64 V smem->reg copy, 128 rescale_o"""
import os, re, shutil, subprocess, sys
name, mask = sys.argv[1], int(sys.argv[2])
stages = int(sys.argv[3]) if len(sys.argv) > 3 else 3
stagger = int(sys.argv[4]) if len(sys.argv) > 4 else 0
SRC = "/home/chai/worktrees/sa-opus55/sageattention3_blackwell/sageattn3"
out = f"/tmp/opus55/abl/{name}"
shutil.rmtree(out, ignore_errors=True)
shutil.copytree(SRC, f"{out}/sageattn3", ignore=shutil.ignore_patterns("*.so", "__pycache__", "quantization"))
ml = f"{out}/sageattn3/blackwell/mainloop_tma_ws.h"
s = open(ml).read()
def sub(pat, rep, cnt):
    global s
    n = s.count(pat)
    assert n == cnt, (pat, n)
    s = s.replace(pat, rep)
s = "#ifndef SAGE3_ABL\n#define SAGE3_ABL 0\n#endif\n" + s
sub("cute::gemm(tiled_mma_qk,", "if constexpr (!(SAGE3_ABL & 1)) cute::gemm(tiled_mma_qk,", 4)
sub("softmax_fused.template online_softmax_with_quant", "if constexpr (!(SAGE3_ABL & 2)) softmax_fused.template online_softmax_with_quant", 3)
sub("quantize(_0{}, tSrS_converion_view);", "if constexpr (!(SAGE3_ABL & 4)) quantize(_0{}, tSrS_converion_view);", 3)
sub("quantize(v_block + 1, tSrS_converion_view);", "if constexpr (!(SAGE3_ABL & 4)) quantize(v_block + 1, tSrS_converion_view);", 3)
sub("cute::gemm(tiled_mma_pv,", "if constexpr (!(SAGE3_ABL & 8)) cute::gemm(tiled_mma_pv,", 4)
sub("add_delta_s(tSrS);", "if constexpr (!(SAGE3_ABL & (16|256|512|1024))) add_delta_s(tSrS); else if constexpr (SAGE3_ABL & 256) clear(tSrS); else if constexpr (SAGE3_ABL & 512) clear(tSrS); else if constexpr (SAGE3_ABL & 1024) fill(tSrS, mainloop_params.softmax_scale_log2);", 3)
# bit 512: zero-init the accumulator, then add delta_s after the QK gemm
sub("auto add_delta_s = [&](auto& acc) {", """auto post_delta_s = [&](auto& acc) {
            ElementDS const* ds_stage = &sDS(_0{}, _0{}, smem_pipe_read_k.index());
            auto acc_float4 = recast<float4>(acc);
            int const col0 = (threadIdx.x % 4) * 8;
            cutlass::NumericArrayConverter<float, ElementDS, 8> cvt;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < 4; i++) {
                uint4 raw = *reinterpret_cast<uint4 const*>(ds_stage + col0 + i * 32);
                cutlass::Array<float, 8> f = cvt(reinterpret_cast<cutlass::Array<ElementDS, 8> const&>(raw));
                float4 a;
                a = acc_float4(make_coord(make_coord(_0{}, _0{}), _0{}), _0{}, i); a.x += f[0]; a.y += f[1]; a.z += f[2]; a.w += f[3]; acc_float4(make_coord(make_coord(_0{}, _0{}), _0{}), _0{}, i) = a;
                a = acc_float4(make_coord(make_coord(_0{}, _0{}), _1{}), _0{}, i); a.x += f[0]; a.y += f[1]; a.z += f[2]; a.w += f[3]; acc_float4(make_coord(make_coord(_0{}, _0{}), _1{}), _0{}, i) = a;
                a = acc_float4(make_coord(make_coord(_0{}, _1{}), _0{}), _0{}, i); a.x += f[4]; a.y += f[5]; a.z += f[6]; a.w += f[7]; acc_float4(make_coord(make_coord(_0{}, _1{}), _0{}), _0{}, i) = a;
                a = acc_float4(make_coord(make_coord(_0{}, _1{}), _1{}), _0{}, i); a.x += f[4]; a.y += f[5]; a.z += f[6]; a.w += f[7]; acc_float4(make_coord(make_coord(_0{}, _1{}), _1{}), _0{}, i) = a;
            }
        };
        auto add_delta_s = [&](auto& acc) {""", 1)
# post-add must happen before the K stage is released: insert right before each K release in the 3 loops
sub("pipeline_k.consumer_release(smem_pipe_read_k);", "if constexpr (SAGE3_ABL & 512) post_delta_s(tSrS); pipeline_k.consumer_release(smem_pipe_read_k);", 3)
sub("copy(smem_tiled_copy_K, tSsK_stage", "if constexpr (!(SAGE3_ABL & 32)) copy(smem_tiled_copy_K, tSsK_stage", 1)
sub("copy(smem_tiled_copy_SFK, tSsSFK_stage", "if constexpr (!(SAGE3_ABL & 32)) copy(smem_tiled_copy_SFK, tSsSFK_stage", 1)
sub("copy(smem_tiled_copy_V, tOsVt_stage", "if constexpr (!(SAGE3_ABL & 64)) copy(smem_tiled_copy_V, tOsVt_stage", 1)
sub("copy(smem_tiled_copy_SFV, tOsSFVt_stage", "if constexpr (!(SAGE3_ABL & 64)) copy(smem_tiled_copy_SFV, tOsSFVt_stage", 1)
sub("softmax_fused.rescale_o(tOrO_store, tOrO);", "if constexpr (!(SAGE3_ABL & 128)) softmax_fused.rescale_o(tOrO_store, tOrO);", 2)
import re as _re
# bit 2048: wait for V + copy V block 0 before softmax (latency overlaps softmax)
if mask & 2048:
    sm = "if constexpr (!(SAGE3_ABL & 2)) softmax_fused.template online_softmax_with_quant"
    assert s.count(sm) == 3
    s = s.replace(sm, "if constexpr (SAGE3_ABL & 2048) { consumer_wait(pipeline_v, smem_pipe_read_v); copy_v_block(_0{}); }\n            " + sm)
    pat = _re.compile(r"consumer_wait\(pipeline_v, smem_pipe_read_v\);\n(\s*)copy_v_block\(_0\{\}\);")
    assert len(pat.findall(s)) == 3, len(pat.findall(s))
    s = pat.sub(lambda m: "if constexpr (!(SAGE3_ABL & 2048)) { consumer_wait(pipeline_v, smem_pipe_read_v);\n" + m.group(1) + "copy_v_block(_0{}); }", s)
absinit = os.environ.get("ABSINIT", "")
if absinit:
    anc = "        consumer_wait(pipeline_k, smem_pipe_read_k);\n        copy_k_block(_0{});"
    assert s.count(anc) == 1
    s = s.replace(anc, ("        fill(AbsMaxP, 0.f);\n" if absinit == "zero" else "        fill(AbsMaxP, -INFINITY);\n") + anc)
if os.environ.get("KT2"):
    old = """            Tensor tScS = thread_mma_qk.partition_C(cS);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tSrS); ++i) {
                if constexpr (!Is_causal) {  // Just masking based on col
                    if (k_token(int(get<1>(tScS(i)))) >="""
    new = """            Tensor tScS = thread_mma_qk.partition_C(cS);
            Tensor tScS0 = tiled_mma_qk.get_thread_slice(0).partition_C(cS);  // compile-time coords
            int const tok_thr = k_token(int(get<1>(tScS(0))) - int(get<1>(tScS0(0))));
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tSrS); ++i) {
                if constexpr (!Is_causal) {  // Just masking based on col
                    if (k_token(int(get<1>(tScS0(i)))) + tok_thr >="""
    assert s.count(old) == 1
    s = s.replace(old, new)
open(ml, "w").write(s)
if absinit == "ninf":
    sf = f"{out}/sageattn3/blackwell/softmax_fused.h"
    t = open(sf).read()
    old = "            acc_conversion_flatten(j, i) /= AbsMaxP(i);"
    assert t.count(old) == 1
    # fully masked 16-col group: AbsMaxP = exp2(-inf) = 0 -> make it 1 so P = 0/1 = 0 (no NaN)
    t = t.replace("        for (int i = 0; i < size(AbsMaxP); ++i) {\n            CUTLASS_PRAGMA_UNROLL", "        for (int i = 0; i < size(AbsMaxP); ++i) {\n            if (AbsMaxP(i) == 0.f) AbsMaxP(i) = 1.f;\n            CUTLASS_PRAGMA_UNROLL", 1)
    assert "AbsMaxP(i) = 1.f" in t
    open(sf, "w").write(t)
kw = f"{out}/sageattn3/blackwell/kernel_ws.h"
k = open(kw).read()
anchor = "            collective_mainloop.mma(mainloop_params"
assert k.count(anchor) == 1
k = k.replace(anchor, f"""            if constexpr ({stagger} > 0) {{
                if (warp_group_role == WarpGroupRole::Consumer1) {{
                    long long t0 = clock64();
                    while (clock64() - t0 < {stagger}) {{ }}
                }}
            }}
""" + anchor)
open(kw, "w").write(k)
# restrict instantiations to bf16 / D128 / non-causal / per_block_mean
lh = f"{out}/sageattn3/blackwell/launch.h"
t = open(lh).read()
old = t[t.index("template<typename T, int Headdim, typename O = cutlass::bfloat16_t>"):]
new = f"""template<typename T, int Headdim, typename O = cutlass::bfloat16_t>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream) {{
    TORCH_CHECK(!params.is_causal && params.per_block_mean && Headdim == 128, "ablation build: bf16/D128/non-causal/per_block only");
    if constexpr (Headdim == 128) {{
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, {stages}, 1, true, T, O>, false>(params, stream);
    }}
}}
"""
open(lh, "w").write(t.replace(old, new))
ac = f"{out}/sageattn3/blackwell/api.cu"
a = open(ac).read()
a = a.replace("""    BOOL_SWITCH(params.is_bf16, IsBF16, ([&] {
        run_mha_fwd_dispatch_dtype<IsBF16>(params, stream);
    }));""", """    TORCH_CHECK(params.is_bf16, "ablation build: bf16 only");
    run_mha_fwd_dispatch_dtype<true>(params, stream);""")
open(ac, "w").write(a)
torch_inc = "/home/chai/ComfyUI/.venv/lib/python3.13/site-packages/torch/include"
cut = "/home/chai/worktrees/sa-opus55/sageattention3_blackwell/csrc/cutlass"
py_inc = "/home/chai/.local/share/uv/python/cpython-3.13.14-linux-aarch64-gnu/include/python3.13"
cmd = ["/usr/local/cuda/bin/nvcc", f"-I{out}/sageattn3", f"-I{cut}/include", f"-I{cut}/tools/util/include",
       f"-I{torch_inc}", f"-I{torch_inc}/torch/csrc/api/include", "-I/usr/local/cuda/include", f"-I{py_inc}",
       "-c", ac, "-o", f"{out}/api.o", "--expt-relaxed-constexpr", "--compiler-options", "-fPIC", "-O3", "-std=c++20",
       "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
       "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
       "--expt-extended-lambda", "--use_fast_math", "--ptxas-options=--verbose", "-lineinfo",
       "-DCUTLASS_DEBUG_TRACE_LEVEL=0", "-DNDEBUG", "-DQBLKSIZE=128", "-DKBLKSIZE=128", "-DCTA256", "-DDQINRMEM",
       "-DEXECMODE=0", "-gencode", "arch=compute_121a,code=sm_121a", "--threads", "4",
       "-DTORCH_API_INCLUDE_EXTENSION_H", "-DTORCH_EXTENSION_NAME=fp4attn_cuda",
       f"-DSAGE3_ABL={mask}"]
r = subprocess.run(cmd, capture_output=True, text=True)
open(f"{out}/build.log", "w").write(r.stdout + r.stderr)
if r.returncode:
    print(r.stderr[-3000:]); sys.exit(1)
tl = "/home/chai/ComfyUI/.venv/lib/python3.13/site-packages/torch/lib"
link = ["g++", "-shared", f"{out}/api.o", f"-L{tl}", "-L/usr/local/cuda/lib64", "-lc10", "-ltorch", "-ltorch_cpu",
        "-ltorch_python", "-lcudart", "-lc10_cuda", "-ltorch_cuda", "-lcuda", "-o", f"{out}/fp4attn_cuda.cpython-313-aarch64-linux-gnu.so"]
r = subprocess.run(link, capture_output=True, text=True)
if r.returncode:
    print(r.stderr[-2000:]); sys.exit(1)
regs = re.findall(r"(\d+) bytes spill stores.*?\n.*?Used (\d+) registers", open(f"{out}/build.log").read())
print(name, "built; spill/regs:", regs)
