import os, re, subprocess, sys
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
name='orig'
torch_inc = "/home/chai/ComfyUI/.venv/lib/python3.13/site-packages/torch/include"
cut = "/home/chai/worktrees/sa-opus55/sageattention3_blackwell/csrc/cutlass"
py_inc = "/home/chai/.local/share/uv/python/cpython-3.13.14-linux-aarch64-gnu/include/python3.13"
cmd = ["/usr/local/cuda/bin/nvcc", f"-I{src}/sageattn3", f"-I{cut}/include", f"-I{cut}/tools/util/include",
       f"-I{torch_inc}", f"-I{torch_inc}/torch/csrc/api/include", "-I/usr/local/cuda/include", f"-I{py_inc}",
       "-c", f"{src}/sageattn3/blackwell/api.cu", "-o", f"{out}/api.o", "--expt-relaxed-constexpr", "--compiler-options", "-fPIC", "-O3", "-std=c++20",
       "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
       "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
       "--expt-extended-lambda", "--use_fast_math", "--ptxas-options=--verbose", "-lineinfo",
       "-DCUTLASS_DEBUG_TRACE_LEVEL=0", "-DNDEBUG", "-DQBLKSIZE=128", "-DKBLKSIZE=128", "-DCTA256", "-DDQINRMEM",
       "-DEXECMODE=0", "-gencode", "arch=compute_121a,code=sm_121a", "--threads", "4",
       "-DTORCH_API_INCLUDE_EXTENSION_H", "-DTORCH_EXTENSION_NAME=fp4attn_cuda",
       ]
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
