// GB10 micro-benchmarks: FP4 block-scaled mma.sync peak, and MUFU/cvt rates.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

template <int ILP>
__global__ void mma_fp4(float *out, int iters, uint32_t seed) {
  uint32_t a0 = seed ^ threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7, b0 = a0 * 11, b1 = a0 * 13;
  uint32_t sfa = 0x38383838u, sfb = 0x38383838u;  // ue4m3 1.0
  uint16_t z = 0;
  float d[ILP][4];
#pragma unroll
  for (int i = 0; i < ILP; i++) d[i][0] = d[i][1] = d[i][2] = d[i][3] = 0.f;
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int i = 0; i < ILP; i++) {
      asm volatile(
          "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
          "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
          : "+f"(d[i][0]), "+f"(d[i][1]), "+f"(d[i][2]), "+f"(d[i][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
            "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
    }
  }
  float s = 0;
#pragma unroll
  for (int i = 0; i < ILP; i++) s += d[i][0] + d[i][1] + d[i][2] + d[i][3];
  if (s == 1234.5f) out[0] = s;
}

// op 0: ex2.approx.ftz.f32   op 1: rcp.approx.ftz.f32  op 2: cvt e2m1x2 (2 elems)
// op 3: FFMA                 op 4: cvt e4m3x2
template <int OP>
__global__ void alu(float *out, int iters) {
  float x[8];
  uint32_t acc = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) x[i] = 0.001f * (threadIdx.x + i);
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if constexpr (OP == 0) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(x[i]));
      if constexpr (OP == 1) asm volatile("rcp.approx.ftz.f32 %0, %0;" : "+f"(x[i]));
      if constexpr (OP == 3) asm volatile("fma.rn.f32 %0, %0, %0, %0;" : "+f"(x[i]));
      if constexpr (OP == 2) {
        uint16_t r;
        asm volatile("{.reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %1, %2; mov.b16 %0, {b, 0};}" : "=h"(r) : "f"(x[i]), "f"(x[(i + 1) & 7]));
        acc += r;
      }
      if constexpr (OP == 4) {
        uint16_t r;
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(x[i]), "f"(x[(i + 1) & 7]));
        acc += r;
      }
    }
  }
  float s = acc;
  for (int i = 0; i < 8; i++) s += x[i];
  if (s == 1234.5f) out[0] = s;
}

template <typename K>
float timeit(K k, dim3 g, dim3 b) {
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  k(); cudaDeviceSynchronize();
  cudaEventRecord(e0);
  for (int r = 0; r < 5; r++) k();
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  return ms / 5;
}

int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
  int clk_khz = 0; cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0);
  int nsm = p.multiProcessorCount;
  printf("%s  SMs=%d  reported clock=%.0f MHz\n", p.name, nsm, clk_khz / 1e3);
  float *out; cudaMalloc(&out, 4);
  const int iters = 20000;
  for (int warps : {4, 8, 12, 16}) {
    for (int ctas_per_sm : {1, 2}) {
      auto run = [&](auto kern, int ilp) {
        dim3 g(nsm * ctas_per_sm), b(32 * warps);
        float ms = timeit([&] { kern<<<g, b>>>(out, iters, 1u); }, g, b);
        double flops = 2.0 * 16 * 8 * 64 * (double)ilp * iters * warps * g.x;
        printf("  mma fp4 m16n8k64  warps/CTA=%2d CTAs/SM=%d ILP=%d : %7.1f TFLOPS\n", warps, ctas_per_sm, ilp, flops / ms / 1e9);
      };
      run(mma_fp4<1>, 1); run(mma_fp4<2>, 2); run(mma_fp4<4>, 4);
    }
  }
  const char *names[] = {"ex2.approx", "rcp.approx", "cvt.e2m1x2", "ffma", "cvt.e4m3x2"};
  auto runalu = [&](auto kern, int op) {
    dim3 g(nsm * 4), b(256);
    float ms = timeit([&] { kern<<<g, b>>>(out, iters); }, g, b);
    double ops = 8.0 * iters * g.x * b.x;
    printf("  %-11s : %8.1f Gop/s = %6.1f /clk/SM (at reported clock)\n", names[op], ops / ms / 1e6,
           ops / (ms * 1e-3) / nsm / (clk_khz * 1e3));
  };
  runalu(alu<0>, 0); runalu(alu<1>, 1); runalu(alu<2>, 2); runalu(alu<3>, 3); runalu(alu<4>, 4);
  return 0;
}
