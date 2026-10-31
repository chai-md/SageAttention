// int8 (s32 acc) and fp8 e4m3 (f32 / f16 acc) mma.sync m16n8k32 peaks on GB10
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
template <int KIND>
__global__ void k(float *out, int iters) {
  uint32_t a0 = threadIdx.x * 0x01010101u, a1 = a0 ^ 3, a2 = a0 ^ 5, a3 = a0 ^ 7, b0 = a0 ^ 9, b1 = a0 ^ 11;
  int d[4][4] = {}; float f[4][4] = {}; uint32_t h[4][2] = {};
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      if constexpr (KIND == 0)
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+r"(d[i][0]), "+r"(d[i][1]), "+r"(d[i][2]), "+r"(d[i][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      if constexpr (KIND == 1)
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
            : "+f"(f[i][0]), "+f"(f[i][1]), "+f"(f[i][2]), "+f"(f[i][3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      if constexpr (KIND == 2)
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
            : "+r"(h[i][0]), "+r"(h[i][1]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
  float s = 0;
  for (int i = 0; i < 4; i++) s += d[i][0] + f[i][0] + h[i][0];
  if (s == 1234.5f) out[0] = s;
}
int main() {
  float *o; cudaMalloc(&o, 4); int nsm = 48, iters = 20000;
  const char *nm[] = {"int8 m16n8k32 s32acc", "fp8 m16n8k32 f32acc", "fp8 m16n8k32 f16acc"};
  void (*ks[])(float *, int) = {k<0>, k<1>, k<2>};
  for (int K = 0; K < 3; K++) {
    dim3 g(nsm * 2), b(256);
    ks[K]<<<g, b>>>(o, iters); cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0); for (int r = 0; r < 5; r++) ks[K]<<<g, b>>>(o, iters); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 5;
    double ops = 2.0 * 16 * 8 * 32 * 4.0 * iters * (b.x / 32) * g.x;
    printf("  %-24s : %7.1f TOPS\n", nm[K], ops / ms / 1e9);
  }
}
