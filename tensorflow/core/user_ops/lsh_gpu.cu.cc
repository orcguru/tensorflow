#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

__global__ void LshGpuKernel(const int M, const int lsh_m, const int L, const int* a, const int* b, int* out) {
  int p = blockDim.x * blockIdx.x + threadIdx.x;
  if (p < M) {
    for (int li = 0; li < L; li++) {
      unsigned long h = 0;
      int offset_b = p*lsh_m*L+li*lsh_m;
      for (int i = 0; i < lsh_m; i++) {
        h = h + ((unsigned long)((unsigned int)a[i+1]) * (unsigned long)((unsigned int)b[offset_b+i]));
        h = (h & 4294967295UL) + 5 * (h >> 32);
        if (h >= 4294967291UL) {
          h = h - 4294967291UL;
        }
      }
      if (a[0] != 0) {
        out[p*L+li] = (unsigned int)(h%(unsigned long)a[0]);
      } else {
        out[p*L+li] = (unsigned int)h;
      }
    }
  }
}

void LshGpuKernelLauncher(const int lsh_m, const int M, const int L, const int* a, const int* b, int* out) {
  int blockCnt = (M/1024) + 1;
  LshGpuKernel<<<blockCnt, 1024>>>(M, lsh_m, L, a, b, out);

  cudaError_t cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess) {
    printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
  }
}
