#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

__global__ void LshGpuKernel(const int P, const int N, const int* a, const int* b, int* out) {
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p < P) {
      unsigned long h = 0;
      for (int i = 0; i < N; i++) {
        h = h + ((unsigned long)((unsigned int)a[i+1]) * (unsigned long)((unsigned int)b[p*N+i]));
        h = (h & 4294967295UL) + 5 * (h >> 32);
        if (h >= 4294967291UL) {
          h = h - 4294967291UL;
        }
      }
      if (a[0] != 0) {
        out[p] = (unsigned int)(h%(unsigned long)a[0]);
      } else {
        out[p] = (unsigned int)h;
      }
    }
}

void LshGpuKernelLauncher(const int P, const int N, const int* a, const int* b, int* out) {
    int blockCnt = (P/1024) + 1;
    LshGpuKernel<<<blockCnt, 1024>>>(P, N, a, b, out);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }
}
