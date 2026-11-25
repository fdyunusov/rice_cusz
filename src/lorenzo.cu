#include "lorenzo.cuh"

#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void kernel_prequant_lorenzo_delta(const float* x, int32_t* delta,
                                              size_t n, double inv_ebx2) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;

  // Dual-quant step 1: prequantization in double for tighter eb adherence
  // d_pre = round(x / (2*eb)) ; inv_ebx2 = 1/(2*eb)
  int32_t d_pre = (int32_t)llrint((double)x[i] * inv_ebx2);

  int32_t pred = 0;
  if (i > 0)
    pred = (int32_t)llrint((double)x[i - 1] * inv_ebx2);
  delta[i] = d_pre - pred;
}

__global__ void kernel_scale_from_prequant(const int32_t* d_pre, float* out,
                                           size_t n, double ebx2) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = (float)((double)d_pre[i] * ebx2);
}

// ---------------------------------------------------------------------------
// Device API
// ---------------------------------------------------------------------------

void lorenzo_quant_device(const float* d_x, int32_t* d_delta, size_t n, float eb,
                          cudaStream_t stream) {
  if (n == 0) return;
  if (!(eb > 0.f)) {
    fprintf(stderr, "lorenzo_quant_device: eb must be > 0\n");
    exit(EXIT_FAILURE);
  }
  double inv_ebx2 = 1.0 / (2.0 * (double)eb);
  const int BS = 256;
  int grid = (int)((n + BS - 1) / BS);
  kernel_prequant_lorenzo_delta<<<grid, BS, 0, stream>>>(d_x, d_delta, n,
                                                         inv_ebx2);
  CUDA_CHECK(cudaGetLastError());
}

void lorenzo_reconstruct_device(const int32_t* d_delta, float* d_out, size_t n,
                                float eb, cudaStream_t stream) {
  if (n == 0) return;
  double ebx2 = 2.0 * (double)eb;

  // Inclusive scan of deltas recovers prequantized integers:
  // d_pre[i] = sum_{j=0..i} delta[j]
  int32_t* d_pre = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pre, n * sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpyAsync(d_pre, d_delta, n * sizeof(int32_t),
                             cudaMemcpyDeviceToDevice, stream));

  thrust::device_ptr<int32_t> ptr(d_pre);
  thrust::inclusive_scan(thrust::cuda::par.on(stream), ptr, ptr + n, ptr);

  const int BS = 256;
  int grid = (int)((n + BS - 1) / BS);
  kernel_scale_from_prequant<<<grid, BS, 0, stream>>>(d_pre, d_out, n, ebx2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFree(d_pre));
}

// ---------------------------------------------------------------------------
// Host reference
// ---------------------------------------------------------------------------

void lorenzo_quant_host(const float* x, int32_t* delta, size_t n, float eb) {
  double inv_ebx2 = 1.0 / (2.0 * (double)eb);
  int32_t prev = 0;
  for (size_t i = 0; i < n; ++i) {
    int32_t d_pre = (int32_t)llrint((double)x[i] * inv_ebx2);
    int32_t pred = (i == 0) ? 0 : prev;
    delta[i] = d_pre - pred;
    prev = d_pre;
  }
}

void lorenzo_reconstruct_host(const int32_t* delta, float* out, size_t n,
                              float eb) {
  double ebx2 = 2.0 * (double)eb;
  int64_t acc = 0;
  for (size_t i = 0; i < n; ++i) {
    acc += delta[i];
    out[i] = (float)((double)acc * ebx2);
  }
}
