#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

#define CHECK_CUDA(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
            cudaGetErrorString(err)); \
    exit(1); \
  } \
} while (0)

inline double gb_per_s(size_t bytes, float ms) {
  return (bytes / 1e9) / (ms / 1e3);
}

struct Timer {
  cudaEvent_t a{}, b{};
  Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
  ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
  void start() { cudaEventRecord(a); }
  float stop_ms() {
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); return ms;
  }
};
