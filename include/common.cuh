#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <stdexcept>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err__));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

inline double seconds_since(
    const std::chrono::high_resolution_clock::time_point& t0) {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double>(clock::now() - t0).count();
}

struct CompStats {
  size_t n = 0;
  size_t compressed_bytes = 0;
  double compress_s = 0.0;
  double decompress_s = 0.0;
  float max_abs_err = 0.0f;
  float eb = 0.0f;
  int rice_k = 0;

  double ratio() const {
    return n ? (double)(n * sizeof(float)) / (double)compressed_bytes : 0.0;
  }
  double compress_GBps() const {
    return compress_s > 0 ? (n * sizeof(float) / 1e9) / compress_s : 0.0;
  }
  double decompress_GBps() const {
    return decompress_s > 0 ? (n * sizeof(float) / 1e9) / decompress_s : 0.0;
  }
};
