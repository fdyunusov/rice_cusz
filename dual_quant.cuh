#pragma once
#include "common.cuh"

// Compress: float -> codes (uint16) + outlier lists
__global__ void kernel_dual_quant_1d(
    const float* __restrict__ in,
    uint16_t* __restrict__ codes,   // radius + q, or 0xFFFF if outlier
    float* __restrict__ outliers_val,
    int* __restrict__ outliers_idx,
    int* __restrict__ outlier_count,
    int n, float ebx2, int radius)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float ebx2_r = 1.0f / ebx2;
  // prequant
  int d0 = (int)lrintf(in[i] * ebx2_r);
  int prev = 0;
  if (i > 0) prev = (int)lrintf(in[i - 1] * ebx2_r); // note: uses neighbor's prequant of original
  // Better: prequant all first into int buffer. See two-pass below.

  int q = d0 - prev;
  if (abs(q) < radius) {
    codes[i] = (uint16_t)(q + radius);
  } else {
    codes[i] = 0xFFFF; // sentinel
    int pos = atomicAdd(outlier_count, 1);
    outliers_val[pos] = in[i];
    outliers_idx[pos] = i;
  }
}
