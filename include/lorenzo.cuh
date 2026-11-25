#pragma once

#include "common.cuh"

/*
 * Dual-quantization + 1D Lorenzo prediction (algorithm idea from cuSZ / SZ).
 *
 * Compress:
 *   d_pre[i] = round(x[i] / (2*eb))
 *   pred[i]  = d_pre[i-1]   (0 if i==0)
 *   delta[i] = d_pre[i] - pred[i]
 *
 * Decompress:
 *   d_pre[i] = inclusive_scan(delta)[i]   because d_pre[i] = sum_{j=0..i} delta[j]
 *   x_hat[i] = d_pre[i] * (2*eb)
 *
 * Guarantees |x - x_hat| < eb for finite inputs (prequantization error bound).
 */

/** Device/host: prequant + Lorenzo deltas into signed int32 codes. */
void lorenzo_quant_device(const float* d_x, int32_t* d_delta, size_t n,
                          float eb, cudaStream_t stream = 0);

/** Reconstruct floats from decoded deltas via inclusive scan + scale. */
void lorenzo_reconstruct_device(const int32_t* d_delta, float* d_out, size_t n,
                                float eb, cudaStream_t stream = 0);

/** Host helpers for testing without Rice. */
void lorenzo_quant_host(const float* x, int32_t* delta, size_t n, float eb);
void lorenzo_reconstruct_host(const int32_t* delta, float* out, size_t n,
                              float eb);
