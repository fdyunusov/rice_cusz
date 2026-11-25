#pragma once

#include "common.cuh"

/*
 * GPU Golomb-Rice codec for signed integer quant codes.
 *
 * Pipeline:
 *   signed delta -> ZigZag unsigned -> Rice(k) bitstring
 *
 * Rice(k) for unsigned v:
 *   q = v >> k          (unary quotient: q ones then a zero)
 *   r = v & ((1<<k)-1)  (k-bit remainder)
 *   bit length = q + 1 + k
 *
 * Parallel encode:
 *   1) per-symbol bit length
 *   2) exclusive scan -> write offsets
 *   3) pack bits into a uint32 word array
 *
 * Parallel decode (block-wise):
 *   Each CUDA block owns a contiguous range of symbols whose bitstream
 *   slice is described by [bit_offset, bit_length]. Blocks decode sequentially
 *   inside the block (simple + correct). For class-project scale this is enough.
 */

struct RiceArchive {
  uint32_t n = 0;           // number of symbols
  uint32_t rice_k = 0;      // Rice parameter k
  uint32_t nwords = 0;      // number of uint32 words in bitstream
  uint64_t total_bits = 0;  // valid bits in bitstream
  // Followed in the byte buffer by:
  //   uint32_t words[nwords]
};

/** Encode signed deltas on GPU; returns host byte buffer (header + words). */
std::vector<uint8_t> rice_compress_device(const int32_t* d_delta, size_t n,
                                          int rice_k,
                                          cudaStream_t stream = 0);

/** Decode archive into device deltas (allocates/overwrites d_delta[n]). */
void rice_decompress_device(const uint8_t* host_archive, size_t archive_bytes,
                            int32_t* d_delta, size_t n_expected,
                            cudaStream_t stream = 0);

/** Host reference encode/decode for unit tests. */
std::vector<uint8_t> rice_compress_host(const int32_t* delta, size_t n,
                                        int rice_k);
void rice_decompress_host(const uint8_t* archive, size_t archive_bytes,
                          int32_t* delta_out, size_t n_expected);

__host__ __device__ inline uint32_t zigzag_encode(int32_t n) {
  return (uint32_t)((n << 1) ^ (n >> 31));
}

__host__ __device__ inline int32_t zigzag_decode(uint32_t n) {
  return (int32_t)((n >> 1) ^ -(int32_t)(n & 1u));
}

__host__ __device__ inline uint32_t rice_bitlen(uint32_t v, int k) {
  return (v >> k) + 1u + (uint32_t)k;
}
