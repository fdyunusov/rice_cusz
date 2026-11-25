#include "rice.cuh"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cstring>
#include <stdexcept>

static constexpr uint32_t kChunkSize = 256;

struct RiceHeader {
  uint32_t n;
  uint32_t rice_k;
  uint32_t nwords;
  uint32_t chunk_size;
  uint32_t num_chunks;
  uint32_t reserved;
  uint64_t total_bits;
};

// ---------------------------------------------------------------------------
// Bit I/O
// ---------------------------------------------------------------------------

__host__ __device__ inline void write_bits_atomic(uint32_t* words,
                                                  uint64_t bit_off,
                                                  uint32_t value, int nbits) {
  while (nbits > 0) {
    uint32_t word = (uint32_t)(bit_off >> 5);
    uint32_t bit = (uint32_t)(bit_off & 31ull);
    int space = 32 - (int)bit;
    int take = nbits < space ? nbits : space;
    uint32_t mask = (take == 32) ? 0xffffffffu : ((1u << take) - 1u);
    uint32_t chunk = value & mask;
#ifdef __CUDA_ARCH__
    atomicOr(&words[word], chunk << bit);
#else
    words[word] |= (chunk << bit);
#endif
    value >>= take;
    nbits -= take;
    bit_off += (uint64_t)take;
  }
}

__host__ __device__ inline uint32_t read_bit(const uint32_t* words,
                                            uint64_t& bit_off) {
  uint32_t word = (uint32_t)(bit_off >> 5);
  uint32_t bit = (uint32_t)(bit_off & 31ull);
  uint32_t b = (words[word] >> bit) & 1u;
  ++bit_off;
  return b;
}

__host__ __device__ inline uint32_t read_bits(const uint32_t* words,
                                             uint64_t& bit_off, int nbits) {
  uint32_t v = 0;
  for (int i = 0; i < nbits; ++i)
    v |= read_bit(words, bit_off) << i;
  return v;
}

__host__ __device__ inline void rice_write_symbol(uint32_t* words,
                                                  uint64_t bit_off, uint32_t v,
                                                  int k) {
  uint32_t q = v >> k;
  uint32_t r = (k >= 32) ? v : (v & ((1u << k) - 1u));

  while (q > 0) {
    int take = q > 32u ? 32 : (int)q;
    uint32_t ones = (take == 32) ? 0xffffffffu : ((1u << take) - 1u);
    write_bits_atomic(words, bit_off, ones, take);
    bit_off += (uint64_t)take;
    q -= (uint32_t)take;
  }
  write_bits_atomic(words, bit_off, 0u, 1);
  bit_off += 1;
  if (k > 0) write_bits_atomic(words, bit_off, r, k);
}

__host__ __device__ inline uint32_t rice_read_symbol(const uint32_t* words,
                                                     uint64_t& bit_off, int k) {
  uint32_t q = 0;
  while (read_bit(words, bit_off) == 1u) ++q;
  uint32_t r = (k > 0) ? read_bits(words, bit_off, k) : 0u;
  return (q << k) | r;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void kernel_bitlens(const int32_t* delta, uint64_t* bitlen, size_t n,
                               int k) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  bitlen[i] = (uint64_t)rice_bitlen(zigzag_encode(delta[i]), k);
}

__global__ void kernel_pack(const int32_t* delta, const uint64_t* offsets,
                            uint32_t* words, size_t n, int k) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  rice_write_symbol(words, offsets[i], zigzag_encode(delta[i]), k);
}

__global__ void kernel_chunk_offsets(const uint64_t* sym_offsets, size_t n,
                                     uint32_t chunk_size, uint64_t* chunk_offs,
                                     uint32_t num_chunks, uint64_t total_bits) {
  uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c > num_chunks) return;
  if (c == num_chunks) {
    chunk_offs[c] = total_bits;
    return;
  }
  size_t idx = (size_t)c * (size_t)chunk_size;
  chunk_offs[c] = (idx < n) ? sym_offsets[idx] : total_bits;
}

__global__ void kernel_decode_chunks(const uint32_t* words,
                                     const uint64_t* chunk_offs, int32_t* delta,
                                     size_t n, int k, uint32_t chunk_size,
                                     uint32_t num_chunks) {
  uint32_t c = blockIdx.x;
  if (c >= num_chunks || threadIdx.x != 0) return;

  size_t begin = (size_t)c * (size_t)chunk_size;
  if (begin >= n) return;
  size_t end = begin + chunk_size;
  if (end > n) end = n;

  uint64_t bit_off = chunk_offs[c];
  for (size_t i = begin; i < end; ++i) {
    uint32_t v = rice_read_symbol(words, bit_off, k);
    delta[i] = zigzag_decode(v);
  }
}

// ---------------------------------------------------------------------------
// Device API
// ---------------------------------------------------------------------------

std::vector<uint8_t> rice_compress_device(const int32_t* d_delta, size_t n,
                                          int rice_k, cudaStream_t stream) {
  if (n == 0) return {};
  if (rice_k < 0 || rice_k > 16) {
    fprintf(stderr, "rice_k must be in [0,16]\n");
    exit(EXIT_FAILURE);
  }

  uint64_t* d_bitlen = nullptr;
  uint64_t* d_offsets = nullptr;
  CUDA_CHECK(cudaMalloc(&d_bitlen, n * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_offsets, n * sizeof(uint64_t)));

  const int BS = 256;
  int grid = (int)((n + BS - 1) / BS);
  kernel_bitlens<<<grid, BS, 0, stream>>>(d_delta, d_bitlen, n, rice_k);
  CUDA_CHECK(cudaGetLastError());

  thrust::device_ptr<uint64_t> bl(d_bitlen);
  thrust::device_ptr<uint64_t> off(d_offsets);
  thrust::exclusive_scan(thrust::cuda::par.on(stream), bl, bl + n, off);

  uint64_t last_off = 0, last_len = 0;
  CUDA_CHECK(cudaMemcpyAsync(&last_off, d_offsets + (n - 1), sizeof(uint64_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(&last_len, d_bitlen + (n - 1), sizeof(uint64_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  uint64_t total_bits = last_off + last_len;

  uint32_t nwords = (uint32_t)((total_bits + 31ull) / 32ull);
  if (nwords == 0) nwords = 1;

  uint32_t* d_words = nullptr;
  CUDA_CHECK(cudaMalloc(&d_words, (size_t)nwords * sizeof(uint32_t)));
  CUDA_CHECK(
      cudaMemsetAsync(d_words, 0, (size_t)nwords * sizeof(uint32_t), stream));

  kernel_pack<<<grid, BS, 0, stream>>>(d_delta, d_offsets, d_words, n, rice_k);
  CUDA_CHECK(cudaGetLastError());

  uint32_t num_chunks = (uint32_t)((n + kChunkSize - 1) / kChunkSize);
  uint64_t* d_chunk_offs = nullptr;
  CUDA_CHECK(cudaMalloc(&d_chunk_offs, (size_t)(num_chunks + 1) * sizeof(uint64_t)));
  int cg = (int)((num_chunks + 1 + 255) / 256);
  kernel_chunk_offsets<<<cg, 256, 0, stream>>>(d_offsets, n, kChunkSize,
                                               d_chunk_offs, num_chunks,
                                               total_bits);
  CUDA_CHECK(cudaGetLastError());

  RiceHeader hdr{};
  hdr.n = (uint32_t)n;
  hdr.rice_k = (uint32_t)rice_k;
  hdr.nwords = nwords;
  hdr.chunk_size = kChunkSize;
  hdr.num_chunks = num_chunks;
  hdr.reserved = 0;
  hdr.total_bits = total_bits;

  size_t bytes = sizeof(RiceHeader) +
                 (size_t)(num_chunks + 1) * sizeof(uint64_t) +
                 (size_t)nwords * sizeof(uint32_t);
  std::vector<uint8_t> out(bytes);
  uint8_t* ptr = out.data();
  std::memcpy(ptr, &hdr, sizeof(hdr));
  ptr += sizeof(hdr);

  CUDA_CHECK(cudaMemcpyAsync(ptr, d_chunk_offs,
                             (size_t)(num_chunks + 1) * sizeof(uint64_t),
                             cudaMemcpyDeviceToHost, stream));
  ptr += (size_t)(num_chunks + 1) * sizeof(uint64_t);
  CUDA_CHECK(cudaMemcpyAsync(ptr, d_words, (size_t)nwords * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(d_bitlen));
  CUDA_CHECK(cudaFree(d_offsets));
  CUDA_CHECK(cudaFree(d_words));
  CUDA_CHECK(cudaFree(d_chunk_offs));
  return out;
}

void rice_decompress_device(const uint8_t* host_archive, size_t archive_bytes,
                            int32_t* d_delta, size_t n_expected,
                            cudaStream_t stream) {
  if (archive_bytes < sizeof(RiceHeader)) {
    fprintf(stderr, "rice archive too small\n");
    exit(EXIT_FAILURE);
  }
  RiceHeader hdr{};
  std::memcpy(&hdr, host_archive, sizeof(hdr));
  if ((size_t)hdr.n != n_expected) {
    fprintf(stderr, "rice archive n mismatch: %u vs %zu\n", hdr.n, n_expected);
    exit(EXIT_FAILURE);
  }

  const uint8_t* ptr = host_archive + sizeof(RiceHeader);
  size_t chunk_bytes = (size_t)(hdr.num_chunks + 1) * sizeof(uint64_t);
  size_t word_bytes = (size_t)hdr.nwords * sizeof(uint32_t);
  if (sizeof(RiceHeader) + chunk_bytes + word_bytes > archive_bytes) {
    fprintf(stderr, "rice archive truncated\n");
    exit(EXIT_FAILURE);
  }

  uint64_t* d_chunk_offs = nullptr;
  uint32_t* d_words = nullptr;
  CUDA_CHECK(cudaMalloc(&d_chunk_offs, chunk_bytes));
  CUDA_CHECK(cudaMalloc(&d_words, word_bytes));
  CUDA_CHECK(cudaMemcpyAsync(d_chunk_offs, ptr, chunk_bytes,
                             cudaMemcpyHostToDevice, stream));
  ptr += chunk_bytes;
  CUDA_CHECK(cudaMemcpyAsync(d_words, ptr, word_bytes, cudaMemcpyHostToDevice,
                             stream));

  kernel_decode_chunks<<<hdr.num_chunks, 32, 0, stream>>>(
      d_words, d_chunk_offs, d_delta, hdr.n, (int)hdr.rice_k, hdr.chunk_size,
      hdr.num_chunks);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(d_chunk_offs));
  CUDA_CHECK(cudaFree(d_words));
}

// ---------------------------------------------------------------------------
// Host reference
// ---------------------------------------------------------------------------

std::vector<uint8_t> rice_compress_host(const int32_t* delta, size_t n,
                                        int rice_k) {
  std::vector<uint64_t> offsets(n);
  uint64_t total = 0;
  for (size_t i = 0; i < n; ++i) {
    offsets[i] = total;
    total += rice_bitlen(zigzag_encode(delta[i]), rice_k);
  }
  uint32_t nwords = (uint32_t)((total + 31ull) / 32ull);
  if (nwords == 0) nwords = 1;
  std::vector<uint32_t> words(nwords, 0);

  for (size_t i = 0; i < n; ++i)
    rice_write_symbol(words.data(), offsets[i], zigzag_encode(delta[i]),
                      rice_k);

  uint32_t num_chunks = (uint32_t)((n + kChunkSize - 1) / kChunkSize);
  std::vector<uint64_t> chunk_offs(num_chunks + 1);
  for (uint32_t c = 0; c < num_chunks; ++c) {
    size_t idx = (size_t)c * kChunkSize;
    chunk_offs[c] = (idx < n) ? offsets[idx] : total;
  }
  chunk_offs[num_chunks] = total;

  RiceHeader hdr{};
  hdr.n = (uint32_t)n;
  hdr.rice_k = (uint32_t)rice_k;
  hdr.nwords = nwords;
  hdr.chunk_size = kChunkSize;
  hdr.num_chunks = num_chunks;
  hdr.total_bits = total;

  std::vector<uint8_t> out(sizeof(hdr) + chunk_offs.size() * sizeof(uint64_t) +
                           words.size() * sizeof(uint32_t));
  uint8_t* p = out.data();
  std::memcpy(p, &hdr, sizeof(hdr));
  p += sizeof(hdr);
  std::memcpy(p, chunk_offs.data(), chunk_offs.size() * sizeof(uint64_t));
  p += chunk_offs.size() * sizeof(uint64_t);
  std::memcpy(p, words.data(), words.size() * sizeof(uint32_t));
  return out;
}

void rice_decompress_host(const uint8_t* archive, size_t /*archive_bytes*/,
                          int32_t* delta_out, size_t n_expected) {
  RiceHeader hdr{};
  std::memcpy(&hdr, archive, sizeof(hdr));
  if ((size_t)hdr.n != n_expected)
    throw std::runtime_error("host rice n mismatch");

  const uint8_t* p = archive + sizeof(RiceHeader);
  p += (size_t)(hdr.num_chunks + 1) * sizeof(uint64_t);
  const uint32_t* words = reinterpret_cast<const uint32_t*>(p);

  uint64_t bit_off = 0;
  for (uint32_t i = 0; i < hdr.n; ++i) {
    uint32_t v = rice_read_symbol(words, bit_off, (int)hdr.rice_k);
    delta_out[i] = zigzag_decode(v);
  }
}
