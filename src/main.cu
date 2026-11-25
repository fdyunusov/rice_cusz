#include "common.cuh"
#include "lorenzo.cuh"
#include "rice.cuh"

#include <fstream>
#include <iostream>
#include <random>
#include <cstring>

static void usage(const char* argv0) {
  printf("Usage: %s [options]\n", argv0);
  printf("  --n N           number of floats (default 1048576)\n");
  printf("  --eb EB         absolute error bound (default 1e-3)\n");
  printf("  --rice-k K      Rice parameter k (default 2)\n");
  printf("  --file PATH     load raw float32 binary (overrides --n size)\n");
  printf("  --pattern NAME  synthetic: smooth | noisy | ramp (default smooth)\n");
  printf("  --self-test     run host Rice + Lorenzo checks, then GPU path\n");
  printf("  --help          show this help\n");
}

static std::vector<float> make_synthetic(size_t n, const std::string& pattern) {
  std::vector<float> x(n);
  std::mt19937 rng(42);
  std::normal_distribution<float> noise(0.f, 1.f);

  if (pattern == "ramp") {
    for (size_t i = 0; i < n; ++i) x[i] = 0.001f * (float)i;
  } else if (pattern == "noisy") {
    for (size_t i = 0; i < n; ++i)
      x[i] = noise(rng);
  } else {  // smooth
    for (size_t i = 0; i < n; ++i) {
      float t = (float)i / 1000.f;
      x[i] = sinf(t) + 0.1f * sinf(4.f * t) + 0.01f * noise(rng);
    }
  }
  return x;
}

static std::vector<float> load_bin(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open file: " + path);
  std::streamsize bytes = in.tellg();
  if (bytes < 0 || (bytes % (std::streamsize)sizeof(float)) != 0)
    throw std::runtime_error("file size is not a multiple of float");
  size_t n = (size_t)bytes / sizeof(float);
  in.seekg(0);
  std::vector<float> x(n);
  in.read(reinterpret_cast<char*>(x.data()), bytes);
  return x;
}

static float max_abs_err(const float* a, const float* b, size_t n) {
  float m = 0.f;
  for (size_t i = 0; i < n; ++i) {
    float e = fabsf(a[i] - b[i]);
    if (e > m) m = e;
  }
  return m;
}

static void self_test() {
  printf("=== Self-test (host) ===\n");
  const size_t n = 4096;
  auto x = make_synthetic(n, "smooth");
  const float eb = 1e-3f;
  const int k = 2;

  std::vector<int32_t> delta(n), delta2(n);
  std::vector<float> xhat(n);

  lorenzo_quant_host(x.data(), delta.data(), n, eb);
  auto arch = rice_compress_host(delta.data(), n, k);
  rice_decompress_host(arch.data(), arch.size(), delta2.data(), n);
  for (size_t i = 0; i < n; ++i) {
    if (delta[i] != delta2[i]) {
      fprintf(stderr, "Rice host mismatch at %zu: %d vs %d\n", i, delta[i],
              delta2[i]);
      exit(EXIT_FAILURE);
    }
  }
  lorenzo_reconstruct_host(delta2.data(), xhat.data(), n, eb);
  float err = max_abs_err(x.data(), xhat.data(), n);
  printf("Host Rice round-trip OK. archive=%zu bytes  max_err=%.6g  eb=%.6g\n",
         arch.size(), err, eb);
  if (err > eb + 1e-6f) {
    fprintf(stderr, "ERROR: host max error exceeds eb\n");
    exit(EXIT_FAILURE);
  }
}

static CompStats run_gpu(const std::vector<float>& x, float eb, int rice_k) {
  const size_t n = x.size();
  CompStats st;
  st.n = n;
  st.eb = eb;
  st.rice_k = rice_k;

  float* d_x = nullptr;
  float* d_out = nullptr;
  int32_t* d_delta = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_delta, n * sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Warmup
  lorenzo_quant_device(d_x, d_delta, n, eb);
  auto warm = rice_compress_device(d_delta, n, rice_k);
  rice_decompress_device(warm.data(), warm.size(), d_delta, n);
  lorenzo_reconstruct_device(d_delta, d_out, n, eb);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed compress
  auto t0 = std::chrono::high_resolution_clock::now();
  lorenzo_quant_device(d_x, d_delta, n, eb);
  auto archive = rice_compress_device(d_delta, n, rice_k);
  CUDA_CHECK(cudaDeviceSynchronize());
  st.compress_s = seconds_since(t0);
  st.compressed_bytes = archive.size();

  // Timed decompress
  auto t1 = std::chrono::high_resolution_clock::now();
  rice_decompress_device(archive.data(), archive.size(), d_delta, n);
  lorenzo_reconstruct_device(d_delta, d_out, n, eb);
  CUDA_CHECK(cudaDeviceSynchronize());
  st.decompress_s = seconds_since(t1);

  std::vector<float> xhat(n);
  CUDA_CHECK(cudaMemcpy(xhat.data(), d_out, n * sizeof(float),
                        cudaMemcpyDeviceToHost));
  st.max_abs_err = max_abs_err(x.data(), xhat.data(), n);

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_delta));
  return st;
}

int main(int argc, char** argv) {
  size_t n = 1 << 20;
  float eb = 1e-3f;
  int rice_k = 2;
  std::string file;
  std::string pattern = "smooth";
  bool do_self = false;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--help")) {
      usage(argv[0]);
      return 0;
    } else if (!strcmp(argv[i], "--n") && i + 1 < argc) {
      n = (size_t)strtoull(argv[++i], nullptr, 10);
    } else if (!strcmp(argv[i], "--eb") && i + 1 < argc) {
      eb = strtof(argv[++i], nullptr);
    } else if (!strcmp(argv[i], "--rice-k") && i + 1 < argc) {
      rice_k = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--file") && i + 1 < argc) {
      file = argv[++i];
    } else if (!strcmp(argv[i], "--pattern") && i + 1 < argc) {
      pattern = argv[++i];
    } else if (!strcmp(argv[i], "--self-test")) {
      do_self = true;
    } else {
      fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      usage(argv[0]);
      return 1;
    }
  }

  if (do_self) self_test();

  std::vector<float> x;
  if (!file.empty()) {
    x = load_bin(file);
    printf("Loaded %zu floats from %s\n", x.size(), file.c_str());
  } else {
    x = make_synthetic(n, pattern);
    printf("Synthetic pattern='%s'  n=%zu\n", pattern.c_str(), n);
  }

  printf("eb=%.6g  rice_k=%d\n", eb, rice_k);

  CompStats st = run_gpu(x, eb, rice_k);

  printf("\n=== Results ===\n");
  printf("original_bytes     : %zu\n", st.n * sizeof(float));
  printf("compressed_bytes   : %zu\n", st.compressed_bytes);
  printf("compression_ratio  : %.3fx\n", st.ratio());
  printf("compress_time_s    : %.6f\n", st.compress_s);
  printf("decompress_time_s  : %.6f\n", st.decompress_s);
  printf("compress_GB/s      : %.3f\n", st.compress_GBps());
  printf("decompress_GB/s    : %.3f\n", st.decompress_GBps());
  printf("max_abs_error      : %.6g\n", st.max_abs_err);
  printf("error_bound        : %.6g\n", st.eb);

  // Float32 round-trip can slightly exceed the ideal real-arithmetic bound
  // when |x| is large (ULP grows) or 2*eb is not binary-exact.
  const float tol = st.eb * 1.01f + 1e-5f;
  if (st.max_abs_err > tol) {
    fprintf(stderr, "FAIL: max abs error exceeds error bound (tol=%.6g)\n",
            tol);
    return 2;
  }
  printf("PASS: max abs error within eb tolerance (tol=%.6g)\n", tol);
  return 0;
}
