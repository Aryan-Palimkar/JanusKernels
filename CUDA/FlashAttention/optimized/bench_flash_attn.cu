#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <limits>
#include <string>

#include "attention.cu"

namespace {

inline void checkCuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << msg << " -> " << cudaGetErrorString(err) << std::endl;
    std::exit(1);
  }
}

template<int BLOCK_Q, int BLOCK_KV, int NUM_WARPS>
float run_bench_cfg(int bs, int num_heads, int len_q, int len_kv, int iters, int warmup) {
  constexpr int DIM = 64;

  if (len_q % BLOCK_Q != 0 || len_kv % BLOCK_KV != 0) {
    return -1.0f;
  }

  const size_t q_elems = static_cast<size_t>(bs) * num_heads * len_q * DIM;
  const size_t kv_elems = static_cast<size_t>(bs) * num_heads * len_kv * DIM;
  const size_t o_elems = q_elems;

  std::vector<half> hQ(q_elems);
  std::vector<half> hK(kv_elems);
  std::vector<half> hV(kv_elems);

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t i = 0; i < q_elems; ++i) hQ[i] = __float2half(dist(rng));
  for (size_t i = 0; i < kv_elems; ++i) hK[i] = __float2half(dist(rng));
  for (size_t i = 0; i < kv_elems; ++i) hV[i] = __float2half(dist(rng));

  half *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
  checkCuda(cudaMalloc(&dQ, q_elems * sizeof(half)), "cudaMalloc dQ");
  checkCuda(cudaMalloc(&dK, kv_elems * sizeof(half)), "cudaMalloc dK");
  checkCuda(cudaMalloc(&dV, kv_elems * sizeof(half)), "cudaMalloc dV");
  checkCuda(cudaMalloc(&dO, o_elems * sizeof(half)), "cudaMalloc dO");

  checkCuda(cudaMemcpy(dQ, hQ.data(), q_elems * sizeof(half), cudaMemcpyHostToDevice), "copy Q");
  checkCuda(cudaMemcpy(dK, hK.data(), kv_elems * sizeof(half), cudaMemcpyHostToDevice), "copy K");
  checkCuda(cudaMemcpy(dV, hV.data(), kv_elems * sizeof(half), cudaMemcpyHostToDevice), "copy V");

  dim3 block(NUM_WARPS * WARP_SIZE);
  dim3 grid(cdiv(len_q, BLOCK_Q), num_heads, bs);
  const size_t smem_bytes = static_cast<size_t>(BLOCK_Q + 4 * BLOCK_KV) * DIM * sizeof(half);

  checkCuda(cudaFuncSetAttribute(
      flash_attn_forward<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_bytes), "set max dynamic shared memory size");
    checkCuda(cudaFuncSetAttribute(
      flash_attn_forward<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS>,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100), "set shared memory carveout");

  for (int i = 0; i < warmup; ++i) {
    flash_attn_forward<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS><<<grid, block, smem_bytes>>>(
        dQ, dK, dV, dO,
        static_cast<unsigned int>(bs),
        static_cast<unsigned int>(num_heads),
        static_cast<unsigned int>(len_q),
        static_cast<unsigned int>(len_kv));
  }
  checkCuda(cudaGetLastError(), "kernel launch warmup");
  checkCuda(cudaDeviceSynchronize(), "sync after warmup");

  cudaEvent_t start{}, stop{};
  checkCuda(cudaEventCreate(&start), "event create start");
  checkCuda(cudaEventCreate(&stop), "event create stop");

  checkCuda(cudaEventRecord(start), "event record start");
  for (int i = 0; i < iters; ++i) {
    flash_attn_forward<BLOCK_Q, BLOCK_KV, DIM, NUM_WARPS><<<grid, block, smem_bytes>>>(
        dQ, dK, dV, dO,
        static_cast<unsigned int>(bs),
        static_cast<unsigned int>(num_heads),
        static_cast<unsigned int>(len_q),
        static_cast<unsigned int>(len_kv));
  }
  checkCuda(cudaEventRecord(stop), "event record stop");
  checkCuda(cudaEventSynchronize(stop), "event synchronize stop");
  checkCuda(cudaGetLastError(), "kernel launch measured");

  float total_ms = 0.0f;
  checkCuda(cudaEventElapsedTime(&total_ms, start, stop), "elapsed time");

  checkCuda(cudaEventDestroy(start), "destroy start event");
  checkCuda(cudaEventDestroy(stop), "destroy stop event");

  checkCuda(cudaFree(dQ), "free dQ");
  checkCuda(cudaFree(dK), "free dK");
  checkCuda(cudaFree(dV), "free dV");
  checkCuda(cudaFree(dO), "free dO");

  return total_ms / static_cast<float>(iters);
}

struct Result {
  std::string name;
  float avg_ms = 0.0f;
  double tflops = -1.0;
  bool ran = false;
};

} // namespace

int main(int argc, char** argv) {
  int bs = 1;
  int num_heads = 8;
  int len_q = 4096;
  int len_kv = 4096;
  int iters = 5;
  int warmup = 3;

  if (argc > 1) bs = std::atoi(argv[1]);
  if (argc > 2) num_heads = std::atoi(argv[2]);
  if (argc > 3) len_q = std::atoi(argv[3]);
  if (argc > 4) len_kv = std::atoi(argv[4]);
  if (argc > 5) iters = std::atoi(argv[5]);
  if (argc > 6) warmup = std::atoi(argv[6]);

  constexpr int DIM = 64;
  const double flops = 4.0 * static_cast<double>(bs) * num_heads * len_q * len_kv * DIM;

  std::vector<Result> results;
  results.reserve(3);

  auto run_and_record = [&](const char* name, float avg_ms) {
    Result r;
    r.name = name;
    r.avg_ms = avg_ms;
    r.ran = avg_ms > 0.0f;
    if (r.ran) {
      r.tflops = flops / (static_cast<double>(avg_ms) * 1.0e-3) / 1.0e12;
    }
    results.push_back(r);
  };

  run_and_record("BQ64_BKV32_W4", run_bench_cfg<64, 32, 4>(bs, num_heads, len_q, len_kv, iters, warmup));

  double best_tflops = -1.0;
  float best_ms = 0.0f;
  std::string best_name;
  for (const auto& r : results) {
    if (r.ran && r.tflops > best_tflops) {
      best_tflops = r.tflops;
      best_ms = r.avg_ms;
      best_name = r.name;
    }
  }

  std::cout << "FlashAttention forward benchmark" << std::endl;
  std::cout << "Config: bs=" << bs
            << ", heads=" << num_heads
            << ", len_q=" << len_q
            << ", len_kv=" << len_kv
            << ", dim=" << DIM << std::endl;
  std::cout << "Candidate results:" << std::endl;
  for (const auto& r : results) {
    if (r.ran) {
      std::cout << "  " << r.name << ": " << r.avg_ms << " ms, " << r.tflops << " TFLOPS" << std::endl;
    } else {
      std::cout << "  " << r.name << ": skipped (shape incompatibility)" << std::endl;
    }
  }
  std::cout << "Best kernel: " << best_name << std::endl;
  std::cout << "Average kernel time: " << best_ms << " ms" << std::endl;
  std::cout << "Throughput: " << best_tflops << " TFLOPS" << std::endl;

  return 0;
}
