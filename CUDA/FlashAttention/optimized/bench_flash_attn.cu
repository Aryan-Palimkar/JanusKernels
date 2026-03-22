#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>

#include "attention.cu"

namespace {

inline void checkCuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << msg << " -> " << cudaGetErrorString(err) << std::endl;
    std::exit(1);
  }
}

float run_bench(int bs, int num_heads, int len_q, int len_kv, int iters, int warmup) {
  constexpr int BLOCK_Q = 64;
  constexpr int BLOCK_KV = 64;
  constexpr int DIM = 64;
  constexpr int NUM_WARPS = 4;

  if (len_q % BLOCK_Q != 0 || len_kv % BLOCK_KV != 0) {
    std::cerr << "len_q must be multiple of " << BLOCK_Q
              << " and len_kv must be multiple of " << BLOCK_KV << std::endl;
    std::exit(1);
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
  const size_t smem_bytes = static_cast<size_t>(BLOCK_Q + 2 * BLOCK_KV) * DIM * sizeof(half);

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

} // namespace

int main(int argc, char** argv) {
  int bs = 1;
  int num_heads = 8;
  int len_q = 4096;
  int len_kv = 8192;
  int iters = 5;
  int warmup = 3;

  if (argc > 1) bs = std::atoi(argv[1]);
  if (argc > 2) num_heads = std::atoi(argv[2]);
  if (argc > 3) len_q = std::atoi(argv[3]);
  if (argc > 4) len_kv = std::atoi(argv[4]);
  if (argc > 5) iters = std::atoi(argv[5]);
  if (argc > 6) warmup = std::atoi(argv[6]);

  constexpr int DIM = 64;

  const float avg_ms = run_bench(bs, num_heads, len_q, len_kv, iters, warmup);

  const double flops = 4.0 * static_cast<double>(bs) * num_heads * len_q * len_kv * DIM;
  const double tflops = flops / (static_cast<double>(avg_ms) * 1.0e-3) / 1.0e12;

  std::cout << "FlashAttention forward benchmark" << std::endl;
  std::cout << "Config: bs=" << bs
            << ", heads=" << num_heads
            << ", len_q=" << len_q
            << ", len_kv=" << len_kv
            << ", dim=" << DIM << std::endl;
  std::cout << "Average kernel time: " << avg_ms << " ms" << std::endl;
  std::cout << "Throughput: " << tflops << " TFLOPS" << std::endl;

  return 0;
}
