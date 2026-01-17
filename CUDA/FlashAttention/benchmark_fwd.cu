#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <random>
#include <algorithm>
#include <iomanip>
#include "flash_attention_fwd.cuh"

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

constexpr int BLOCK_SIZE_Q = 64;
constexpr int BLOCK_SIZE_KV = 64;
constexpr int EMB_DIM = 32;

void random_init(float* data, size_t size) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

int main() {
    const int batch_size = 4;
    const int num_heads = 12;
    const int seq_len = 1024;
    const int warmup_iters = 5;
    const int benchmark_iters = 20;
    
    const int stride_seq = EMB_DIM; 
    const int stride_head = seq_len * EMB_DIM;
    const int stride_batch = num_heads * stride_head;

    const size_t total_elements = (size_t)batch_size * num_heads * seq_len * EMB_DIM;
    const size_t data_size_bytes = total_elements * sizeof(float);

    std::cout << "--- Flash Attention Benchmark ---" << std::endl;
    std::cout << "Batch Size: " << batch_size << std::endl;
    std::cout << "Num Heads:  " << num_heads << std::endl;
    std::cout << "Seq Len:    " << seq_len << std::endl;
    std::cout << "Emb Dim:    " << EMB_DIM << std::endl;
    std::cout << "Block Q:    " << BLOCK_SIZE_Q << std::endl;
    std::cout << "Block KV:   " << BLOCK_SIZE_KV << std::endl;
    std::cout << "---------------------------------" << std::endl;

    float *h_Q, *h_K, *h_V, *h_O;
    float *d_Q, *d_K, *d_V, *d_O, *d_M;

    // Host alloc
    h_Q = (float*)malloc(data_size_bytes);
    h_K = (float*)malloc(data_size_bytes);
    h_V = (float*)malloc(data_size_bytes);
    h_O = (float*)malloc(data_size_bytes);

    random_init(h_Q, total_elements);
    random_init(h_K, total_elements);
    random_init(h_V, total_elements);

    // Device alloc
    CHECK_CUDA(cudaMalloc(&d_Q, data_size_bytes));
    CHECK_CUDA(cudaMalloc(&d_K, data_size_bytes));
    CHECK_CUDA(cudaMalloc(&d_V, data_size_bytes));
    CHECK_CUDA(cudaMalloc(&d_O, data_size_bytes));
    
    size_t m_size_bytes = (size_t)batch_size * num_heads * seq_len * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_M, m_size_bytes));

    // Copy to Device
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, data_size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, data_size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, data_size_bytes, cudaMemcpyHostToDevice));
    
    // Zero out output
    CHECK_CUDA(cudaMemset(d_O, 0, data_size_bytes));


    // Kernel Launch Parameters
    int num_q_blocks = (seq_len + BLOCK_SIZE_Q - 1) / BLOCK_SIZE_Q;

    dim3 grid_dim(batch_size * num_heads * num_q_blocks);
    dim3 block_dim(128); 

    size_t smem_bytes = (BLOCK_SIZE_Q * EMB_DIM + 
                         BLOCK_SIZE_KV * EMB_DIM + 
                         BLOCK_SIZE_KV * EMB_DIM) * sizeof(float);

    float scale = 1.0f / sqrtf((float)EMB_DIM);

    // Warmup
    std::cout << "Starting warmup (" << warmup_iters << " iterations)..." << std::endl;
    for(int i = 0; i < warmup_iters; i++) {
        flash_attn_fwd<BLOCK_SIZE_Q, BLOCK_SIZE_KV, EMB_DIM><<<grid_dim, block_dim, smem_bytes>>>(
            d_Q, d_K, d_V, d_M, d_O,
            stride_batch, stride_head, stride_seq,
            batch_size, seq_len, num_heads,
            scale
        );
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    std::cout << "Starting benchmark (" << benchmark_iters << " iterations)..." << std::endl;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for(int i = 0; i < benchmark_iters; i++) {
        flash_attn_fwd<BLOCK_SIZE_Q, BLOCK_SIZE_KV, EMB_DIM><<<grid_dim, block_dim, smem_bytes>>>(
            d_Q, d_K, d_V, d_M, d_O,
            stride_batch, stride_head, stride_seq,
            batch_size, seq_len, num_heads,
            scale
        );
    }
    cudaEventRecord(stop);
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time_ms = milliseconds / benchmark_iters;
    
    double flops_per_run = 4.0 * (double)batch_size * (double)num_heads * 
                           (double)seq_len * (double)seq_len * (double)EMB_DIM;
    
    double gflops = (flops_per_run * 1e-9) / (avg_time_ms / 1000.0f);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Total Elements: " << total_elements << std::endl;
    std::cout << "Average Time:   " << avg_time_ms << " ms" << std::endl;
    std::cout << "Throughput:     " << gflops << " GFLOPS" << std::endl;

    // Cleanup
    free(h_Q); free(h_K); free(h_V); free(h_O);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_M);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}