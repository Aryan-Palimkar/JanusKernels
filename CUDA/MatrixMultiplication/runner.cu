#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cmath>
#include <cublas_v2.h>
#include "naive.cuh"
#include "global_mem_coalesce.cuh"
#include "shared_mem_block.cuh"
#include "tiled_register_blocking.cuh"
#include "2dtiled_register_block.cuh"
#include "vectorized_smem.cuh"
#include <algorithm>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error in %s at line %d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


void init_matrix(float *mat, int rows, int cols){
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main(int argc, char **argv){
    if(argc < 5){
        std::cerr << "Usage: " << argv[0] << " <M> <K> <N> <kernel_id>\n";
        std::cerr << "  kernel_id:\n";
        std::cerr << "    0: naive\n";
        std::cerr << "    1: global_mem_coalesce\n";
        std::cerr << "    2: shared_mem_block (Tiled)\n";
        std::cerr << "    3: tiled_register_blocking (1D Work-per-thread)\n";
        std::cerr << "    4: tiled_2d_register_blocking (2D Work-per-thread)\n";
        std::cerr << "    5: cublas_gemm\n";
        std::cerr << "    6: tiled_2d_register_blocking_vectorized\n";
        return 1;
    }

    const int M = std::stoi(argv[1]);
    const int K = std::stoi(argv[2]);
    const int N = std::stoi(argv[3]);
    const int kernel_id = std::stoi(argv[4]);
    if(kernel_id < 0 || kernel_id > 6){
        std::cerr << "Invalid kernel_id: " << kernel_id << std::endl;
        return 1;
    }

    std::cout << "Matrix Dimensions: M=" << M << ", K=" << K << ", N=" << N << std::endl;

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);
    std::vector<float> h_C_ref(M * N);

    srand(time(NULL));
    init_matrix(h_A.data(), M, K);
    init_matrix(h_B.data(), K, N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Kernel selection and benchmarking ---
    std::string kernel_name;
    float ms = 0.0f;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    float *d_C_ref;
    CUDA_CHECK(cudaMalloc(&d_C_ref, size_C));
    CUDA_CHECK(cudaMemset(d_C_ref, 0, size_C));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha, d_B, N, d_A, K, &beta, d_C_ref, N));
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, size_C, cudaMemcpyDeviceToHost));

    auto verify_results = [&](const std::string &name){
        CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost));
        double max_abs_err = 0.0;
        double max_rel_err = 0.0;
        for(size_t idx = 0; idx < h_C.size(); ++idx){
            double ref = static_cast<double>(h_C_ref[idx]);
            double val = static_cast<double>(h_C[idx]);
            double diff = std::fabs(val - ref);
            max_abs_err = std::max(max_abs_err, diff);
            double denom = std::max(1e-6, std::fabs(ref));
            max_rel_err = std::max(max_rel_err, diff / denom);
        }
        std::cout << "Verification (" << name << "): max_abs_error = "
                  << max_abs_err << ", max_rel_error = " << max_rel_err
                  << (max_rel_err < 1e-3 ? " [PASS]" : " [WARN]") << std::endl;
    };

    switch(kernel_id){
        case 0: {
            kernel_name = "naive";
            dim3 block(16, 16);
            dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
            std::cout << "Benchmarking: " << kernel_name << std::endl;

            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            verify_results(kernel_name);

            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            CUDA_CHECK(cudaEventRecord(start));
            naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
            CUDA_CHECK(cudaEventRecord(stop));
            break;
        }
        case 1: {
            kernel_name = "global_mem_coalesce";
            const int BLOCK_SIZE = 32;
            dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
            dim3 block(BLOCK_SIZE * BLOCK_SIZE);
            std::cout << "Benchmarking: " << kernel_name << std::endl;

            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            global_mem_coalesce<BLOCK_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            verify_results(kernel_name);

            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            CUDA_CHECK(cudaEventRecord(start));
            global_mem_coalesce<BLOCK_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
            CUDA_CHECK(cudaEventRecord(stop));
            break;
        }
        case 2: {
            kernel_name = "shared_mem_block (Tiled)";
            const int TILE_SIZE = 32;
            dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
            dim3 block(TILE_SIZE, TILE_SIZE);
            std::cout << "Benchmarking: " << kernel_name << std::endl;

            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            shared_mem_block<TILE_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            verify_results(kernel_name);

            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            CUDA_CHECK(cudaEventRecord(start));
            shared_mem_block<TILE_SIZE><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
            CUDA_CHECK(cudaEventRecord(stop));
            break;
        }
        case 3: {
            kernel_name = "tiled_register_blocking (1D Work-per-thread)";
            const int TILE_SIZE = 64;
            const int TM = 8;
            dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
            dim3 block(TILE_SIZE, TILE_SIZE / TM);
            std::cout << "Benchmarking: " << kernel_name << std::endl;

            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            tiled_register_blocking<TILE_SIZE, TM><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            verify_results(kernel_name);

            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            CUDA_CHECK(cudaEventRecord(start));
            tiled_register_blocking<TILE_SIZE, TM><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
            CUDA_CHECK(cudaEventRecord(stop));
            break;
        }
        case 4: {
            kernel_name = "tiled_2d_register_blocking (2D Work-per-thread)";
            const int TILE_SIZE = 64;
            const int TM = 4;
            const int TN = 4;
            dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
            dim3 block(TILE_SIZE / TN, TILE_SIZE / TM);
            std::cout << "Benchmarking: " << kernel_name << std::endl;

            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            tiled_2d_register_blocking<TILE_SIZE, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            verify_results(kernel_name);

            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            CUDA_CHECK(cudaEventRecord(start));
            tiled_2d_register_blocking<TILE_SIZE, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
            CUDA_CHECK(cudaEventRecord(stop));
            break;
        }
        case 5: {
            kernel_name = "tiled_2d_register_blocking_vectorized";
            const int TILE_SIZE = 64;
            const int TM = 4;
            const int TN = 4;
            dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
            dim3 block(TILE_SIZE / TN, TILE_SIZE / TM);
            std::cout << "Benchmarking: " << kernel_name << std::endl;

            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            tiled_2d_register_blocking_vectorized<TILE_SIZE, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            verify_results(kernel_name);

            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            CUDA_CHECK(cudaEventRecord(start));
            tiled_2d_register_blocking_vectorized<TILE_SIZE, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
            CUDA_CHECK(cudaEventRecord(stop));
            break;
        }
        case 6: {
            kernel_name = "cublas_gemm";
            std::cout << "Benchmarking: " << kernel_name << std::endl;

            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                     &alpha, d_B, N, d_A, K, &beta, d_C, N));
            CUDA_CHECK(cudaDeviceSynchronize());
            verify_results(kernel_name);

            CUDA_CHECK(cudaMemset(d_C, 0, size_C));
            CUDA_CHECK(cudaEventRecord(start));
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                     &alpha, d_B, N, d_A, K, &beta, d_C, N));
            CUDA_CHECK(cudaEventRecord(stop));
            break;
        }
    }

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    double num_ops = 2.0 * (double)M * (double)K * (double)N;
    double gflops = (num_ops / (ms / 1000.0)) / 1e9;

    std::cout << "\n--- Results for " << kernel_name << " ---\n";
    std::cout << "Execution Time: " << ms << " ms\n";
    std::cout << "Throughput:     " << gflops << " GFLOPS\n";

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_C_ref));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}