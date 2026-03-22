#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <vector>
#include <random>
#define N_STAGES 3

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                           \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status__ = (call);                                      \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                               \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, \
                         static_cast<int>(status__));                          \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__device__ __forceinline__ void ldmatrix_x4(unsigned *dst, uint4* src){
    unsigned ptx_src = __cvta_generic_to_shared(src);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(ptx_src)
    );
}

__device__ __forceinline__ void ldmatrix_x2(unsigned* dst, uint4* src){
    unsigned ptx_src = __cvta_generic_to_shared(src);
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst[0]), "=r"(dst[1])
        : "r"(ptx_src)
    );
}

__device__ __forceinline__ void ldmatrix_x2_trans(unsigned* dst, uint4* src){
    unsigned ptx_src = __cvta_generic_to_shared(src);
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(dst[0]), "=r"(dst[1])
        : "r"(ptx_src)
    );
}

__device__ __forceinline__ void mma_m16n8k16(const unsigned* A, const unsigned* B, float* C, float* D){
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
    );
}

__device__ __forceinline__ void cp_async(uint4* dst, const uint4* src){
    unsigned ptx_dst = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
        :: "r"(ptx_dst), "l"(src), "n"(16)
    );
}

__forceinline__ 
__device__ void mma_m16n8k16_f16(const unsigned *A, const unsigned *B, unsigned *C, unsigned *D) {
  asm (
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(D[0]), "=r"(D[1])
      :
      "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
      "r"(B[0]), "r"(B[1]),
      "r"(C[0]), "r"(C[1])
      );
}


template <int M, int N, int K>
__global__ __launch_bounds__(16 * 16) void mma_gemm_f16f16_f32(const half* A, const half* B, float* C){
    __shared__ uint4 A_smem[N_STAGES * 64][8];
    __shared__ uint4 B_smem[N_STAGES * 64][8];

    uint4 (*a_load_ptr)[8];
    uint4 (*b_load_ptr)[8];
    uint4 (*a_store_ptr)[8];
    uint4 (*b_store_ptr)[8];

    const int block_m = blockIdx.y * 128;
    const int block_n = blockIdx.x * 128;
    
    const uint4* A_block_gmem = reinterpret_cast<const uint4*>(A + block_m * K);
    const uint4* B_block_gmem = reinterpret_cast<const uint4*>(B + block_n * K);

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    const int warp_offset_a = 32 * (warp_id / 4);
    const int warp_offset_b = 16 * (warp_id % 4);

    unsigned A_register[4][8];
    unsigned B_register[4][4];
    float acc_register[4][4][4] = {0.0f};

    const int store_row = warp_id * 4 + lane_id / 8;
    const int store_col = (lane_id % 8) ^ (lane_id / 8);

    const int load_row_a = (lane_id % 16) / 2;
    const int load_col_a = (lane_id / 8 + 4 * (lane_id % 2)) ^ (load_row_a % 4);
    const int load_row_b = (lane_id % 8) / 2;
    const int load_col_b = (lane_id / 8 + 4 * (lane_id % 2)) ^ (load_row_b % 4);

    const uint4* a_global_addr = A_block_gmem + (warp_id * 8 + lane_id / 4) * (K / 8) + (lane_id % 4);
    const uint4* b_global_addr = B_block_gmem + (warp_id * 8 + lane_id / 4) * (K / 8) + (lane_id % 4);

    for (int stage = 0; stage < N_STAGES - 1; stage++) {
        int k_start = stage * 4;
        a_store_ptr = A_smem + 64 * stage;
        b_store_ptr = B_smem + 64 * stage;
        
        cp_async(a_store_ptr[store_row] + store_col, a_global_addr + k_start);
        cp_async(a_store_ptr[store_row + 32] + store_col, a_global_addr + 64 * (K / 8) + k_start);
        cp_async(b_store_ptr[store_row] + store_col, b_global_addr + k_start);
        cp_async(b_store_ptr[store_row + 32] + store_col, b_global_addr + 64 * (K / 8) + k_start);
        
        asm volatile("cp.async.commit_group;\n" ::);
    }

    for (int block_k = 0; block_k < K / 32; block_k++) {
        int k_start = (N_STAGES - 1 + block_k) * 4;
        
        a_store_ptr = A_smem + 64 * ((block_k + N_STAGES - 1) % N_STAGES);
        b_store_ptr = B_smem + 64 * ((block_k + N_STAGES - 1) % N_STAGES);
        a_load_ptr = A_smem + 64 * (block_k % N_STAGES);
        b_load_ptr = B_smem + 64 * (block_k % N_STAGES);

        asm volatile("cp.async.wait_group %0;\n" :: "n"(N_STAGES - 2));
        __syncthreads();

        for (int m = 0; m < 4; m++) {
            ldmatrix_x4(A_register[m], a_load_ptr[m * 8 + warp_offset_a + load_row_a] + load_col_a);
            ldmatrix_x4(A_register[m] + 4, a_load_ptr[m * 8 + warp_offset_a + load_row_a] + (load_col_a ^ 2));
        }
        for (int n = 0; n < 4; n++) {
            ldmatrix_x2(B_register[n], b_load_ptr[n * 4 + warp_offset_b + load_row_b] + load_col_b);
            ldmatrix_x2(B_register[n] + 2, b_load_ptr[n * 4 + warp_offset_b + load_row_b] + (load_col_b ^ 2));
        }

        k_start = (k_start > 512 - 4) ? 512 - 4 : k_start;
        
        cp_async(a_store_ptr[store_row] + store_col, a_global_addr + k_start);
        cp_async(a_store_ptr[store_row + 32] + store_col, a_global_addr + 64 * (K / 8) + k_start);
        cp_async(b_store_ptr[store_row] + store_col, b_global_addr + k_start);
        cp_async(b_store_ptr[store_row + 32] + store_col, b_global_addr + 64 * (K / 8) + k_start);
        
        asm volatile("cp.async.commit_group;\n" ::);

        for (int m = 0; m < 4; m++) {
            for (int n = 0; n < 4; n++) {
                mma_m16n8k16(A_register[m], B_register[n], acc_register[m][n], acc_register[m][n]);
                mma_m16n8k16(A_register[m] + 4, B_register[n] + 2, acc_register[m][n], acc_register[m][n]);
            }
        }
    }
    
    const int group_id = lane_id >> 2;
    const int group_lane_id = lane_id % 4;
    
    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 4; n++) {
            float2 d0 = make_float2(acc_register[m][n][0], acc_register[m][n][1]);
            float2 d2 = make_float2(acc_register[m][n][2], acc_register[m][n][3]);
            
            int c_row_0 = block_m + m * 16 + 2 * warp_offset_a + group_id;
            int c_row_2 = c_row_0 + 8;
            int c_col = block_n + n * 8 + 2 * warp_offset_b + 2 * group_lane_id;
            
            float2* c_out_0 = reinterpret_cast<float2*>(&C[c_row_0 * N + c_col]);
            float2* c_out_2 = reinterpret_cast<float2*>(&C[c_row_2 * N + c_col]);
            
            *c_out_0 = d0;
            *c_out_2 = d2;
        }
    }
}

int main() {
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;
    constexpr int warmup = 3;
    constexpr int iters = 5;

    const size_t bytes_a = static_cast<size_t>(M) * K * sizeof(half);
    const size_t bytes_b = static_cast<size_t>(K) * N * sizeof(half);
    const size_t bytes_c = static_cast<size_t>(M) * N * sizeof(float);

    std::vector<half> h_a(static_cast<size_t>(M) * K);
    std::vector<half> h_b(static_cast<size_t>(K) * N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (size_t i = 0; i < h_a.size(); ++i) {
        h_a[i] = __float2half(dist(rng));
    }
    for (size_t i = 0; i < h_b.size(); ++i) {
        h_b[i] = __float2half(dist(rng));
    }

    half* d_a = nullptr;
    half* d_b = nullptr;
    float* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
    CUDA_CHECK(cudaMalloc(&d_b, bytes_b));
    CUDA_CHECK(cudaMalloc(&d_c, bytes_c));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes_a, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes_b, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c, 0, bytes_c));

    dim3 block(32, 8);
    dim3 grid(N / 128, M / 128);

    for (int i = 0; i < warmup; ++i) {
        mma_gemm_f16f16_f32<M, N, K><<<grid, block>>>(d_a, d_b, d_c);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        mma_gemm_f16f16_f32<M, N, K><<<grid, block>>>(d_a, d_b, d_c);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    const double ms_per_iter = static_cast<double>(ms) / iters;

    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double tflops = flops / (ms_per_iter * 1.0e-3) / 1.0e12;

    const double bytes_moved = static_cast<double>(bytes_a + bytes_b + bytes_c);
    const double bandwidth_gbs = bytes_moved / (ms_per_iter * 1.0e-3) / 1.0e9;

    std::printf("mma_gemm_f16f16_f32 benchmark\n");
    std::printf("M=%d N=%d K=%d\n", M, N, K);
    std::printf("Average kernel time: %.3f ms\n", ms_per_iter);
    std::printf("Throughput: %.3f TFLOP/s\n", tflops);
    std::printf("Effective bandwidth: %.3f GB/s\n", bandwidth_gbs);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cudaEvent_t cublas_start, cublas_stop;
    CUDA_CHECK(cudaEventCreate(&cublas_start));
    CUDA_CHECK(cudaEventCreate(&cublas_stop));

    for (int i = 0; i < warmup; ++i) {
        CUBLAS_CHECK(cublasGemmEx(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_b, CUDA_R_16F, N,
            d_a, CUDA_R_16F, K,
            &beta,
            d_c, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(cublas_start));
    for (int i = 0; i < iters; ++i) {
        CUBLAS_CHECK(cublasGemmEx(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_b, CUDA_R_16F, N,
            d_a, CUDA_R_16F, K,
            &beta,
            d_c, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaEventRecord(cublas_stop));
    CUDA_CHECK(cudaEventSynchronize(cublas_stop));

    float cublas_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&cublas_ms, cublas_start, cublas_stop));
    const double cublas_ms_per_iter = static_cast<double>(cublas_ms) / iters;
    const double cublas_tflops = flops / (cublas_ms_per_iter * 1.0e-3) / 1.0e12;
    const double cublas_bandwidth_gbs = bytes_moved / (cublas_ms_per_iter * 1.0e-3) / 1.0e9;

    std::printf("\ncuBLAS GEMM benchmark (same conditions)\n");
    std::printf("Average kernel time: %.3f ms\n", cublas_ms_per_iter);
    std::printf("Throughput: %.3f TFLOP/s\n", cublas_tflops);
    std::printf("Effective bandwidth: %.3f GB/s\n", cublas_bandwidth_gbs);

    CUDA_CHECK(cudaEventDestroy(cublas_start));
    CUDA_CHECK(cudaEventDestroy(cublas_stop));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}