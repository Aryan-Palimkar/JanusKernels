#pragma once

#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(x)                                                                                                  \
  {                                                                                                                    \
    auto error = x;                                                                                                    \
    if (error != cudaSuccess) {                                                                                        \
      std::cerr << "CUDA error - L" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl;                     \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  }

constexpr int WARP_SIZE = 32;

__device__ __host__ constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template<int STRIDE>
__device__ __forceinline__ uint32_t swizzle(uint32_t offset_bytes){
  static_assert(STRIDE % 16 == 0, "STRIDE must be 16-byte aligned for cp.async");

  constexpr uint32_t CHUNKS_PER_ROW = STRIDE / 16;
  if constexpr (CHUNKS_PER_ROW <= 1) return offset_bytes;

  const uint32_t row_idx = (offset_bytes / STRIDE) & 0b111;
  const uint32_t xor_chunks = row_idx % CHUNKS_PER_ROW;
  return offset_bytes ^ (xor_chunks << 4);
}


template<int TILE_ROWS, int TILE_COLS, int NUM_THREADS>
__device__ __forceinline__ void tileMemcpy(uint32_t dst, const half* src, const unsigned int src_stride, unsigned int tid){
  constexpr unsigned int TILE_COLS_VECTORIZED = TILE_COLS / 8;

  static_assert(NUM_THREADS % TILE_COLS_VECTORIZED == 0, "NUM_THREADS must divide vectorized tile columns");

  constexpr unsigned int ROWS_PER_ITER = NUM_THREADS / TILE_COLS_VECTORIZED;
  constexpr unsigned int NUM_ITERS = TILE_ROWS / ROWS_PER_ITER;
  const unsigned int thread_row = tid / TILE_COLS_VECTORIZED;
  const unsigned int thread_col = tid % TILE_COLS_VECTORIZED;

  #pragma unroll
  for(unsigned int i = 0; i < NUM_ITERS; i++){
    const unsigned int row = i * ROWS_PER_ITER + thread_row;

    const uint32_t tile_off = (row * TILE_COLS + thread_col * 8) * sizeof(half);
    const uint32_t dst_addr = dst + swizzle<TILE_COLS * sizeof(half)>(tile_off);

    const half* src_addr = src + row * src_stride + thread_col * 8;

    asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :
      : "r"(dst_addr), "l"(src_addr)
    );
  }
}


__device__ __forceinline__ void ldmatrix_x4(uint32_t regs[4], uint32_t addr){
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
    : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
    : "r"(addr)
  ); 
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t regs[2], uint32_t addr){
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
    : "=r"(regs[0]), "=r"(regs[1])
    : "r"(addr)
  ); 
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t regs[2], uint32_t addr) {
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
    : "=r"(regs[0]), "=r"(regs[1])
    : "r"(addr)
  );
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t regs[4], uint32_t addr){
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
    : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3])
    : "r"(addr)
  ); 
}

__device__ __forceinline__ void mma_m16n8k16(uint32_t A[4], uint32_t B[2], float D[4]){
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, "
    "{%4, %5, %6, %7}, "
    "{%8, %9}, "
    "{%10, %11, %12, %13};"
    : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
      "r"(B[0]), "r"(B[1]),
      "f"(D[0]), "f"(D[1]), "f"(D[2]), "f"(D[3])
    );
}