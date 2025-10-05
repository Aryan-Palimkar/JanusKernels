#pragma once

template <int BLOCK_SIZE>
__global__ void global_mem_coalesce(float *A, float *B, float *C, int M, int K, int N){
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE;
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE;

    if(row < M && col < N){
        float sum = 0.0f;
        for(int i = 0; i < K; i++){
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}