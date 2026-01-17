#pragma once

template <int TILE_SIZE>
__global__ void shared_mem_block(float *A, float *B, float *C, int M, int K, int N){
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, bx = blockIdx.x;
    int ty = threadIdx.y, by = blockIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for(int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++){
        if(row < M && tile * TILE_SIZE + tx < K){
            As[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if(col < N && tile * TILE_SIZE + ty < K){
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else{
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k++){
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if(row < M && col < N)
        C[row * N + col] = sum;
}