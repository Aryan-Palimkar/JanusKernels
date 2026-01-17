#pragma once

__global__ void naive(float *A, float *B, float *C, int M, int K, int N){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    if(row < M && col < N){
        for(int i = 0; i < K; i++){
            sum += A[row * K + i] * B[N * i + col]; 
        }

        C[row * N + col] = sum;
    }
}