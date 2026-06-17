#include <metal_stdlib>
using namespace metal;

kernel void naive_matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* Output  [[buffer(2)]],
    constant uint3* dims  [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = (*dims)[0]; 
    uint N = (*dims)[1];
    uint K = (*dims)[2]; 
    
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) {
        return;
    }
    
    float sum = 0.0;
    for (uint k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    Output[row * N + col] = sum;
}