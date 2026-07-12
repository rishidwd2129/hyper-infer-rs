#include <metal_stdlib>
using namespace metal;
// 1. 'kernel' defines this as a Compute Shader (general-purpose math, not graphics).
kernel void naive_matmul(
    // 2. 'device' means this pointer lives in Global GPU Memory (Unified Memory).
    // The [[buffer(X)]] tags are the binding slots. Rust will drop memory directly into these slots.
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* Output  [[buffer(2)]],
    
    // 3. 'constant' is for tiny, frequently accessed data. It lives in ultra-fast cache.
    constant uint3* dims  [[buffer(3)]],
    
    // 4. THIS IS THE MAGIC: The hardware assigns each micro-thread a unique 2D coordinate.
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = (*dims)[0]; // Output Rows
    uint N = (*dims)[1]; // Output Columns
    uint K = (*dims)[2]; // Shared inner dimension
    
    // gid.y is the row we are computing. gid.x is the column.
    uint row = gid.y;
    uint col = gid.x;
    
    // Guard: Threadgroups are launched in blocks (e.g., 16x16). 
    // If our matrix size isn't a perfect multiple of 16, some threads fall out of bounds. Kill them.
    if (row >= M || col >= N) {
        return;
    }
    
    // 5. Compute exactly ONE float of the output matrix.
    float sum = 0.0;
    for (uint k = 0; k < K; ++k) {
        // A is read row-wise, B is read col-wise
        sum += A[row * K + k] * B[k * N + col];
    }
    
    // 6. Write the final computed float to global memory
    Output[row * N + col] = sum;
}