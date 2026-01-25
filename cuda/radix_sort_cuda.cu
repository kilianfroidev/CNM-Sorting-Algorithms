#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

#define THREADS_PER_BLOCK 256

// Mark elements based on bit value (1 for 0-bit, 0 for 1-bit)
__global__ void radixMarkKernel(int* input, int* flags, int bitPos, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int bit = (input[idx] >> bitPos) & 1;
    flags[idx] = (bit == 0) ? 1 : 0;  // 1 for 0-bit, 0 for 1-bit
}

// Count zeros using reduction in shared memory
__global__ void countZerosKernel(int* flags, int* blockCounts, int n) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? flags[idx] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write block sum
    if (tid == 0) {
        blockCounts[blockIdx.x] = sdata[0];
    }
}

extern "C" void radix_sort_cuda(int a[], int n) {
    if (n <= 1) return;
    
    // Find max value to determine number of bits
    int maxVal = a[0];
    for (int i = 1; i < n; i++) {
        if (a[i] > maxVal) maxVal = a[i];
    }
    
    int numBits = 0;
    int temp = maxVal;
    while (temp > 0) {
        temp >>= 1;
        numBits++;
    }
    if (numBits == 0) numBits = 1;
    
    int* d_input;
    int* d_output;
    int* d_flags;
    int* d_blockCounts;
    size_t size = n * sizeof(int);
    
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_flags, size);
    
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaMalloc(&d_blockCounts, blocks * sizeof(int));
    
    cudaMemcpy(d_input, a, size, cudaMemcpyHostToDevice);
    
    int* h_input = new int[n];
    int* h_flags = new int[n];
    int* h_output = new int[n];
    
    // Process each bit from least significant to most significant
    for (int bit = 0; bit < numBits; bit++) {
        // Mark elements on GPU
        radixMarkKernel<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_flags, bit, n);
        cudaDeviceSynchronize();
        
        // Copy flags to CPU for deterministic scatter
        cudaMemcpy(h_input, d_input, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_flags, d_flags, size, cudaMemcpyDeviceToHost);
        
        // Count zeros and scatter on CPU (deterministic and correct)
        int zeroCount = 0;
        for (int i = 0; i < n; i++) {
            if (h_flags[i] == 1) zeroCount++;
        }
        
        int zeroPos = 0;
        int onePos = zeroCount;
        for (int i = 0; i < n; i++) {
            if (h_flags[i] == 1) {
                h_output[zeroPos++] = h_input[i];
            } else {
                h_output[onePos++] = h_input[i];
            }
        }
        
        // Copy result back to GPU
        cudaMemcpy(d_input, h_output, size, cudaMemcpyHostToDevice);
    }
    
    cudaMemcpy(a, d_input, size, cudaMemcpyDeviceToHost);
    
    delete[] h_input;
    delete[] h_flags;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_flags);
    cudaFree(d_blockCounts);
}
