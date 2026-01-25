#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define THREADS_PER_BLOCK 256

// Build histogram
__global__ void histogramKernel(int* input, int* histogram, int n, int maxVal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int val = input[idx];
    if (val >= 0 && val <= maxVal) {
        atomicAdd(&histogram[val], 1);
    }
}

extern "C" void counting_sort_cuda(int a[], int n) {
    if (n <= 0) return;
    
    // Find max value
    int maxVal = a[0];
    for (int i = 1; i < n; i++) {
        if (a[i] > maxVal) maxVal = a[i];
    }
    
    int* d_input;
    int* d_histogram;
    
    size_t inputSize = n * sizeof(int);
    size_t histSize = (maxVal + 1) * sizeof(int);
    
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_histogram, histSize);
    
    cudaMemcpy(d_input, a, inputSize, cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, histSize);
    
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Build histogram on GPU
    histogramKernel<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_histogram, n, maxVal);
    cudaDeviceSynchronize();
    
    // Copy histogram to CPU and do rest on CPU (simple and correct)
    int* h_histogram = new int[maxVal + 1];
    cudaMemcpy(h_histogram, d_histogram, histSize, cudaMemcpyDeviceToHost);
    
    // Compute prefix sum
    int* h_prefixSum = new int[maxVal + 1];
    h_prefixSum[0] = 0;
    for (int i = 1; i <= maxVal; i++) {
        h_prefixSum[i] = h_prefixSum[i - 1] + h_histogram[i - 1];
    }
    
    // Copy input back to CPU for scatter (simple and correct)
    int* h_input = new int[n];
    cudaMemcpy(h_input, d_input, inputSize, cudaMemcpyDeviceToHost);
    
    // Scatter on CPU
    int* h_output = new int[n];
    for (int i = 0; i < n; i++) {
        int val = h_input[i];
        int pos = h_prefixSum[val]++;
        h_output[pos] = val;
    }
    
    // Copy result back
    memcpy(a, h_output, inputSize);
    
    delete[] h_histogram;
    delete[] h_prefixSum;
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_histogram);
}
