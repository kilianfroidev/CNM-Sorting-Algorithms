#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

#define THREADS_PER_BLOCK 256

// Odd-even transposition sort - simple and correct
__global__ void mergeOddEvenSortKernel(int* arr, int n, int phase) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (phase % 2 == 0) {
        // Even phase: compare (0,1), (2,3), (4,5), ...
        int pos = 2 * idx;
        if (pos + 1 < n && arr[pos] > arr[pos + 1]) {
            int temp = arr[pos];
            arr[pos] = arr[pos + 1];
            arr[pos + 1] = temp;
        }
    } else {
        // Odd phase: compare (1,2), (3,4), (5,6), ...
        int pos = 2 * idx + 1;
        if (pos + 1 < n && arr[pos] > arr[pos + 1]) {
            int temp = arr[pos];
            arr[pos] = arr[pos + 1];
            arr[pos + 1] = temp;
        }
    }
}

extern "C" void merge_sort_cuda(int a[], int n) {
    if (n <= 1) return;
    
    int* d_arr;
    size_t size = n * sizeof(int);
    
    cudaMalloc(&d_arr, size);
    cudaMemcpy(d_arr, a, size, cudaMemcpyHostToDevice);
    
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    for (int phase = 0; phase < n; phase++) {
        mergeOddEvenSortKernel<<<blocks, THREADS_PER_BLOCK>>>(d_arr, n, phase);
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(a, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}
