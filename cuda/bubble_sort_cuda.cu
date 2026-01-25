#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define THREADS_PER_BLOCK 256

// Parallel bubble sort - different from merge sort
// Uses a wave-front approach where comparisons propagate through the array
__global__ void bubbleSortWaveKernel(int* arr, int n, int wave) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each wave compares pairs at specific positions
    // Wave 0: positions 0, 2, 4, 6, ... (compare with next)
    // Wave 1: positions 1, 3, 5, 7, ... (compare with next)
    int pos = 2 * idx + (wave % 2);
    
    if (pos + 1 < n) {
        if (arr[pos] > arr[pos + 1]) {
            int temp = arr[pos];
            arr[pos] = arr[pos + 1];
            arr[pos + 1] = temp;
        }
    }
}

extern "C" void bubble_sort_cuda(int a[], int n) {
    if (n <= 1) return;
    
    int* d_arr;
    size_t size = n * sizeof(int);
    
    cudaMalloc(&d_arr, size);
    cudaMemcpy(d_arr, a, size, cudaMemcpyHostToDevice);
    
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Bubble sort: do n-1 passes, each pass has multiple waves
    // This is different from merge sort which does n phases of odd-even
    for (int pass = 0; pass < n - 1; pass++) {
        // Each pass needs enough waves to ensure largest element bubbles to end
        // Do n waves per pass to guarantee all comparisons are made
        for (int wave = 0; wave < n; wave++) {
            bubbleSortWaveKernel<<<blocks, THREADS_PER_BLOCK>>>(d_arr, n, wave);
            cudaDeviceSynchronize();
        }
    }
    
    cudaMemcpy(a, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}
