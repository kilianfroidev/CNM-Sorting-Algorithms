#ifndef CUDA_SORT_HEADER
#define CUDA_SORT_HEADER

// CUDA sorting function declarations
// These functions are implemented in .cu files and must be compiled with nvcc

extern "C" {
    void merge_sort_cuda(int a[], int n);
    void quick_sort_cuda(int a[], int n);
    void radix_sort_cuda(int a[], int n);
    void counting_sort_cuda(int a[], int n);
    void bubble_sort_cuda(int a[], int n);
}

#endif

