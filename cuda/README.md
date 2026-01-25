# CUDA Sorting Algorithms - Compilation Guide

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (nvcc compiler)
- C++ compiler (g++)

## Compilation

To compile the CUDA sorting algorithms, you need to:

1. Compile the CUDA source files into object files:
```bash
nvcc -c cuda/merge_sort_cuda.cu -o cuda/merge_sort_cuda.o
nvcc -c cuda/quick_sort_cuda.cu -o cuda/quick_sort_cuda.o
nvcc -c cuda/radix_sort_cuda.cu -o cuda/radix_sort_cuda.o
nvcc -c cuda/counting_sort_cuda.cu -o cuda/counting_sort_cuda.o
nvcc -c cuda/bubble_sort_cuda.cu -o cuda/bubble_sort_cuda.o
```

2. Compile the test program with CUDA support:
```bash
g++ -DUSE_CUDA test.cpp cuda/*.o -o test_cuda -lcudart -L/usr/local/cuda/lib64
```