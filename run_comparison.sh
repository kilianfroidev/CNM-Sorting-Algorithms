#!/bin/bash

# Script to run CPU vs CUDA comparison and generate CSV files

echo "Compiling CUDA object files..."
cd cuda
nvcc -c merge_sort_cuda.cu -o merge_sort_cuda.o
nvcc -c radix_sort_cuda.cu -o radix_sort_cuda.o
nvcc -c counting_sort_cuda.cu -o counting_sort_cuda.o
nvcc -c bubble_sort_cuda.cu -o bubble_sort_cuda.o
cd ..

echo "Compiling test program with CUDA support..."
g++ -DUSE_CUDA test.cpp cuda/*.o -o test_cuda -lcudart -L/usr/local/cuda/lib64

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Running comparison (CPU vs CUDA)..."
    echo "This may take a while..."
    ./test_cuda --compare
    echo ""
    echo "Comparison CSV files generated:"
    echo "  - comparison_almost_sorted.csv"
    echo "  - comparison_random.csv"
    echo "  - comparison_almost_inverted.csv"
    echo "  - results_almost_sorted_cpu.csv"
    echo "  - results_random_cpu.csv"
    echo "  - results_almost_inverted_cpu.csv"
    echo "  - results_almost_sorted_cuda.csv"
    echo "  - results_random_cuda.csv"
    echo "  - results_almost_inverted_cuda.csv"
else
    echo "Compilation failed!"
    exit 1
fi

