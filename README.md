# CUDA Parallel Sorting Algorithms - Lab Analysis

We use `test.cpp`. We test on `N_ITERATION`, calculate standard deviation and mean time with different seeds. Also, we test in function of `size` and `ratio`. Size is the data size and ratio is a value between 0 and 0.5 that describe how ordered it is. 0 is fully ordered and 0.5 is almost random.

We also tested in totally random data (with duplicate). 

## Stage 2: Application Bottleneck Analysis

### Execution Time and Complexity

**Time Complexity:**
- **Merge Sort**: O(n log n) - Each level processes n elements, log n levels
- **Radix Sort**: O(n log k) where k is the range - Processes each bit position
- **Counting Sort**: O(n + k) where k is the range - Linear scan + histogram
- **Bubble Sort**: O(n²) - Nested loops with n iterations each

**Bottlenecks:**
1. **Sequential dependencies**: Merge operations, partitioning steps
2. **Memory access patterns**: Sequential memory access
3. **Branch divergence**: Conditional operations reduce GPU efficiency
4. **Small problem sizes**: Overhead dominates for small arrays

### Parts to Accelerate

**Merge Sort:**
- **Accelerate**: Parallel merge operations, segment sorting
- **Why**: Independent segments can be sorted in parallel, merge can use parallel algorithms

**Radix Sort:**
- **Accelerate**: Bit extraction, prefix sum computation, scatter operations
- **Why**: Each bit position processed independently, prefix sum is highly parallelizable

**Counting Sort:**
- **Accelerate**: Histogram construction, prefix sum, scatter
- **Why**: All operations are data-parallel with independent elements

**Bubble Sort:**
- **Accelerate**: Comparison-swap operations within each phase
- **Why**: Odd-even transposition allows parallel comparisons

### Theoretical Performance

**Speedup Potential:**
- **Ideal speedup**: Up to number of CUDA cores (e.g., 2048 cores = up to 2048x)

## Stage 3: Acceleration Strategy

### CPU vs GPU Acceleration

**GPU Acceleration (CUDA):**
- **Merge Sort**: Parallel segment sorting and merging on GPU
- **Radix Sort**: Parallel bit processing and prefix sum on GPU
- **Counting Sort**: Parallel histogram and scatter on GPU
- **Bubble Sort**: Odd-even transposition network on GPU


## Stage 4: Performance Analysis

**Factors Affecting Speedup:**
1. **Problem Size**: Larger arrays benefit more (amortize transfer overhead)
2. **Ratio**: Larger ratio, less ordered the data is. 

We have **TERRIBLE** performance on GPU.
We do not have actual any speed-up and the higher the size of the array is, the better the speed-up is but it is alway at least 100x slower.

To look for results, look at `output` folder.

**Hypothesis**
1. Not enough time to actual do a really good algorithm.
2. Sorting algorithms are mostly sequential.
3. GPU memory transfer is SLOW and most GPU algorithms have a lot of data transfer to do. 
4. The GPU would be faster but at really large data size.

### Future Improvements

**Potential Optimizations:**
1. **OpenMP/SIMD**: Use OpenMP and SIMD on CPU (proably works best for smaller array)
2. **Shared Memory**: Use shared memory for frequently accessed data

**What Should Have Been Done Differently:**
1. Start libraries for baseline performance
2. Profile earlier to identify actual bottlenecks
3. Given more time at the modelisation
