#include <stdio.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>

#include "helper/DataGenerator.cpp"
#include "helper/timer.h"
#include "sorting-methods/all.h"

typedef void (*sort_ptr)(int[], int);
sort_ptr sort_basic_methods[] = {
    selection_sort, selection_sort_optimize1, insertion_sort,
    binary_insertion_sort, bubble_sort, bubble_sort_optimize1,
    shaker_sort, shaker_sort_optimize1, shell_sort,
    heap_sort, merge_sort, quick_sort,
    couting_sort, radix_sort, flash_sort};

sort_ptr sort_GPU_methods[];

int main()
{
    size_t seed = 0;
    int dataPowerSize = 20;       // to 1M elements
    int dataSizes[dataPowerSize]; // Specify the size as needed
    generateSize(dataSizes, dataPowerSize);

    int dataRatioSize = 20;
    double dataRatios[dataRatioSize];
    generateRatio(dataRatios, dataRatioSize);

    // Sequencials
    for (int sizeId = 0; sizeId < dataPowerSize; ++sizeId)
    {
        // Almost Sorted

        // Random

        // Almost Inverted

        // then, cycle through each sequential/parrael
    }

    int n = 100;
    int a[n];
    GenerateData(a, n, 2, 0.1, 3);
    for (int i = 0; i < n; ++i)
    {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}