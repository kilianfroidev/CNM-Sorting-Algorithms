#include <stdio.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>

#include "helper/DataGenerator.cpp"
#include "helper/timer.h"
#include "sorting-methods/all.h"

#define NUM_SORT 5
#define NUM_ITERATION 100

// Must be in the same order (index based)
typedef void (*sort_ptr)(int[], int);
sort_ptr sort_methods[] = {bubble_sort_optimize1, merge_sort, quick_sort, couting_sort, radix_sort};
std::string sort_name[] = {"basic_bubble_sort", "basic_merge_sort", "basic_quick_sort", "basic_counting_sort", "basic_radix_sort"};

struct Stat
{
    int sortId;
    int size;
    double ratio;
    double meanTime;
    double stdDeviation;
    bool areAllSorted;
};

bool isSorted(int data[], int size)
{
    for (int i = 0; i < size - 1; ++i)
    {
        if (data[i] <= data[i + 1])
        {
            return false;
        }
    }
    return true;
}

void timeMeasureData(Stat sortStat[NUM_SORT], int size, double ratio, int dataType)
{
    int64_t times[NUM_SORT][NUM_ITERATION] = {};
    int64_t sumY[NUM_SORT] = {};
    int64_t sumY2[NUM_SORT] = {};
    bool notAllSorted[NUM_SORT] = {};
    for (int seedId = 0; seedId < NUM_ITERATION; ++seedId)
    {
        int data[size];
        // Generate once
        GenerateData(data, size, dataType, ratio, seedId);

        for (int algoId = 0; algoId < NUM_SORT; ++algoId)
        {
            int data_temp[size];
            std::copy(data, data + size, data_temp); // DeepCopy

            // Execution
            auto last_time = std::chrono::high_resolution_clock::now();
            sort_methods[algoId](data_temp, size);
            auto cur_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(cur_time - last_time);

            times[algoId][seedId] = duration.count();
            sumY[algoId] += times[algoId][seedId];
            sumY2[algoId] += times[algoId][seedId] * times[algoId][seedId];

            if (!isSorted(data_temp, size))
            {
                notAllSorted[algoId] = true;
                std::cerr << "Algorithme " + sort_name[algoId] + " ne trie pas correctement!" << std::endl;
            }
        }
    }

    for (int algoId = 0; algoId < NUM_SORT; ++algoId)
    {
        sortStat[algoId].meanTime = static_cast<double>(sumY[algoId]) / NUM_ITERATION;
        sortStat[algoId].stdDeviation = sqrt((static_cast<double>(sumY[algoId]) * static_cast<double>(sumY[algoId]) / NUM_ITERATION - static_cast<double>(sumY2[algoId])) / (NUM_ITERATION - 1));
        sortStat[algoId].ratio = ratio;
        sortStat[algoId].size = size;
        sortStat[algoId].sortId = algoId;
        sortStat[algoId].areAllSorted = !notAllSorted[algoId];
    }
}

int main()
{

    int numberOfIteration = 100;
    int numberOfSort = 6;

    // size_t seed = 0;
    int dataPowerSize = 20;       // to 1M elements
    int dataSizes[dataPowerSize]; // Specify the size as needed
    generateSize(dataSizes, dataPowerSize);

    int dataRatioSize = 20;
    double dataRatios[dataRatioSize];
    generateRatio(dataRatios, dataRatioSize);

    // Sequencials
    for (int sizeId = 0; sizeId < dataPowerSize; ++sizeId)
    {
        int a[dataSizes[sizeId]];
        // Almost Sorted
        for (int ratioId = 0; ratioId < dataRatioSize; ++ratioId)
        {

            // basic
            // Seed Variation
            for (int sortId = 0; sortId < numberOfSort; ++sortId)
            {
                int64_t times[numberOfIteration];
                Stat stat = timeMeasureData();
            }

            // parallel

            for (int sortId = 0; sortId < numberOfSort; ++sortId)
            {
                for (size_t seed = 0; seed < numberOfIteration; ++seed)
                {
                    GenerateData(a, dataSizes[sizeId], SORTED_DATA, dataRatios[ratioId], seed);

                    /*
                    Call the parallel sort + time measure + statistics
                    */
                }
            }
        }
        // Random
        int a[dataSizes[sizeId]];
        GenerateData(a, dataSizes[sizeId], RANDOM_DATA, 0, seed);

        // Almost Inverted
        for (int ratioId = 0; ratioId < dataRatioSize; ++ratioId)
        {
            // Seed Variation
            int a[dataSizes[sizeId]];
            GenerateData(a, dataSizes[sizeId], SORTED_DATA, dataRatios[ratioId], seed);
        }
    }
}