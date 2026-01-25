#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include "helper/DataGenerator.cpp"
#include "helper/timer.h"
#include "sorting-methods/all.h"

// Include CUDA headers if available
#ifdef USE_CUDA
#include "cuda/cuda_sort.h"
#endif

#define NUM_SORT 8
#define NUM_ITERATION 16

// Must match the same order - 10 sorting algorithms
typedef void (*sort_ptr)(int[], int);
sort_ptr sort_methods[] = {bubble_sort_optimize1,
                           merge_sort,
                           couting_sort,
                           radix_sort,
                           heap_sort,
                           insertion_sort,
                           selection_sort,
                           shell_sort,
                           shaker_sort};

std::string sort_name[] = {"bubble_sort",
                           "merge_sort",
                           "counting_sort",
                           "radix_sort",
                           "heap_sort",
                           "insertion_sort",
                           "selection_sort",
                           "shell_sort",
                           "shaker_sort"};

#ifdef USE_CUDA
// CUDA versions (only first 5 have CUDA implementations)
sort_ptr sort_methods_cuda[] = {bubble_sort_cuda,
                                 merge_sort_cuda,
                                 counting_sort_cuda,
                                 radix_sort_cuda,
                                 nullptr,  // heap_sort - no CUDA
                                 nullptr,  // insertion_sort - no CUDA
                                 nullptr,  // selection_sort - no CUDA
                                 nullptr,  // shell_sort - no CUDA
                                 nullptr}; // shaker_sort - no CUDA

std::string sort_name_cuda[] = {"cuda_bubble_sort",
                                "cuda_merge_sort",
                                "cuda_counting_sort",
                                "cuda_radix_sort",
                                "cuda_heap_sort",      // placeholder
                                "cuda_insertion_sort", // placeholder
                                "cuda_selection_sort", // placeholder
                                "cuda_shell_sort",     // placeholder
                                "cuda_shaker_sort"};  // placeholder
#endif

struct Stat
{
    int sortId;
    int size;
    double ratio;
    double meanTime;
    double stdDeviation;
    bool areAllSorted;
};

// Check ascending order
bool isSorted(const std::vector<int> &data)
{
    for (size_t i = 0; i + 1 < data.size(); ++i)
    {
        if (data[i] > data[i + 1])
            return false;
    }
    return true;
}

void timeMeasureData(std::vector<Stat> &sortStat, int size, double ratio, int dataType, bool useCuda = false)
{
    std::vector<double> sumY(NUM_SORT, 0.0);
    std::vector<double> sumY2(NUM_SORT, 0.0);
    std::vector<bool> notAllSorted(NUM_SORT, false);

#ifdef USE_CUDA
    sort_ptr* methods = useCuda ? sort_methods_cuda : sort_methods;
    std::string* names = useCuda ? sort_name_cuda : sort_name;
#else
    sort_ptr* methods = sort_methods;
    std::string* names = sort_name;
    if (useCuda) {
        std::cerr << "Warning: CUDA not available, using CPU versions\n";
    }
#endif

    for (int seedId = 0; seedId < NUM_ITERATION; ++seedId)
    {
        std::vector<int> data(size);
        GenerateData(data.data(), size, dataType, ratio, seedId); // fill data

        for (int algoId = 0; algoId < NUM_SORT; ++algoId)
        {
#ifdef USE_CUDA
            // Skip if CUDA version doesn't exist
            if (useCuda && methods[algoId] == nullptr) {
                continue;
            }
#endif
            std::vector<int> data_temp = data; // deep copy
            auto start = std::chrono::high_resolution_clock::now();
            methods[algoId](data_temp.data(), size);
            auto end = std::chrono::high_resolution_clock::now();
            double duration = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

            sumY[algoId] += duration;
            sumY2[algoId] += duration * duration;

            if (!isSorted(data_temp))
            {
                notAllSorted[algoId] = true;
                std::cerr << "Algorithm " << names[algoId] << " did not sort correctly!\n";
            }
        }
    }

    // Fill stats
    for (int algoId = 0; algoId < NUM_SORT; ++algoId)
    {
        double mean = sumY[algoId] / NUM_ITERATION;
        double variance = (sumY2[algoId] / NUM_ITERATION - mean * mean) * NUM_ITERATION / (NUM_ITERATION - 1);
        sortStat[algoId] = {algoId, size, ratio, mean, std::sqrt(variance), !notAllSorted[algoId]};
    }
}

void writeStatsToCSV(const std::string &filename,
                     const std::vector<std::vector<std::vector<Stat>>> &allStats,
                     const std::string &dataTypeLabel,
                     std::string* names = sort_name)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing\n";
        return;
    }

    // Write header
    file << "DataType,Size,Ratio,Algorithm,MeanTime(ns),StdDeviation(ns),AllSorted\n";

    // Write data
    for (const auto &sizeGroup : allStats)
    {
        for (const auto &ratioGroup : sizeGroup)
        {
            for (const auto &stat : ratioGroup)
            {
                file << dataTypeLabel << ","
                     << stat.size << ","
                     << stat.ratio << ","
                     << names[stat.sortId] << ","
                     << stat.meanTime << ","
                     << stat.stdDeviation << ","
                     << (stat.areAllSorted ? "true" : "false") << "\n";
            }
        }
    }

    file.close();
    std::cout << "Statistics written to " << filename << "\n";
}

void writeSingleTypeToCSV(const std::string &filename,
                          const std::vector<std::vector<Stat>> &stats,
                          const std::string &dataTypeLabel,
                          std::string* names = sort_name)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing\n";
        return;
    }

    // Write header
    file << "DataType,Size,Algorithm,MeanTime(ns),StdDeviation(ns),AllSorted\n";

    // Write data
    for (const auto &sizeGroup : stats)
    {
        for (const auto &stat : sizeGroup)
        {
            file << dataTypeLabel << ","
                 << stat.size << ","
                 << names[stat.sortId] << ","
                 << stat.meanTime << ","
                 << stat.stdDeviation << ","
                 << (stat.areAllSorted ? "true" : "false") << "\n";
        }
    }

    file.close();
    std::cout << "Statistics written to " << filename << "\n";
}

// Write comparison CSV showing CPU vs CUDA side by side
void writeComparisonCSV(const std::string &filename,
                        const std::vector<std::vector<Stat>> &cpuStats,
                        const std::vector<std::vector<Stat>> &cudaStats,
                        const std::string &dataTypeLabel)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing\n";
        return;
    }

    // Write header
    file << "DataType,Size,Algorithm,CPU_MeanTime(ns),CPU_StdDeviation(ns),CPU_AllSorted,"
         << "CUDA_MeanTime(ns),CUDA_StdDeviation(ns),CUDA_AllSorted,Speedup\n";

    // Write data - match by size and algorithm
    for (size_t sizeIdx = 0; sizeIdx < cpuStats.size() && sizeIdx < cudaStats.size(); ++sizeIdx)
    {
        const auto &cpuSizeGroup = cpuStats[sizeIdx];
        const auto &cudaSizeGroup = cudaStats[sizeIdx];

        for (size_t algoIdx = 0; algoIdx < cpuSizeGroup.size() && algoIdx < cudaSizeGroup.size(); ++algoIdx)
        {
            const auto &cpuStat = cpuSizeGroup[algoIdx];
            const auto &cudaStat = cudaSizeGroup[algoIdx];

            // Only include algorithms that have both CPU and CUDA versions (first 5)
            if (algoIdx < 5)
            {
                double speedup = (cudaStat.meanTime > 0) ? cpuStat.meanTime / cudaStat.meanTime : 0.0;
                
                file << dataTypeLabel << ","
                     << cpuStat.size << ","
                     << sort_name[algoIdx] << ","
                     << cpuStat.meanTime << ","
                     << cpuStat.stdDeviation << ","
                     << (cpuStat.areAllSorted ? "true" : "false") << ","
                     << cudaStat.meanTime << ","
                     << cudaStat.stdDeviation << ","
                     << (cudaStat.areAllSorted ? "true" : "false") << ","
                     << speedup << "\n";
            }
        }
    }

    file.close();
    std::cout << "Comparison statistics written to " << filename << "\n";
}

// Write comparison CSV for ratio-based data (almost sorted, almost inverted)
void writeComparisonCSVWithRatio(const std::string &filename,
                                 const std::vector<std::vector<std::vector<Stat>>> &cpuStats,
                                 const std::vector<std::vector<std::vector<Stat>>> &cudaStats,
                                 const std::string &dataTypeLabel)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing\n";
        return;
    }

    // Write header
    file << "DataType,Size,Ratio,Algorithm,CPU_MeanTime(ns),CPU_StdDeviation(ns),CPU_AllSorted,"
         << "CUDA_MeanTime(ns),CUDA_StdDeviation(ns),CUDA_AllSorted,Speedup\n";

    // Write data - match by size, ratio, and algorithm
    for (size_t sizeIdx = 0; sizeIdx < cpuStats.size() && sizeIdx < cudaStats.size(); ++sizeIdx)
    {
        const auto &cpuSizeGroup = cpuStats[sizeIdx];
        const auto &cudaSizeGroup = cudaStats[sizeIdx];

        for (size_t ratioIdx = 0; ratioIdx < cpuSizeGroup.size() && ratioIdx < cudaSizeGroup.size(); ++ratioIdx)
        {
            const auto &cpuRatioGroup = cpuSizeGroup[ratioIdx];
            const auto &cudaRatioGroup = cudaSizeGroup[ratioIdx];

            for (size_t algoIdx = 0; algoIdx < cpuRatioGroup.size() && algoIdx < cudaRatioGroup.size(); ++algoIdx)
            {
                const auto &cpuStat = cpuRatioGroup[algoIdx];
                const auto &cudaStat = cudaRatioGroup[algoIdx];

                // Only include algorithms that have both CPU and CUDA versions (first 5)
                if (algoIdx < 5)
                {
                    double speedup = (cudaStat.meanTime > 0) ? cpuStat.meanTime / cudaStat.meanTime : 0.0;
                    
                    file << dataTypeLabel << ","
                         << cpuStat.size << ","
                         << cpuStat.ratio << ","
                         << sort_name[algoIdx] << ","
                         << cpuStat.meanTime << ","
                         << cpuStat.stdDeviation << ","
                         << (cpuStat.areAllSorted ? "true" : "false") << ","
                         << cudaStat.meanTime << ","
                         << cudaStat.stdDeviation << ","
                         << (cudaStat.areAllSorted ? "true" : "false") << ","
                         << speedup << "\n";
                }
            }
        }
    }

    file.close();
    std::cout << "Comparison statistics written to " << filename << "\n";
}

int main(int argc, char* argv[])
{
    // Check if comparison mode is requested (runs both CPU and CUDA)
    bool comparisonMode = false;
    bool useCuda = false;
    
    if (argc > 1) {
        if (std::string(argv[1]) == "--compare") {
            comparisonMode = true;
#ifdef USE_CUDA
            std::cout << "Running in comparison mode (CPU vs CUDA)\n";
#else
            std::cout << "Warning: CUDA not compiled, comparison mode unavailable\n";
            comparisonMode = false;
#endif
        } else if (std::string(argv[1]) == "--cuda") {
            useCuda = true;
#ifdef USE_CUDA
            std::cout << "Running in CUDA mode only\n";
#else
            std::cout << "Warning: CUDA not compiled, using CPU versions\n";
            useCuda = false;
#endif
        }
    } else {
        std::cout << "Running in CPU mode (use --cuda for CUDA, --compare for comparison)\n";
    }

    int dataPowerSize = 8;
    std::vector<int> dataSizes(dataPowerSize);
    generateSize(dataSizes.data(), dataPowerSize);

    int dataRatioSize = 6;
    std::vector<double> dataRatios(dataRatioSize);
    generateRatio(dataRatios.data(), dataRatioSize);

    // Storage for CPU results
    std::vector<std::vector<std::vector<Stat>>> cpuSortedStats(dataPowerSize);
    std::vector<std::vector<Stat>> cpuRandomStats(dataPowerSize);
    std::vector<std::vector<std::vector<Stat>>> cpuInvertedStats(dataPowerSize);

    // Storage for CUDA results (if comparison mode)
    std::vector<std::vector<std::vector<Stat>>> cudaSortedStats(dataPowerSize);
    std::vector<std::vector<Stat>> cudaRandomStats(dataPowerSize);
    std::vector<std::vector<std::vector<Stat>>> cudaInvertedStats(dataPowerSize);

    // Run CPU benchmarks
    std::cout << "\n=== Running CPU benchmarks ===\n";
    for (int sizeId = 0; sizeId < dataPowerSize; ++sizeId)
    {
        int size = dataSizes[sizeId];
        std::cout << "Processing size: " << size << " (CPU)\n";

        // Almost sorted
        std::vector<std::vector<Stat>> sortStatSorted(dataRatioSize, std::vector<Stat>(NUM_SORT));
        for (int ratioId = 0; ratioId < dataRatioSize; ++ratioId)
        {
            timeMeasureData(sortStatSorted[ratioId], size, dataRatios[ratioId], 1, false);
        }
        cpuSortedStats[sizeId] = sortStatSorted;

        // Random
        std::vector<Stat> sortStatRandom(NUM_SORT);
        timeMeasureData(sortStatRandom, size, 0.0, 0, false);
        cpuRandomStats[sizeId] = sortStatRandom;

        // Almost inverted
        std::vector<std::vector<Stat>> sortStatInverted(dataRatioSize, std::vector<Stat>(NUM_SORT));
        for (int ratioId = 0; ratioId < dataRatioSize; ++ratioId)
        {
            timeMeasureData(sortStatInverted[ratioId], size, dataRatios[ratioId], 2, false);
        }
        cpuInvertedStats[sizeId] = sortStatInverted;
    }

    // Write CPU results to CSV files
    writeStatsToCSV("results_almost_sorted_cpu.csv", cpuSortedStats, "AlmostSorted", sort_name);
    writeSingleTypeToCSV("results_random_cpu.csv", cpuRandomStats, "Random", sort_name);
    writeStatsToCSV("results_almost_inverted_cpu.csv", cpuInvertedStats, "AlmostInverted", sort_name);

#ifdef USE_CUDA
    // Run CUDA benchmarks if in comparison mode or CUDA-only mode
    if (comparisonMode || useCuda) {
        std::cout << "\n=== Running CUDA benchmarks ===\n";
        for (int sizeId = 0; sizeId < dataPowerSize; ++sizeId)
        {
            int size = dataSizes[sizeId];
            std::cout << "Processing size: " << size << " (CUDA)\n";

            // Almost sorted
            std::vector<std::vector<Stat>> sortStatSorted(dataRatioSize, std::vector<Stat>(NUM_SORT));
            for (int ratioId = 0; ratioId < dataRatioSize; ++ratioId)
            {
                timeMeasureData(sortStatSorted[ratioId], size, dataRatios[ratioId], 1, true);
            }
            cudaSortedStats[sizeId] = sortStatSorted;

            // Random
            std::vector<Stat> sortStatRandom(NUM_SORT);
            timeMeasureData(sortStatRandom, size, 0.0, 0, true);
            cudaRandomStats[sizeId] = sortStatRandom;

            // Almost inverted
            std::vector<std::vector<Stat>> sortStatInverted(dataRatioSize, std::vector<Stat>(NUM_SORT));
            for (int ratioId = 0; ratioId < dataRatioSize; ++ratioId)
            {
                timeMeasureData(sortStatInverted[ratioId], size, dataRatios[ratioId], 2, true);
            }
            cudaInvertedStats[sizeId] = sortStatInverted;
        }

        // Write CUDA results to CSV files
        writeStatsToCSV("results_almost_sorted_cuda.csv", cudaSortedStats, "AlmostSorted", sort_name_cuda);
        writeSingleTypeToCSV("results_random_cuda.csv", cudaRandomStats, "Random", sort_name_cuda);
        writeStatsToCSV("results_almost_inverted_cuda.csv", cudaInvertedStats, "AlmostInverted", sort_name_cuda);
    }

    // Create comparison CSV files if in comparison mode
    if (comparisonMode) {
        std::cout << "\n=== Creating comparison CSV files ===\n";
        writeComparisonCSVWithRatio("comparison_almost_sorted.csv", cpuSortedStats, cudaSortedStats, "AlmostSorted");
        writeComparisonCSV("comparison_random.csv", cpuRandomStats, cudaRandomStats, "Random");
        writeComparisonCSVWithRatio("comparison_almost_inverted.csv", cpuInvertedStats, cudaInvertedStats, "AlmostInverted");
    }
#endif

    std::cout << "\nAll benchmarking completed successfully!\n";
    return 0;
}