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

#define NUM_SORT 5
#define NUM_ITERATION 100

// Must match the same order
typedef void (*sort_ptr)(int[], int);
sort_ptr sort_methods[] = {bubble_sort_optimize1,
                           merge_sort,
                           quick_sort,
                           couting_sort,
                           radix_sort};

std::string sort_name[] = {"basic_bubble_sort",
                           "basic_merge_sort",
                           "basic_quick_sort",
                           "basic_counting_sort",
                           "basic_radix_sort"};

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

void timeMeasureData(std::vector<Stat> &sortStat, int size, double ratio, int dataType)
{
    std::vector<double> sumY(NUM_SORT, 0.0);
    std::vector<double> sumY2(NUM_SORT, 0.0);
    std::vector<bool> notAllSorted(NUM_SORT, false);

    for (int seedId = 0; seedId < NUM_ITERATION; ++seedId)
    {
        std::vector<int> data(size);
        GenerateData(data.data(), size, dataType, ratio, seedId); // fill data

        for (int algoId = 0; algoId < NUM_SORT; ++algoId)
        {
            std::vector<int> data_temp = data; // deep copy
            auto start = std::chrono::high_resolution_clock::now();
            sort_methods[algoId](data_temp.data(), size);
            auto end = std::chrono::high_resolution_clock::now();
            double duration = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

            sumY[algoId] += duration;
            sumY2[algoId] += duration * duration;

            if (!isSorted(data_temp))
            {
                notAllSorted[algoId] = true;
                std::cerr << "Algorithm " << sort_name[algoId] << " did not sort correctly!\n";
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
                     const std::string &dataTypeLabel)
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
                     << sort_name[stat.sortId] << ","
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
                          const std::string &dataTypeLabel)
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
                 << sort_name[stat.sortId] << ","
                 << stat.meanTime << ","
                 << stat.stdDeviation << ","
                 << (stat.areAllSorted ? "true" : "false") << "\n";
        }
    }

    file.close();
    std::cout << "Statistics written to " << filename << "\n";
}

int main()
{
    int dataPowerSize = 6;
    std::vector<int> dataSizes(dataPowerSize);
    generateSize(dataSizes.data(), dataPowerSize);

    int dataRatioSize = 6;
    std::vector<double> dataRatios(dataRatioSize);
    generateRatio(dataRatios.data(), dataRatioSize);

    // Storage for all results
    std::vector<std::vector<std::vector<Stat>>> allSortedStats(dataPowerSize);
    std::vector<std::vector<Stat>> allRandomStats(dataPowerSize);
    std::vector<std::vector<std::vector<Stat>>> allInvertedStats(dataPowerSize);

    // Benchmark varying sizes
    for (int sizeId = 0; sizeId < dataPowerSize; ++sizeId)
    {
        int size = dataSizes[sizeId];
        std::cout << "Processing size: " << size << "\n";

        // Almost sorted
        std::vector<std::vector<Stat>> sortStatSorted(dataRatioSize, std::vector<Stat>(NUM_SORT));
        for (int ratioId = 0; ratioId < dataRatioSize; ++ratioId)
        {
            timeMeasureData(sortStatSorted[ratioId], size, dataRatios[ratioId], 1);
        }
        allSortedStats[sizeId] = sortStatSorted;

        // Random
        std::vector<Stat> sortStatRandom(NUM_SORT);
        timeMeasureData(sortStatRandom, size, 0.0, 0);
        allRandomStats[sizeId] = sortStatRandom;

        // Almost inverted
        std::vector<std::vector<Stat>> sortStatInverted(dataRatioSize, std::vector<Stat>(NUM_SORT));
        for (int ratioId = 0; ratioId < dataRatioSize; ++ratioId)
        {
            timeMeasureData(sortStatInverted[ratioId], size, dataRatios[ratioId], 2);
        }
        allInvertedStats[sizeId] = sortStatInverted;
    }

    // Write all results to CSV files
    writeStatsToCSV("results_almost_sorted.csv", allSortedStats, "AlmostSorted");
    writeSingleTypeToCSV("results_random.csv", allRandomStats, "Random");
    writeStatsToCSV("results_almost_inverted.csv", allInvertedStats, "AlmostInverted");

    // Write combined results
    std::ofstream combinedFile("results_all_combined.csv");
    if (combinedFile.is_open())
    {
        combinedFile << "DataType,Size,Ratio,Algorithm,MeanTime(ns),StdDeviation(ns),AllSorted\n";

        // Write sorted data
        for (const auto &sizeGroup : allSortedStats)
        {
            for (const auto &ratioGroup : sizeGroup)
            {
                for (const auto &stat : ratioGroup)
                {
                    combinedFile << "AlmostSorted," << stat.size << "," << stat.ratio << ","
                                 << sort_name[stat.sortId] << "," << stat.meanTime << ","
                                 << stat.stdDeviation << "," << (stat.areAllSorted ? "true" : "false") << "\n";
                }
            }
        }

        // Write random data
        for (const auto &sizeGroup : allRandomStats)
        {
            for (const auto &stat : sizeGroup)
            {
                combinedFile << "Random," << stat.size << ",0.0,"
                             << sort_name[stat.sortId] << "," << stat.meanTime << ","
                             << stat.stdDeviation << "," << (stat.areAllSorted ? "true" : "false") << "\n";
            }
        }

        // Write inverted data
        for (const auto &sizeGroup : allInvertedStats)
        {
            for (const auto &ratioGroup : sizeGroup)
            {
                for (const auto &stat : ratioGroup)
                {
                    combinedFile << "AlmostInverted," << stat.size << "," << stat.ratio << ","
                                 << sort_name[stat.sortId] << "," << stat.meanTime << ","
                                 << stat.stdDeviation << "," << (stat.areAllSorted ? "true" : "false") << "\n";
                }
            }
        }

        combinedFile.close();
        std::cout << "Combined statistics written to results_all_combined.csv\n";
    }

    std::cout << "All benchmarking completed successfully!\n";
    return 0;
}