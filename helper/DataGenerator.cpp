#include <time.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>

template <class T>
void HoanVi(T &a, T &b)
{
    T x = a;
    a = b;
    b = x;
}

//-------------------------------------------------

// Hàm phát sinh mảng dữ liệu ngẫu nhiên
void GenerateRandomData(int a[], int n, std::mt19937 &rng)
{
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int i = 0; i < n; i++)
        a[i] = dist(rng);
}

void TransformQuasiDataExact(int a[], int n, int c, bool inverted, std::mt19937 &rng)
{
    // Start with sorted array
    for (int i = 0; i < n; ++i)
        a[i] = i;

    // Optional: shuffle order of indices to move for randomness
    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i)
        indices[i] = i;
    std::shuffle(indices.begin(), indices.end(), rng);

    int remaining = c;

    for (int idx = 0; idx < n && remaining > 0; ++idx)
    {
        int i = indices[idx];

        // Max moves this element can make without exceeding array bounds
        int max_move = std::min(remaining, n - 1 - i);

        if (max_move > 0)
        {
            int j = i + max_move;

            // Rotate a[i] to position j
            std::rotate(a + i, a + j, a + j + 1);

            remaining -= max_move;
        }
    }

    if (inverted)
    {
        std::reverse(a, a + n);
    }
}

void generateSize(int a[], int size)
{
    int s = 1;
    for (int i = 0; i < size; ++i)
    {
        a[i] = s;
        s *= 2;
    }
}

void generateRatio(double a[], int size)
{
    int s = 0;
    for (int i = 0; i < size; ++i)
    {
        a[i] = s;
        s += 1.0 / (2 * size);
    }
}

#define RANDOM_DATA 0
#define SORTED_DATA 1
#define REVERSE_DATA 2
#define PERMUTATION 5

void GenerateData(int a[], int n, int typ, double c, size_t seed)
{
    std::mt19937 rng(seed);

    int max_inv = n * (n - 1) / 2;
    int num_inversion = static_cast<int>(std::round(c * max_inv));

    switch (typ)
    {
    case RANDOM_DATA:
        GenerateRandomData(a, n, rng);
        break;

    case SORTED_DATA:
        TransformQuasiDataExact(a, n, num_inversion, false, rng);
        break;

    case REVERSE_DATA:
        TransformQuasiDataExact(a, n, num_inversion, true, rng);
        break;

    default:
        GenerateRandomData(a, n, rng);
    }
}