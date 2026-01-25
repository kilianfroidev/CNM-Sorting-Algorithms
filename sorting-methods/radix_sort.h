#ifndef RADIX_SORT_HEADER
#define RADIX_SORT_HEADER

#include "../helper/utils.h"

// Iterative radix sort to avoid stack overflow
void radix_sort(int a[], int n) {
    if (n <= 1) return;
    
    // Find maximum value to determine number of bits
    int maxVal = a[0];
    for (int i = 1; i < n; i++) {
        if (a[i] > maxVal) maxVal = a[i];
    }
    
    // Calculate number of bits needed
    int numBits = 0;
    int temp = maxVal;
    while (temp > 0) {
        temp >>= 1;
        numBits++;
    }
    if (numBits == 0) numBits = 1;
    
    // Temporary arrays for sorting
    int* tempArr = new int[n];
    int* input = a;
    int* output = tempArr;
    
    // Process each bit from least significant to most significant
    for (int bit = 0; bit < numBits; bit++) {
        int zeroCount = 0;
        
        // Count zeros (elements with 0-bit at current position)
        for (int i = 0; i < n; i++) {
            if (((input[i] >> bit) & 1) == 0) {
                zeroCount++;
            }
        }
        
        // Scatter elements: zeros first, then ones
        int zeroPos = 0;
        int onePos = zeroCount;
        for (int i = 0; i < n; i++) {
            if (((input[i] >> bit) & 1) == 0) {
                output[zeroPos++] = input[i];
            } else {
                output[onePos++] = input[i];
            }
        }
        
        // Swap input and output for next iteration
        int* temp = input;
        input = output;
        output = temp;
    }
    
    // Copy result back to original array if needed
    if (input != a) {
        for (int i = 0; i < n; i++) {
            a[i] = input[i];
        }
    }
    
    delete[] tempArr;
}

#endif
