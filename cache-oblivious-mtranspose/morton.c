#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

#include "immintrin.h"
#include "transposes.h"

void kernal(double *a, double *b, int N, int offsetRow, int offsetCol) {
    __m256d block1 = _mm256_load_pd(a + (N * 0) + (N * offsetRow) + offsetCol);
    __m256d block2 = _mm256_load_pd(a + (N * 1) + (N * offsetRow) + offsetCol);
    __m256d block3 = _mm256_load_pd(a + (N * 2) + (N * offsetRow) + offsetCol);
    __m256d block4 = _mm256_load_pd(a + (N * 3) + (N * offsetRow) + offsetCol);

    __m256d block1_T = _mm256_permute4x64_pd(block1, 0b11011000);
    __m256d block2_T = _mm256_permute4x64_pd(block2, 0b11011000);
    __m256d block3_T = _mm256_permute4x64_pd(block3, 0b11011000);
    __m256d block4_T = _mm256_permute4x64_pd(block4, 0b11011000);

    _mm256_store_pd(b + (N * 0) + (N * offsetRow) + offsetCol, block1_T);
    _mm256_store_pd(b + (N * 1) + (N * offsetRow) + offsetCol, block3_T);
    _mm256_store_pd(b + (N * 2) + (N * offsetRow) + offsetCol, block2_T);
    _mm256_store_pd(b + (N * 3) + (N * offsetRow) + offsetCol, block4_T);
}

void _morton(double *a, double *b, int N, int halfN, int offsetRow,
             int offsetCol) {
    // DEBUG:
    // printf("morton(a,b,N=%d,halfN=%d,r=%d,c=%d)\n", N, halfN, offsetRow,
    //    offsetCol);
    if (halfN == 4) {
        kernal(a, b, N, offsetRow, offsetCol);
        return;
    }
    halfN = halfN / 2;

    // ULEFT URIGHT BLEFT BRIGHT
    _morton(a, b, N, halfN, offsetRow, offsetCol);
    _morton(a, b, N, halfN, offsetRow, offsetCol + halfN);
    _morton(a, b, N, halfN, offsetRow + halfN, offsetCol);
    _morton(a, b, N, halfN, offsetRow + halfN, offsetCol + halfN);
}

void morton(double *a, double *b, int N) {
    // edge case:
    if (N == 2) {
        __m256d block1 = _mm256_load_pd(a);
        __m256d block1_T = _mm256_permute4x64_pd(block1, 0b11011000);
        _mm256_store_pd(b, block1_T);
        return;
    }
    _morton(a, b, N, N, 0, 0);
}
