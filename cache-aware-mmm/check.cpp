#include <immintrin.h>
#include <stdlib.h>

#include <cstdlib>
#include <iostream>

void pack(double* pack_a, double* a, int mc, int m, int k) {
    for (int i = 0; i < mc / m; i++) {
        for (int col = 0; col < k; col += 4) {
            for (int row = 0; row < m; row += 6) {
                __m256d a0 = _mm256_loadu_pd(a + (i * m * k) + (row * k) + col);
                __m256d a1 =
                    _mm256_loadu_pd(a + (i * m * k) + (row * k) + col + k);
                __m256d a2 =
                    _mm256_loadu_pd(a + (i * m * k) + (row * k) + col + 2 * k);
                __m256d a3 =
                    _mm256_loadu_pd(a + (i * m * k) + (row * k) + col + 3 * k);
                __m256d a4 =
                    _mm256_loadu_pd(a + (i * m * k) + (row * k) + col + 4 * k);
                __m256d a5 =
                    _mm256_loadu_pd(a + (i * m * k) + (row * k) + col + 5 * k);

                __m256d b0 = _mm256_permute2f128_pd(a0, a1, 0b00100000);
                __m256d b1 = _mm256_permute2f128_pd(a2, a3, 0b00100000);
                __m256d b2 = _mm256_permute2f128_pd(a4, a5, 0b00100000);
                __m256d b3 = _mm256_permute2f128_pd(a0, a1, 0b00110001);
                __m256d b4 = _mm256_permute2f128_pd(a2, a3, 0b00110001);
                __m256d b5 = _mm256_permute2f128_pd(a4, a5, 0b00110001);
                b0 = _mm256_permute4x64_pd(b0, 0b11011000);
                b1 = _mm256_permute4x64_pd(b1, 0b11011000);
                b2 = _mm256_permute4x64_pd(b2, 0b11011000);
                b3 = _mm256_permute4x64_pd(b3, 0b11011000);
                b4 = _mm256_permute4x64_pd(b4, 0b11011000);
                b5 = _mm256_permute4x64_pd(b5, 0b11011000);
                a0 = _mm256_permute2f128_pd(b0, b1, 0b00100000);
                a1 = _mm256_permute2f128_pd(b2, b0, 0b00110000);
                a2 = _mm256_permute2f128_pd(b2, b1, 0b00010011);
                a3 = _mm256_permute2f128_pd(b3, b4, 0b00100000);
                a4 = _mm256_permute2f128_pd(b5, b3, 0b00110000);
                a5 = _mm256_permute2f128_pd(b5, b4, 0b00010011);

                _mm256_storeu_pd(pack_a + (i * m * k) + (col * 6), a0);
                _mm256_storeu_pd(pack_a + (i * m * k) + (col * 6) + 4, a1);
                _mm256_storeu_pd(pack_a + (i * m * k) + (col * 6) + 8, a2);
                _mm256_storeu_pd(pack_a + (i * m * k) + (col * 6) + 12, a3);
                _mm256_storeu_pd(pack_a + (i * m * k) + (col * 6) + 16, a4);
                _mm256_storeu_pd(pack_a + (i * m * k) + (col * 6) + 20, a5);
            }
        }
    }
}

int main() {
    double* pack_A;
    double* A;
    int m = 6;
    int n = 16;
    int mc = 60;

    A = (double*)malloc(mc * n * sizeof(double));
    pack_A = (double*)malloc(mc * n * sizeof(double));

    for (int i = 0; i != m * n; ++i) {
        A[i] = (rand() % 10) * 1.0;
    }

    pack(pack_A, A, mc, 6, n);

    bool correctness = true;
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            correctness &= (pack_A[j * m + i] == A[i * n + j]);
        }
    }

    if (correctness) {
        std::cout << "Correct" << std::endl;
    } else {
        std::cout << "Incorrect" << std::endl;
    }

    std::cout << "Original Matrix" << std::endl;
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            std::cout << A[i * n + j] << "\t";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Packed for L2 cache" << std::endl;
    for (int i = 0; i != m; ++i) {
        for (int j = 0; j != n; ++j) {
            std::cout << pack_A[j * m + i] << "\t";
        }
        std::cout << std::endl;
    }

    free(A);
    free(pack_A);
    return 0;
}
