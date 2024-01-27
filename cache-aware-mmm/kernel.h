#include <immintrin.h>

inline void kernel(int k, double* restrict a, double* restrict b,
                   double* restrict c) {
    __m256d ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7;
    __m256d ymm8, ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15;

    double* bptr = b;
    double* aptr = a;

    int k_iter = k / 4;
    int k_left = k % 4;

    ymm0 = _mm256_setzero_pd();
    ymm1 = _mm256_setzero_pd();
    ymm2 = _mm256_setzero_pd();
    ymm3 = _mm256_setzero_pd();
    ymm4 = _mm256_setzero_pd();
    ymm5 = _mm256_setzero_pd();
    ymm6 = _mm256_setzero_pd();
    ymm7 = _mm256_setzero_pd();
    ymm8 = _mm256_setzero_pd();
    ymm9 = _mm256_setzero_pd();
    ymm10 = _mm256_setzero_pd();
    ymm11 = _mm256_setzero_pd();

    for (int p = 0; p != k_iter; ++p) {
        ymm15 = _mm256_load_pd(bptr);
        ymm14 = _mm256_load_pd(bptr + 4);

        ymm13 = _mm256_broadcast_sd(aptr + 0);
        ymm0 = _mm256_fmadd_pd(ymm13, ymm15, ymm0);
        ymm1 = _mm256_fmadd_pd(ymm13, ymm14, ymm1);

        ymm12 = _mm256_broadcast_sd(aptr + 1);
        ymm2 = _mm256_fmadd_pd(ymm12, ymm15, ymm2);
        ymm3 = _mm256_fmadd_pd(ymm12, ymm14, ymm3);

        ymm13 = _mm256_broadcast_sd(aptr + 2);
        ymm4 = _mm256_fmadd_pd(ymm13, ymm15, ymm4);
        ymm5 = _mm256_fmadd_pd(ymm13, ymm14, ymm5);

        ymm12 = _mm256_broadcast_sd(aptr + 3);
        ymm6 = _mm256_fmadd_pd(ymm12, ymm15, ymm6);
        ymm7 = _mm256_fmadd_pd(ymm12, ymm14, ymm7);

        ymm13 = _mm256_broadcast_sd(aptr + 4);
        ymm8 = _mm256_fmadd_pd(ymm13, ymm15, ymm8);
        ymm9 = _mm256_fmadd_pd(ymm13, ymm14, ymm9);

        ymm12 = _mm256_broadcast_sd(aptr + 5);
        ymm10 = _mm256_fmadd_pd(ymm12, ymm15, ymm10);
        ymm11 = _mm256_fmadd_pd(ymm12, ymm14, ymm11);

        aptr += 6;
        bptr += 8;

        ymm15 = _mm256_load_pd(bptr);
        ymm14 = _mm256_load_pd(bptr + 4);

        ymm13 = _mm256_broadcast_sd(aptr + 0);
        ymm0 = _mm256_fmadd_pd(ymm13, ymm15, ymm0);
        ymm1 = _mm256_fmadd_pd(ymm13, ymm14, ymm1);

        ymm12 = _mm256_broadcast_sd(aptr + 1);
        ymm2 = _mm256_fmadd_pd(ymm12, ymm15, ymm2);
        ymm3 = _mm256_fmadd_pd(ymm12, ymm14, ymm3);

        ymm13 = _mm256_broadcast_sd(aptr + 2);
        ymm4 = _mm256_fmadd_pd(ymm13, ymm15, ymm4);
        ymm5 = _mm256_fmadd_pd(ymm13, ymm14, ymm5);

        ymm12 = _mm256_broadcast_sd(aptr + 3);
        ymm6 = _mm256_fmadd_pd(ymm12, ymm15, ymm6);
        ymm7 = _mm256_fmadd_pd(ymm12, ymm14, ymm7);

        ymm13 = _mm256_broadcast_sd(aptr + 4);
        ymm8 = _mm256_fmadd_pd(ymm13, ymm15, ymm8);
        ymm9 = _mm256_fmadd_pd(ymm13, ymm14, ymm9);

        ymm12 = _mm256_broadcast_sd(aptr + 5);
        ymm10 = _mm256_fmadd_pd(ymm12, ymm15, ymm10);
        ymm11 = _mm256_fmadd_pd(ymm12, ymm14, ymm11);

        aptr += 6;
        bptr += 8;

        ymm15 = _mm256_load_pd(bptr);
        ymm14 = _mm256_load_pd(bptr + 4);

        ymm13 = _mm256_broadcast_sd(aptr + 0);
        ymm0 = _mm256_fmadd_pd(ymm13, ymm15, ymm0);
        ymm1 = _mm256_fmadd_pd(ymm13, ymm14, ymm1);

        ymm12 = _mm256_broadcast_sd(aptr + 1);
        ymm2 = _mm256_fmadd_pd(ymm12, ymm15, ymm2);
        ymm3 = _mm256_fmadd_pd(ymm12, ymm14, ymm3);

        ymm13 = _mm256_broadcast_sd(aptr + 2);
        ymm4 = _mm256_fmadd_pd(ymm13, ymm15, ymm4);
        ymm5 = _mm256_fmadd_pd(ymm13, ymm14, ymm5);

        ymm12 = _mm256_broadcast_sd(aptr + 3);
        ymm6 = _mm256_fmadd_pd(ymm12, ymm15, ymm6);
        ymm7 = _mm256_fmadd_pd(ymm12, ymm14, ymm7);

        ymm13 = _mm256_broadcast_sd(aptr + 4);
        ymm8 = _mm256_fmadd_pd(ymm13, ymm15, ymm8);
        ymm9 = _mm256_fmadd_pd(ymm13, ymm14, ymm9);

        ymm12 = _mm256_broadcast_sd(aptr + 5);
        ymm10 = _mm256_fmadd_pd(ymm12, ymm15, ymm10);
        ymm11 = _mm256_fmadd_pd(ymm12, ymm14, ymm11);

        aptr += 6;
        bptr += 8;

        ymm15 = _mm256_load_pd(bptr);
        ymm14 = _mm256_load_pd(bptr + 4);

        ymm13 = _mm256_broadcast_sd(aptr + 0);
        ymm0 = _mm256_fmadd_pd(ymm13, ymm15, ymm0);
        ymm1 = _mm256_fmadd_pd(ymm13, ymm14, ymm1);

        ymm12 = _mm256_broadcast_sd(aptr + 1);
        ymm2 = _mm256_fmadd_pd(ymm12, ymm15, ymm2);
        ymm3 = _mm256_fmadd_pd(ymm12, ymm14, ymm3);

        ymm13 = _mm256_broadcast_sd(aptr + 2);
        ymm4 = _mm256_fmadd_pd(ymm13, ymm15, ymm4);
        ymm5 = _mm256_fmadd_pd(ymm13, ymm14, ymm5);

        ymm12 = _mm256_broadcast_sd(aptr + 3);
        ymm6 = _mm256_fmadd_pd(ymm12, ymm15, ymm6);
        ymm7 = _mm256_fmadd_pd(ymm12, ymm14, ymm7);

        ymm13 = _mm256_broadcast_sd(aptr + 4);
        ymm8 = _mm256_fmadd_pd(ymm13, ymm15, ymm8);
        ymm9 = _mm256_fmadd_pd(ymm13, ymm14, ymm9);

        ymm12 = _mm256_broadcast_sd(aptr + 5);
        ymm10 = _mm256_fmadd_pd(ymm12, ymm15, ymm10);
        ymm11 = _mm256_fmadd_pd(ymm12, ymm14, ymm11);

        aptr += 6;
        bptr += 8;
    }

    for (int p = 0; p != k_left; ++p) {
        ymm15 = _mm256_load_pd(bptr);
        ymm14 = _mm256_load_pd(bptr + 4);

        ymm13 = _mm256_broadcast_sd(aptr + 0);
        ymm0 = _mm256_fmadd_pd(ymm13, ymm15, ymm0);
        ymm1 = _mm256_fmadd_pd(ymm13, ymm14, ymm1);

        ymm12 = _mm256_broadcast_sd(aptr + 1);
        ymm2 = _mm256_fmadd_pd(ymm12, ymm15, ymm2);
        ymm3 = _mm256_fmadd_pd(ymm12, ymm14, ymm3);

        ymm13 = _mm256_broadcast_sd(aptr + 2);
        ymm4 = _mm256_fmadd_pd(ymm13, ymm15, ymm4);
        ymm5 = _mm256_fmadd_pd(ymm13, ymm14, ymm5);

        ymm12 = _mm256_broadcast_sd(aptr + 3);
        ymm6 = _mm256_fmadd_pd(ymm12, ymm15, ymm6);
        ymm7 = _mm256_fmadd_pd(ymm12, ymm14, ymm7);

        ymm13 = _mm256_broadcast_sd(aptr + 4);
        ymm8 = _mm256_fmadd_pd(ymm13, ymm15, ymm8);
        ymm9 = _mm256_fmadd_pd(ymm13, ymm14, ymm9);

        ymm12 = _mm256_broadcast_sd(aptr + 5);
        ymm10 = _mm256_fmadd_pd(ymm12, ymm15, ymm10);
        ymm11 = _mm256_fmadd_pd(ymm12, ymm14, ymm11);

        aptr += 6;
        bptr += 8;
    }

    double *c0, *c1, *c2;
    c0 = c;
    c1 = c + 16, c2 = c + 32;

    ymm15 = _mm256_load_pd(c0 + 0);
    ymm14 = _mm256_load_pd(c0 + 4);
    ymm0 = _mm256_add_pd(ymm0, ymm15);
    ymm1 = _mm256_add_pd(ymm1, ymm14);
    _mm256_store_pd(c0 + 0, ymm0);
    _mm256_store_pd(c0 + 4, ymm1);

    ymm15 = _mm256_load_pd(c0 + 8);
    ymm14 = _mm256_load_pd(c0 + 12);
    ymm2 = _mm256_add_pd(ymm2, ymm15);
    ymm3 = _mm256_add_pd(ymm3, ymm14);
    _mm256_store_pd(c0 + 8, ymm2);
    _mm256_store_pd(c0 + 12, ymm3);

    ymm15 = _mm256_load_pd(c1 + 0);
    ymm14 = _mm256_load_pd(c1 + 4);
    ymm4 = _mm256_add_pd(ymm4, ymm15);
    ymm5 = _mm256_add_pd(ymm5, ymm14);
    _mm256_store_pd(c1 + 0, ymm4);
    _mm256_store_pd(c1 + 4, ymm5);

    ymm15 = _mm256_load_pd(c1 + 8);
    ymm14 = _mm256_load_pd(c1 + 12);
    ymm6 = _mm256_add_pd(ymm6, ymm15);
    ymm7 = _mm256_add_pd(ymm7, ymm14);
    _mm256_store_pd(c1 + 8, ymm6);
    _mm256_store_pd(c1 + 12, ymm7);

    ymm15 = _mm256_load_pd(c2 + 0);
    ymm14 = _mm256_load_pd(c2 + 4);
    ymm8 = _mm256_add_pd(ymm8, ymm15);
    ymm9 = _mm256_add_pd(ymm9, ymm14);
    _mm256_store_pd(c2 + 0, ymm8);
    _mm256_store_pd(c2 + 4, ymm9);

    ymm15 = _mm256_load_pd(c2 + 8);
    ymm14 = _mm256_load_pd(c2 + 12);
    ymm10 = _mm256_add_pd(ymm10, ymm15);
    ymm11 = _mm256_add_pd(ymm11, ymm14);
    _mm256_store_pd(c2 + 8, ymm10);
    _mm256_store_pd(c2 + 12, ymm11);
}
