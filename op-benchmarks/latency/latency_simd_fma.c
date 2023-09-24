#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <x86intrin.h>

#include "../jvmacros.h"

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4

#define NUM_INST 100

static __inline__ unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

#define SIMD_FMADD_ASM(src1, src2, src3)                            \
    __asm__ __volatile__("vfmadd132pd %[xsrc1], %[xsrc2], %[xsrc3]" \
                         : [xsrc3] "+x"(src3)                       \
                         : [xsrc2] "x"(src2), [xsrc1] "x"(src1));

#define SIMD_FMADD_ASM10(src1, src2, src3) \
    SIMD_FMADD_ASM(src1, src2, src3)       \
    SIMD_FMADD_ASM(src1, src2, src3)       \
    SIMD_FMADD_ASM(src1, src2, src3)       \
    SIMD_FMADD_ASM(src1, src2, src3)       \
    SIMD_FMADD_ASM(src1, src2, src3)       \
    SIMD_FMADD_ASM(src1, src2, src3)       \
    SIMD_FMADD_ASM(src1, src2, src3)       \
    SIMD_FMADD_ASM(src1, src2, src3)       \
    SIMD_FMADD_ASM(src1, src2, src3)       \
    SIMD_FMADD_ASM(src1, src2, src3)

#define SIMD_FMADD_ASM100(src1, src2, src3) \
    SIMD_FMADD_ASM10(src1, src2, src3)      \
    SIMD_FMADD_ASM10(src1, src2, src3)      \
    SIMD_FMADD_ASM10(src1, src2, src3)      \
    SIMD_FMADD_ASM10(src1, src2, src3)      \
    SIMD_FMADD_ASM10(src1, src2, src3)      \
    SIMD_FMADD_ASM10(src1, src2, src3)      \
    SIMD_FMADD_ASM10(src1, src2, src3)      \
    SIMD_FMADD_ASM10(src1, src2, src3)      \
    SIMD_FMADD_ASM10(src1, src2, src3)      \
    SIMD_FMADD_ASM10(src1, src2, src3)

#define SIMD_FMADD_ASM1000(src1, src2, src3) \
    SIMD_FMADD_ASM100(src1, src2, src3)      \
    SIMD_FMADD_ASM100(src1, src2, src3)      \
    SIMD_FMADD_ASM100(src1, src2, src3)      \
    SIMD_FMADD_ASM100(src1, src2, src3)      \
    SIMD_FMADD_ASM100(src1, src2, src3)      \
    SIMD_FMADD_ASM100(src1, src2, src3)      \
    SIMD_FMADD_ASM100(src1, src2, src3)      \
    SIMD_FMADD_ASM100(src1, src2, src3)      \
    SIMD_FMADD_ASM100(src1, src2, src3)      \
    SIMD_FMADD_ASM100(src1, src2, src3)

// #define SIMD_FMADD(dest, src) \
//     ax = _mm256_load_pd(a);   \
//     bx = _mm256_load_pd(b);   \
//     cx = _mm256_load_pd(c);   \
//     _mm256_fmadd_pd(cx, bx, ax);

int main(int argc, char **argv) {
    puts("------- LATENCY: SIMD FMA -------");
    int runs = atoi(argv[1]);

    unsigned long long st;
    unsigned long long et;
    unsigned long long sum = 0;

    double *a = (double *)aligned_alloc(32, NUM_INST * sizeof(double));
    double *b = (double *)aligned_alloc(32, NUM_INST * sizeof(double));
    double *c = (double *)aligned_alloc(32, NUM_INST * sizeof(double));

    for (int i = 0; i < 4; i++) {
        a[i] = (double)1.0;
        b[i] = 2.0;
        c[i] = 3.0;
    }

    for (int j = 0; j < runs; j++) {
        __m256d ax = _mm256_load_pd(a);
        __m256d bx = _mm256_load_pd(b);
        __m256d cx = _mm256_load_pd(c);

        st = rdtsc();
        // SIMD_FMADD(cx, ax);
        SIMD_FMADD_ASM100(ax, bx, cx);
        et = rdtsc();

        _mm256_store_pd(a, ax);
        _mm256_store_pd(b, bx);
        _mm256_store_pd(c, cx);
        sum += (et - st);
    }

    // PRINT(a, 4);
    // PRINT(b, 4);
    // PRINT(c, 4);

    free(a);
    free(b);
    free(c);

    printf("RDTSC Base Cycles Taken for SIMD_FMA: %llu\n\r", sum);
    printf("Latency: %lf\n\r", MAX_FREQ / BASE_FREQ * sum / (NUM_INST * runs));
    puts("");

    return 0;
}
