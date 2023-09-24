#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <x86intrin.h>

#define MAX_FREQ 3.3
#define BASE_FREQ 2.4

#define NUM_INST 1000
#define NUM_CHAINS 15

static __inline__ unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

#define SIMD_FMADD_ASM(src1, src2, src3)                            \
    __asm__ __volatile__("vfmadd132pd %[xsrc1], %[xsrc2], %[xsrc3]" \
                         : [xsrc3] "+x"(src3)                       \
                         : [xsrc2] "x"(src2), [xsrc1] "x"(src1));

#define SIMD_FMADD_ASM_CHAINS()       \
    SIMD_FMADD_ASM(_0x, _0x, _1x);    \
    SIMD_FMADD_ASM(_2x, _2x, _3x);    \
    SIMD_FMADD_ASM(_4x, _4x, _5x);    \
    SIMD_FMADD_ASM(_6x, _6x, _7x);    \
    SIMD_FMADD_ASM(_8x, _8x, _9x);    \
    SIMD_FMADD_ASM(_10x, _10x, _11x); \
    SIMD_FMADD_ASM(_12x, _12x, _13x); \
    SIMD_FMADD_ASM(_14x, _14x, _15x); \
    SIMD_FMADD_ASM(_16x, _16x, _17x); \
    SIMD_FMADD_ASM(_18x, _18x, _19x); \
    SIMD_FMADD_ASM(_20x, _20x, _21x); \
    SIMD_FMADD_ASM(_22x, _22x, _23x); \
    SIMD_FMADD_ASM(_24x, _24x, _25x); \
    SIMD_FMADD_ASM(_26x, _26x, _27x); \
    SIMD_FMADD_ASM(_28x, _28x, _29x); \
    // SIMD_FMADD_ASM(_30x, _30x, _31x); \
    // SIMD_FMADD_ASM(_32x, _32x, _33x);

#define SIMD_FMADD_ASM_CHAINS_10() \
    SIMD_FMADD_ASM_CHAINS();       \
    SIMD_FMADD_ASM_CHAINS();       \
    SIMD_FMADD_ASM_CHAINS();       \
    SIMD_FMADD_ASM_CHAINS();       \
    SIMD_FMADD_ASM_CHAINS();       \
    SIMD_FMADD_ASM_CHAINS();       \
    SIMD_FMADD_ASM_CHAINS();       \
    SIMD_FMADD_ASM_CHAINS();       \
    SIMD_FMADD_ASM_CHAINS();

#define SIMD_FMADD_ASM_CHAINS_100() \
    SIMD_FMADD_ASM_CHAINS_10();     \
    SIMD_FMADD_ASM_CHAINS_10();     \
    SIMD_FMADD_ASM_CHAINS_10();     \
    SIMD_FMADD_ASM_CHAINS_10();     \
    SIMD_FMADD_ASM_CHAINS_10();     \
    SIMD_FMADD_ASM_CHAINS_10();     \
    SIMD_FMADD_ASM_CHAINS_10();     \
    SIMD_FMADD_ASM_CHAINS_10();     \
    SIMD_FMADD_ASM_CHAINS_10();

#define SIMD_FMADD_ASM_CHAINS_1000() \
    SIMD_FMADD_ASM_CHAINS_100();     \
    SIMD_FMADD_ASM_CHAINS_100();     \
    SIMD_FMADD_ASM_CHAINS_100();     \
    SIMD_FMADD_ASM_CHAINS_100();     \
    SIMD_FMADD_ASM_CHAINS_100();     \
    SIMD_FMADD_ASM_CHAINS_100();     \
    SIMD_FMADD_ASM_CHAINS_100();     \
    SIMD_FMADD_ASM_CHAINS_100();     \
    SIMD_FMADD_ASM_CHAINS_100();

int main(int argc, char **argv) {
    puts("------ THROUGHPUT: SIMD FMA -------");
    int runs = atoi(argv[1]);

    unsigned long long st;
    unsigned long long et;
    unsigned long long sum = 0;

    __m256d _0x = {1.0, 1.0, 1.0, 1.0};
    __m256d _1x = {1.0, 1.0, 1.0, 1.0};
    __m256d _2x = {1.0, 1.0, 1.0, 1.0};
    __m256d _3x = {1.0, 1.0, 1.0, 1.0};
    __m256d _4x = {1.0, 1.0, 1.0, 1.0};
    __m256d _5x = {1.0, 1.0, 1.0, 1.0};
    __m256d _6x = {1.0, 1.0, 1.0, 1.0};
    __m256d _7x = {1.0, 1.0, 1.0, 1.0};
    __m256d _8x = {1.0, 1.0, 1.0, 1.0};
    __m256d _9x = {1.0, 1.0, 1.0, 1.0};
    __m256d _10x = {1.0, 1.0, 1.0, 1.0};
    __m256d _11x = {1.0, 1.0, 1.0, 1.0};
    __m256d _12x = {1.0, 1.0, 1.0, 1.0};
    __m256d _13x = {1.0, 1.0, 1.0, 1.0};
    __m256d _14x = {1.0, 1.0, 1.0, 1.0};
    __m256d _15x = {1.0, 1.0, 1.0, 1.0};
    __m256d _16x = {1.0, 1.0, 1.0, 1.0};
    __m256d _17x = {1.0, 1.0, 1.0, 1.0};
    __m256d _18x = {1.0, 1.0, 1.0, 1.0};
    __m256d _19x = {1.0, 1.0, 1.0, 1.0};
    __m256d _20x = {1.0, 1.0, 1.0, 1.0};
    __m256d _21x = {1.0, 1.0, 1.0, 1.0};
    __m256d _22x = {1.0, 1.0, 1.0, 1.0};
    __m256d _23x = {1.0, 1.0, 1.0, 1.0};
    __m256d _24x = {1.0, 1.0, 1.0, 1.0};
    __m256d _25x = {1.0, 1.0, 1.0, 1.0};
    __m256d _26x = {1.0, 1.0, 1.0, 1.0};
    __m256d _27x = {1.0, 1.0, 1.0, 1.0};
    __m256d _28x = {1.0, 1.0, 1.0, 1.0};
    __m256d _29x = {1.0, 1.0, 1.0, 1.0};
    __m256d _30x = {1.0, 1.0, 1.0, 1.0};
    __m256d _31x = {1.0, 1.0, 1.0, 1.0};
    __m256d _32x = {1.0, 1.0, 1.0, 1.0};
    __m256d _33x = {1.0, 1.0, 1.0, 1.0};

    for (int j = 0; j < runs; j++) {
        st = rdtsc();
        SIMD_FMADD_ASM_CHAINS_1000();
        et = rdtsc();
        sum += (et - st);
    }

    printf("RDTSC Base Cycles Taken for SIMD_FMADD: %llu\n\r", sum);
    printf("TURBO Cycles Taken for SIMD_FMADD: %lf\n\r",
           sum * ((double)MAX_FREQ) / BASE_FREQ);
    printf(
        "Throughput : %lf (chains: %d)\n\r",
        (((double)NUM_INST * runs * NUM_CHAINS) / (sum * MAX_FREQ / BASE_FREQ)),
        NUM_CHAINS);
    puts("");

    return 0;
}
