#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <x86intrin.h>

#include "../jvmacros.h"

#define MAX_FREQ 3.3
#define BASE_FREQ 2.4

#define NUM_INST 8

static __inline__ unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

int main(int argc, char **argv) {
    puts("------ LATENCY: SIMD ADD -------");
    int runs = atoi(argv[1]);

    unsigned long long st;
    unsigned long long et;
    unsigned long long sum = 0;

    double *a = (double *)aligned_alloc(32, NUM_INST * 4 * sizeof(double));
    double *c = (double *)aligned_alloc(32, NUM_INST * 4 * sizeof(double));

    for (int i = 0; i < NUM_INST; i++) {
        a[i] = (double)i;
        c[i] = 0.0;
    }

    __m256d ax, cx;

    for (int j = 0; j < runs; j++) {
        st = rdtsc();
        SIMD_ADD32(c, a);
        et = rdtsc();
        sum += (et - st);
    }

    free(a);
    free(c);

    printf("RDTSC Base Cycles Taken for SIMD_ADD: %llu\n\r", sum);
    printf("Latency: %lf\n\r", MAX_FREQ / BASE_FREQ * sum / (NUM_INST * runs));
    puts("");

    return 0;
}
