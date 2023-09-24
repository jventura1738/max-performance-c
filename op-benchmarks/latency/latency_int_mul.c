#include <immintrin.h>
#include <stddef.h>  // for size_t
#include <stdio.h>
#include <string.h>
#include <x86intrin.h>

#include "../jvmacros.h"

// For normalization purposes:
#define MAX_FREQ 3.3
#define BASE_FREQ 2.4

// Instruction count to test
#define NUM_INST 1000

static __inline__ unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

int main(int argc, char **argv) {
    puts("------ LATENCY: INT MUL ------");
    int runs = atoi(argv[1]);

    unsigned long long st;
    unsigned long long et;
    unsigned long long sum = 0;

    unsigned long long a = 1;
    unsigned long long b = 2;

    for (int j = 0; j < runs; j++) {
        st = rdtsc();
        MULT1000(a, b);
        et = rdtsc();
        sum += (et - st);
    }

    printf("RDTSC Base Cycles Taken for MULT: %llu\n\r", sum);
    printf("Latency: %lf\n\r", MAX_FREQ / BASE_FREQ * sum / (NUM_INST * runs));
    puts("");

    return 0;
}
