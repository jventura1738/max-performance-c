#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "immintrin.h"
#include "kernel.h"
#include "pack.h"

#define RUNS 100
#define MAX_FREQ 3.3
#define BASE_FREQ 2.4

static __inline__ unsigned long long rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

int main() {
    double *a, *pack_a;
    double *b;
    double *c;

    unsigned long long t0, t1, sum;

    int mc = 60;  // mc is the number of rows of A
    int k = 256;  // k is the number of columns of A

    int m = 6;  // m is the number of rows of the kernel
    int n = 8;  // n is the number of columns of the kernel

    posix_memalign((void **)&a, 64, mc * k * sizeof(double));
    posix_memalign((void **)&pack_a, 64, mc * k * sizeof(double));
    posix_memalign((void **)&b, 64, n * k * sizeof(double));
    posix_memalign((void **)&c, 64, mc * n * sizeof(double));

    for (int i = 0; i != k * mc; ++i) {
        a[i] = ((double)rand()) / ((double)RAND_MAX);
    }
    for (int i = 0; i != k * n; ++i) {
        b[i] = ((double)rand()) / ((double)RAND_MAX);
    }
    for (int i = 0; i != mc * n; ++i) {
        c[i] = 0.0;
    }

    printf("%d\t %d\t %d\t", mc, n, k);

    pack(pack_a, a, mc, m, k);

    sum = 0;

    for (int runs = 0; runs != RUNS; ++runs) {
        t0 = rdtsc();
        for (int i = 0; i != mc / m; ++i) {
            kernel(k, pack_a + i * m * k, b, c + i * n * m);
        }
        t1 = rdtsc();
        sum += (t1 - t0);
    }

    printf(" %lf\n", (2.0 * mc * n * k) /
                         ((double)(MAX_FREQ / BASE_FREQ * sum / (1.0 * RUNS))));

    free(a);
    free(pack_a);
    free(b);
    free(c);

    return 0;
}
