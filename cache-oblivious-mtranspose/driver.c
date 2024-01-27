#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

#include "immintrin.h"
#include "transposes.h"

#define MAX_FREQ 3.3
#define BASE_FREQ 2.4

void naive(double *a, double *b, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            b[j * N + i] = a[i * N + j];
        }
    }
}

unsigned long long rdtsc() {
    unsigned long long int x;
    unsigned a, d;

    __asm__ volatile("rdtsc" : "=a"(a), "=d"(d));

    return ((unsigned long long)a) | (((unsigned long long)d) << 32);
}

int check(double *a, double *b, int N) {
    int res = 1;
    for (int i = 0; res && i != N * N; ++i) {
        int tmp = (a[i] == b[i]);
        res &= tmp;
    }
    return res;
}

void print(double *a, int N) {
    puts("-----");
    for (int i = 0; i != N; ++i) {
        for (int j = 0; j != N; ++j) {
            printf("%f ", a[i * N + j]);
        }
        puts(" ");
    }
    puts("-----");
}

int main(int argc, char **argv) {
    double *a, *b, *c;

    int startN = 2;
    int endN = 512;

    int N = 16;  // size of matrix
    int runs = 1;

    if (argc < 4) {
        printf(
            "run like: transpose.x <<Number of Runs>> <<Starting Size>> <<End "
            "Size>>\n");
        exit(0);
    }

    runs = atoi(argv[1]);
    startN = atoi(argv[2]);
    endN = atoi(argv[3]);

    // double *t;
    // posix_memalign((void **)&a, 64, sizeof(double) * 4);
    // posix_memalign((void **)&t, 64, sizeof(double) * 4);
    // for (int i = 0; i < 4; i++) {
    //     a[i] = i;
    //     t[i] = 0.0;
    // }
    // __m256d ax = _mm256_load_pd(a);
    // __m256d tx = _mm256_permute4x64_pd(ax, 0b11011000);
    // // 11011000
    // // 00100111
    // _mm256_store_pd(t, tx);
    // puts("-----");
    // for (int i = 0; i < 4; i++) {
    //     printf("%f ", t[i]);
    // }
    // puts("-----");
    // return 0;

    for (int sizes = startN; sizes <= endN; sizes *= 2) {
        int N = sizes;

        unsigned long long t0, t1;

        posix_memalign((void **)&a, 64, sizeof(double) * N * N);
        posix_memalign((void **)&b, 64, sizeof(double) * N * N);
        posix_memalign((void **)&c, 64, sizeof(double) * N * N);

        // initialize data assuming data is row major order
        for (int i = 0; i != N; ++i) {
            for (int j = 0; j != N; ++j) {
                a[i * N + j] = (i + 1) * 1.0 + (j + 1) * 0.001;
                b[i] = 0.0;
            }
        }

        printf("%d\t", sizes);

        unsigned long long sum = 0;
        int res = 0;

        sum = 0;

        for (int r = 0; r != runs; ++r) {
            t0 = rdtsc();
            morton(a, b, N);  // transpose the matrix
            morton(b, c, N);  // transpose the transposed matrix
            t1 = rdtsc();
            sum += (t1 - t0);
        }

        // WARNING! PASS THIS TEST DOES NOT MEAN THE TRANSPOSE ITSELF IS
        // CORRECT!
        res = check(a, c, N);  // is the final matrix is the same?
        printf("%3.3lf\t%d\t", MAX_FREQ / BASE_FREQ * sum / (2.0 * runs), res);

        printf("\n");

        free(a);
        free(b);
        free(c);
    }

    printf("Done\n");

    return 0;
}
