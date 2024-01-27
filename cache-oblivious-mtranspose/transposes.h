#include <immintrin.h>
#include <x86intrin.h>

#include "immintrin.h"

void kernal(double *a, double *b, int N, int offsetRow, int offsetCol);

void _morton(double *a, double *b, int N, int halfN, int offsetRow,
             int offsetCol);

void morton(double *a, double *b, int N);
