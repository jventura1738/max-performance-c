// Justin's macros

#ifndef JVMACROS_H
#define JVMACROS_H

/* ------------------------------
 * MULTIPLICATION MACROS SECTION
 * These macros are to avoid the
 * use of for loops. Good for
 * constant kernals
 * ------------------------------
 */

#define MULT(dest, src)                           \
    __asm__ __volatile__("imul %[rsrc], %[rdest]" \
                         : [rdest] "+r"(dest)     \
                         : [rsrc] "r"(src));

#define MULT10(dest, src) \
    MULT(dest, src)       \
    MULT(dest, src)       \
    MULT(dest, src)       \
    MULT(dest, src)       \
    MULT(dest, src)       \
    MULT(dest, src)       \
    MULT(dest, src)       \
    MULT(dest, src)       \
    MULT(dest, src)       \
    MULT(dest, src)

#define MULT100(dest, src) \
    MULT10(dest, src)      \
    MULT10(dest, src)      \
    MULT10(dest, src)      \
    MULT10(dest, src)      \
    MULT10(dest, src)      \
    MULT10(dest, src)      \
    MULT10(dest, src)      \
    MULT10(dest, src)      \
    MULT10(dest, src)      \
    MULT10(dest, src)

#define MULT600(dest, src) \
    MULT100(dest, src)     \
    MULT100(dest, src)     \
    MULT100(dest, src)     \
    MULT100(dest, src)     \
    MULT100(dest, src)     \
    MULT100(dest, src)

#define MULT1000(dest, src) \
    MULT600(dest, src)      \
    MULT100(dest, src)      \
    MULT100(dest, src)      \
    MULT100(dest, src)      \
    MULT100(dest, src)

/* ------------------------------
 * SIMD ADD MACROS SECTION
 * These macros are to avoid the
 * use of for loops. Good for
 * constant kernals
 * ------------------------------
 */

#define SIMD_ADD(dest, src)     \
    ax = _mm256_load_pd(src);   \
    cx = _mm256_add_pd(ax, ax); \
    _mm256_store_pd(dest, cx);

// This is for length 32 vector:
#define SIMD_ADD32(dest, src)    \
    ax = _mm256_load_pd(a);      \
    cx = _mm256_add_pd(ax, ax);  \
    _mm256_store_pd(c, cx);      \
    ax = _mm256_load_pd(a + 4);  \
    cx = _mm256_add_pd(ax, ax);  \
    _mm256_store_pd(c + 4, cx);  \
    ax = _mm256_load_pd(a + 8);  \
    cx = _mm256_add_pd(ax, ax);  \
    _mm256_store_pd(c + 8, cx);  \
    ax = _mm256_load_pd(a + 12); \
    cx = _mm256_add_pd(ax, ax);  \
    _mm256_store_pd(c + 12, cx); \
    ax = _mm256_load_pd(a + 16); \
    cx = _mm256_add_pd(ax, ax);  \
    _mm256_store_pd(c + 16, cx); \
    ax = _mm256_load_pd(a + 20); \
    cx = _mm256_add_pd(ax, ax);  \
    _mm256_store_pd(c + 20, cx); \
    ax = _mm256_load_pd(a + 24); \
    cx = _mm256_add_pd(ax, ax);  \
    _mm256_store_pd(c + 24, cx); \
    ax = _mm256_load_pd(a + 28); \
    cx = _mm256_add_pd(ax, ax);  \
    _mm256_store_pd(c + 28, cx);

// ***** HELPERS *****
#define PRINT(src, len)             \
    puts("");                       \
    for (int i = 0; i < len; i++) { \
        printf("%f ", src[i]);      \
    }                               \
    puts("");

#endif