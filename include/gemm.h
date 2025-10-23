#pragma once

#include <stdbool.h>

void gemmf(
    const float *A,         // matrix A
    const int lda,
    const float *B,
    const int ldb,
    float *C,
    const int ldc,
    const int M,            // rows of A and C
    const int N,            // columns of B and C
    const int K,            // columns of A & rows of B
    const float alpha,      // scalar multiplier for A*B
    const float beta,       // scalar multiplier for C
    const bool T_a,
    const bool T_b
);
