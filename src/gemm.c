#include "gemm.h"

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
) 
{
    for (int i = 0; i < M; ++i) {
        float *c_row = &C[i * ldc];

        // scale C row by beta once
        for (int j = 0; j < N; ++j) c_row[j] *= beta;

        for (int k = 0; k < K; ++k) {
            float a = T_a ? A[k * lda + i] : A[i * lda + k];
            const float *b_col = T_b ? &B[0 + k] : &B[k * ldb]; // adjust indexing per layout
                                                                // unroll j loop or rely on compiler
            for (int j = 0; j < N; ++j) {
                float b = T_b ? B[j * ldb + k] : b_col[j];
                c_row[j] += alpha * a * b;
            }
        }
    }

}
