#include "gemm.h"

void gemmf(
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc,
    int M, int N, int K,
    float alpha, float beta,
    bool T_a, bool T_b
)
{
    for (int i = 0; i < M; ++i) {
        float *c_row = &C[i * ldc];

        // scale C row by beta
        for (int j = 0; j < N; ++j)
            c_row[j] *= beta;

        for (int k = 0; k < K; ++k) {
            float a = T_a ? A[k * lda + i] : A[i * lda + k];

            for (int j = 0; j < N; ++j) {
                float b = T_b ? B[j * ldb + k] : B[k * ldb + j];
                c_row[j] += alpha * a * b;
            }
        }
    }
}
