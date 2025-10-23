#pragma once

#include <stdbool.h>
#include <stdlib.h>

typedef struct {
    float *data;
    size_t shape[2];
    size_t stride;
    size_t size;
} matrix;

matrix *create_matrix(size_t rows, size_t cols);
matrix *create_matrix_with_data(size_t rows, size_t cols, void* data);
matrix *create_matrix_random(size_t rows, size_t cols);
matrix *gemm(matrix *a, matrix *b, bool T_a, bool T_b);
void add(matrix *a, matrix *b);
void softmax(matrix *m);
void relu(matrix *m);
void relu_backwards(float *delta, matrix *activated_output);
void print_matrix(matrix *m);
void free_matrix(matrix *m);
