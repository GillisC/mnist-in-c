#include "matrix.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "utils.h"
#include "gemm.h"

size_t padded_cols(size_t cols, size_t element_size, size_t align_bytes)
{
    size_t bytes = cols * element_size;
    size_t padded_bytes =
        ((bytes + align_bytes - 1) / align_bytes) * align_bytes;
    return padded_bytes / element_size;
}


matrix *create_matrix(size_t rows, size_t cols) {
    matrix *m = malloc(sizeof(matrix));

    size_t alignment_factor = 32;
    size_t total_bytes;

    if (cols * sizeof(float) % alignment_factor != 0) {
        total_bytes = rows * cols * sizeof(float);
        m->stride = cols;
    }
    else {
        size_t cols_padded = padded_cols(cols, sizeof(float), alignment_factor);
        total_bytes = rows * cols_padded * sizeof(float);
        m->stride = cols_padded;
    }

    m->data = (float*) (aligned_alloc(alignment_factor, total_bytes));
    if (NULL == m->data) {
        fprintf(stderr, "Could not allocate memory for matrix");
    }

    for (size_t i = 0; i < rows * cols; i++) m->data[i] = 0.0f;

    m->shape[0] = rows;
    m->shape[1] = cols;
    m->size = rows * cols;
    return m;
}


matrix *create_matrix_with_data(size_t rows, size_t cols, void* data) {
    matrix *m = create_matrix(rows, cols);
    memcpy(m->data, data, rows * cols * sizeof(float));

    return m;
}

matrix *create_matrix_random(size_t rows, size_t cols) {
    matrix *m = create_matrix(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            m->data[i * m->stride + j] = random_float();
        }
    }
    return m;
}

void gemm_helper(matrix* c, matrix *a, matrix *b, bool T_a, bool T_b) {
    gemmf(
        a->data, 
        a->stride,
        b->data, 
        b->stride, 
        c->data, 
        c->stride, 
        a->shape[0],
        b->shape[1], 
        a->shape[1], 
        1.0f, 
        0.0f, 
        T_a, 
        T_b
    );
}

void add(matrix *a, matrix *b) {
    if (a->shape[1] != b->shape[1]) {
        fprintf(stderr, "add: invalid dimensions of the matrices: a: [%zu, %zu], b: [%zu, %zu]",
                a->shape[0], a->shape[1], b->shape[0], b->shape[1]);
        exit(1);
    }

    for (size_t i = 0; i < a->shape[0]; i++) {
        for (size_t j = 0; j < a->shape[1]; j++) {
                a->data[i * a->stride + j] += b->data[j];
        }
    }
}

void softmax_row(float *data, size_t length) {

    float max_val = data[0];
    for (size_t i = 1; i < length; i++) {
        if (data[i] > max_val) max_val = data[i];
    }

    float sum = 0.0f;
    for (size_t i = 0; i < length; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    
    for (size_t i = 0; i < length; i++) {
        data[i] /= sum;
    }
}

void softmax(matrix *m) {
    size_t rows = m->shape[0];
    for (size_t row = 0; row < rows; ++row) {
        softmax_row(m->data + (row * m->stride) , m->shape[1]);
    }
}

void relu(matrix *m) {
    for (size_t i = 0; i < m->shape[0]; i++) {
        for (size_t j = 0; j < m->shape[1]; j++) {
            if (m->data[i * m->stride + j] <= 0.0f) {
                m->data[i * m->stride + j] = 0.0f;
            }
        }
    }
}

void relu_backwards(matrix *delta, matrix *activated_output) {
    for (size_t i = 0; i < activated_output->size; i++) {
        if (activated_output->data[i] <= 0.0f)
            delta->data[i] = 0.0f;
    }
}

void print_matrix(matrix *m) {
    size_t rows = m->shape[0];
    size_t cols = m->shape[1];

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            printf("%0.2f ", m->data[i * m->stride + j]);
        }
        printf("\n");
    }
}

void free_matrix(matrix *m) {
    free(m->data);
    free(m);
}
