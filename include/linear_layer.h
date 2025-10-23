#pragma once

#include <stdlib.h>
#include <matrix.h>


typedef struct {
    size_t in_features;
    size_t out_features;

    matrix *W;
    matrix *b;

    matrix *dW;
    matrix *db;

    matrix *input;
} linear_layer;

typedef struct {
    float* delta;
    size_t length;

} delta_vector;

linear_layer *create_linear_layer(size_t in_features, size_t out_features);
void init_linear_layer(linear_layer *layer);

void linear_layer_forward(linear_layer *layer, matrix *output, matrix *input);

// The size of delta_prev is layer->in_features
// The size of delta_next is layer->out_features
void linear_layer_backward(linear_layer *layer, matrix* delta_prev, matrix* delta_next, size_t batch_size);

void linear_layer_update(linear_layer *layer, float learning_rate);
