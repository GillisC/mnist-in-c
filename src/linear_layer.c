#include "linear_layer.h"

#include "utils.h"
#include <math.h>
#include <string.h>

linear_layer *create_linear_layer(size_t in_features, size_t out_features) {
    linear_layer *l = malloc(sizeof(linear_layer));

    l->in_features = in_features;
    l->out_features = out_features;

    l->W = create_matrix(in_features, out_features);
    l->b = create_matrix(1, out_features);

    // we keep the gradients as fields in the struct to avoid a bunch of allocations
    // during the training step
    l->dW = create_matrix(in_features, out_features);
    l->db = create_matrix(1, out_features);

    return l;
}

void init_linear_layer(linear_layer *layer) {
    // set the weights to a small random values
    float limit = sqrtf(6.0 / (layer->in_features * layer->out_features));
    for (size_t i = 0; i < layer->W->size; i++) {
        layer->W->data[i] = (random_float() * 2.0f - 1.0f) * limit;
    }
    memset(layer->b->data, 0, layer->b->size * sizeof(float));
}

matrix* linear_layer_forward(linear_layer *layer, matrix *input) {
    layer->input = input;
    matrix *prediction = gemm(input, layer->W, false, false);
    add(prediction, layer->b);
    return prediction;
}

// Calculates the gradients for the layer given an input matrix, prediction, and the expected output
void linear_layer_backward(linear_layer *layer, float* delta_prev, float* delta_next) {
    memcpy(layer->db->data, delta_next, layer->out_features * sizeof(float));

    for (size_t i = 0; i < layer->in_features; i++) {
        for (size_t j = 0; j < layer->out_features; j++) {
            layer->dW->data[i * layer->dW->stride + j] = layer->input->data[i] * delta_next[j];
        }
    }

    // delta_prev = delta_next * W^T
    for (size_t i = 0; i < layer->in_features; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < layer->out_features; j++) {
            sum += delta_next[j] * layer->W->data[i * layer->W->stride + j];
        }
        delta_prev[i] = sum;
    }
}


void linear_layer_update(linear_layer *layer, float learning_rate) {
    // update the weights
    for (size_t i = 0; i < layer->in_features; i++) {
        for (size_t j = 0; j < layer->out_features; j++) {
            float w_grad = layer->dW->data[i * layer->dW->stride +j];
            layer->W->data[i * layer->W->stride + j] -= w_grad * learning_rate;
        }
    }
    // update the bias
    for (size_t i = 0; i < layer->out_features; i++) {
        float b_grad = layer->db->data[i];
        layer->b->data[i] -= b_grad * learning_rate;
    }
}
