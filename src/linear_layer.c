#include "linear_layer.h"

#include "matrix.h"
#include "utils.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

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

    for (size_t i = 0; i < layer->b->size; i++) {
        layer->b->data[i] = (random_float() * 2.0f - 1.0f) * limit * 0.01f;
    }
}

void linear_layer_forward(linear_layer *layer, matrix *output, matrix *input) {
    layer->input = input;
    gemm_helper(output, input, layer->W, false, false);
    add(output, layer->b);
}

// Calculates the gradients for the layer given an input matrix, prediction, and the expected output
void linear_layer_backward(linear_layer *layer, matrix* delta_prev, matrix* delta_next, size_t batch_size) {
    // Update the bias gradient
    for (size_t j = 0; j < delta_next->shape[1]; j++) {
        float sum = 0.0f;
        for (size_t i = 0; i < batch_size; i++) {
            sum += delta_next->data[i * delta_next->stride + j];
            
        }
        layer->db->data[j] = sum / (float) batch_size;
    }

    // Update the weight gradient
    for (size_t i = 0; i < layer->in_features; i++) {
        for (size_t j = 0; j < layer->out_features; j++) {
            layer->dW->data[i * layer->dW->stride + j] = 0.0f;
        }
    }

    for (size_t b = 0; b < batch_size; b++) {

        float *input_row = &layer->input->data[b * layer->input->stride];
        float *delta_row = &delta_next->data[b * delta_next->stride];

        for (size_t i = 0; i < layer->in_features; i++) {
            float in_val = input_row[i];

            for (size_t j = 0; j < layer->out_features; j++) {
                layer->dW->data[i * layer->dW->stride + j] += in_val * delta_row[j];
            }
        }
    }
    float mean_factor = 1.0f / (float) batch_size;
    for (size_t i = 0; i < layer->in_features; i++) {
        for (size_t j = 0; j < layer->out_features; j++) {
            layer->dW->data[i * layer->dW->stride + j] *= mean_factor; 
        }
    }

    for (size_t b = 0; b < batch_size; b++) {
        float *delta_prev_row = &delta_prev->data[b * delta_prev->stride];
        float *delta_row = &delta_next->data[b * delta_next->stride];

        for (size_t i = 0; i < layer->in_features; i++) {
            double s = 0.0;
            for (size_t j = 0; j < layer->out_features; j++) {
                s += (double)delta_row[j] * (double)layer->W->data[i * layer->W->stride + j];
            }
            delta_prev_row[i] = (float)s;
        }
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
