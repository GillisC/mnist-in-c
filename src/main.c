#include "matrix.h"
#include "mnist.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "operations.h"
#include "linear_layer.h"

#define PIXEL_SCALE(x) (((float) (x)) / 255.0f)

#define BATCH_SIZE 32
#define EPOCHS 10


void init_dataset_indices(size_t *indices, size_t size) {
    for (size_t i = 0; i < size; i++) {
        indices[i] = i;
    }
}

// Fisher-Yates shuffle
void shuffle_dataset_indices(size_t *indices, size_t size) {
    for (size_t i = size - 1; i > 0; i--) {
        size_t j = (size_t) (rand() % (i + 1));
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

void report_stats(size_t curr_epoch, double elapsed_time, double elapsed_time_per_image, float acc, float loss) {
    printf("=Stats==============================================\n");
    printf("Progress: %zu / %zu\n", curr_epoch + 1, (size_t) EPOCHS);
    printf("Average Accuracy: %.1f%%\n", acc * 100);
    printf("Average Loss    : %.2f\n", loss);
    printf("Elapsed time: %.2f seconds\n", elapsed_time);
    printf("Avg image proccessing time: %.4f ms\n", elapsed_time_per_image);
    printf("====================================================\n\n");
}



int main() {
    srand(time(NULL));
    // input
    matrix *image_matrix = create_matrix(BATCH_SIZE, 784);

    linear_layer *l1 = create_linear_layer(784, 256);
    init_linear_layer(l1);

    linear_layer *l2 = create_linear_layer(256, 128);
    init_linear_layer(l2);

    linear_layer *l3 = create_linear_layer(128, 10);
    init_linear_layer(l3);

    mnist_dataset *dataset = load_mnist_dataset("data/train-labels.idx1-ubyte", "data/train-images.idx3-ubyte");

    float lr = 0.01f;

    matrix *out1 = create_matrix(BATCH_SIZE, l1->out_features);
    matrix *out2 = create_matrix(BATCH_SIZE, l2->out_features);
    matrix *out3 = create_matrix(BATCH_SIZE, l3->out_features);

    matrix *delta_0 = create_matrix(BATCH_SIZE, l1->in_features);
    matrix *delta_1 = create_matrix(BATCH_SIZE, l1->out_features);
    matrix *delta_2 = create_matrix(BATCH_SIZE, l2->out_features);
    matrix *delta_3 = create_matrix(BATCH_SIZE, l3->out_features);

    int *labels = malloc(sizeof(int) * BATCH_SIZE);
    
    // Prepare the dataset (once)
    size_t total_samples = dataset->size;
    size_t *dataset_indices = calloc(total_samples, sizeof(size_t));
    size_t steps_per_epoch = total_samples / BATCH_SIZE;

    init_dataset_indices(dataset_indices, dataset->size);
    shuffle_dataset_indices(dataset_indices, dataset->size);

    clock_t start, end;
    // training loop
    for (size_t epoch = 0; epoch < (size_t)EPOCHS; epoch++) {
        start = clock();

        // shuffle per-epoch
        shuffle_dataset_indices(dataset_indices, total_samples);

        float total_loss = 0.0f;
        int correct_count = 0;

        for (size_t step = 0; step < steps_per_epoch; step++) {

            // load a batch using contiguous shuffled indices
            size_t base = step * BATCH_SIZE;
            for (size_t b = 0; b < BATCH_SIZE; b++) {
                size_t dataset_index = dataset_indices[base + b];
                labels[b] = dataset->labels[dataset_index];

                // Copy pixels from dataset into the input matrix
                for (int p = 0; p < MNIST_IMAGE_SIZE; p++) {
                    image_matrix->data[b * image_matrix->stride + p] =
                        PIXEL_SCALE(dataset->images[dataset_index].pixels[p]);
                }
            }

            // Forward
            linear_layer_forward(l1, out1, image_matrix);
            relu(out1);
            linear_layer_forward(l2, out2, out1);
            relu(out2);
            linear_layer_forward(l3, out3, out2);
            softmax(out3);

            // Build delta_2 (softmax CE)
            for (size_t b = 0; b < BATCH_SIZE; b++) {
                size_t correct_label = (size_t) labels[b];
                for (size_t k = 0; k < l3->out_features; k++) {
                    size_t idx = b * delta_3->stride + k;
                    size_t out_idx = b * out3->stride + k;
                    delta_3->data[idx] = out3->data[out_idx] - (k == correct_label ? 1.0f : 0.0f);
                }
            }

            linear_layer_backward(l3, delta_2, delta_3, (size_t)BATCH_SIZE);
            relu_backwards(delta_2, out2);

            linear_layer_backward(l2, delta_1, delta_2, (size_t)BATCH_SIZE);
            relu_backwards(delta_1, out1);

            linear_layer_backward(l1, delta_0, delta_1, (size_t)BATCH_SIZE);

            linear_layer_update(l3, lr);
            linear_layer_update(l2, lr);
            linear_layer_update(l1, lr);

            // accumulate metrics per sample
            for (size_t b = 0; b < BATCH_SIZE; b++) {
                size_t correct_label = (size_t) labels[b];
                float *row = &out3->data[b * out3->stride];
                size_t pred = max(row, out3->shape[1]);
                if (pred == correct_label) correct_count++;
                total_loss += -logf(row[correct_label] + 1e-12f);
            }
        }
        end = clock();

        // Report statistics each epoch
        size_t total_samples = steps_per_epoch * BATCH_SIZE;
        float avg_acc = (float) correct_count / (float) total_samples;
        float avg_loss = total_loss / (float) total_samples;
        double delta_time = ((double)(end - start)) / CLOCKS_PER_SEC;
        double delta_time_per_image = (delta_time * 1000) / total_samples; 

        report_stats(epoch, delta_time, delta_time_per_image, avg_acc, avg_loss);
    }
    return 0;
}
