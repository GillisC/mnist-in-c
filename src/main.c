#include "matrix.h"
#include "mnist.h"
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "operations.h"
#include "linear_layer.h"

#define PIXEL_SCALE(x) (((float) (x)) / 255.0f)


int main()
{
    float lr = 0.001f;
    linear_layer *l1 = create_linear_layer(784, 128);
    init_linear_layer(l1);

    linear_layer *l2 = create_linear_layer(128, 10);
    init_linear_layer(l2);

    mnist_dataset *dataset = load_mnist_dataset("data/train-labels.idx1-ubyte", "data/train-images.idx3-ubyte");
    matrix *image_matrix = create_matrix(1, 784);

    float total_loss = 0.0f;
    int correct_count = 0;

    int N = (int) dataset->size;

    for (int i = 0; i < N; i++) {

        size_t correct = (size_t) (dataset->labels[i]);

        // 1. Flatten the image into buffer
        for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
            image_matrix->data[j] = PIXEL_SCALE(dataset->images[i].pixels[j]);
        }

        // 2. Forward pass
        matrix *out1 = linear_layer_forward(l1, image_matrix);
        relu(out1);
        matrix *out2 = linear_layer_forward(l2, out1);
        softmax(out2);

        // 3. Backpropagation
        float out2_delta_next[10];
        for (size_t k = 0; k < 10; k++) {
            if (k == correct) {
                out2_delta_next[k] = out2->data[k] - 1.0f; 
            }
            else {
                out2_delta_next[k] = out2->data[k];
            }
        }
        float out2_delta_prev[l2->in_features];
        float out1_delta_prev[l1->in_features];

        linear_layer_backward(l2, out2_delta_prev, out2_delta_next);
        relu_backwards(out2_delta_prev, out1);
        linear_layer_backward(l1, out1_delta_prev, out2_delta_prev);

        // 4. Update the layer based on the gradients
        linear_layer_update(l1, lr);
        linear_layer_update(l2, lr);

        if (max(out2->data, 10) == correct) {
            correct_count++;
        }
        total_loss += -log(out2->data[correct]);

        if ((i + 1) % 1000 == 0) {
            float average_loss = total_loss / 1000;
            float accuracy = (float) correct_count / 1000;

            printf("%d/%d: Accuracy: %f | Avg loss: %f\n", i+1, N, accuracy, average_loss);
            correct_count = 0;
            total_loss = 0.0f;
        }
    }

    //print_matrix(lin_layer->W);

    return 0;
}
