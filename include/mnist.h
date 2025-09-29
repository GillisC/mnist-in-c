#pragma once

#include <stdint.h>

#define MNIST_LABEL_MAGIC_NUM 0x00000801
#define MNIST_IMAGE_MAGIC_NUM 0x00000803
#define MNIST_IMG_WIDTH 28
#define MNIST_IMG_HEIGHT 28
#define MNIST_IMAGE_SIZE MNIST_IMG_WIDTH * MNIST_IMG_HEIGHT
#define MNIST_LABELS 10

// Used when we load the header instantly
typedef struct {
    uint32_t magic_number;
    uint32_t num_labels;
} __attribute__((packed)) mnist_label_file_header;

typedef struct {
    uint32_t magic_number;
    uint32_t num_images;
    uint32_t rows;
    uint32_t cols;
} __attribute__((packed)) mnist_image_file_header;

typedef struct {
    uint8_t pixels[MNIST_IMAGE_SIZE];
} __attribute__((packed)) mnist_image;

typedef struct {
    mnist_image *images;
    uint8_t *labels;
    uint32_t size;
    
} mnist_dataset;


mnist_dataset *load_mnist_dataset(const char *mnist_label_file, const char *mnist_image_file);
void free_mnist_dataset(mnist_dataset* dataset);
void print_mnist_image(mnist_image *image);
