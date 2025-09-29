#include "mnist.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>

uint32_t map_uint32(uint32_t in)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return (
        ((in & 0xFF000000) >> 24) |
        ((in & 0x00FF0000) >>  8) |
        ((in & 0x0000FF00) <<  8) |
        ((in & 0x000000FF) << 24)
    );
#else
    return in;
#endif
}



uint8_t *load_mnist_labels(const char *mnist_labels_path, uint32_t *number_of_labels) {
    FILE *stream;
    mnist_label_file_header header;
    uint8_t *labels;

    stream = fopen(mnist_labels_path, "rb");

    if (NULL == stream) {
        fprintf(stderr, "Could not open file: %s\n", mnist_labels_path);
    }
    
    if (1 != fread(&header, sizeof(mnist_label_file_header), 1, stream)) {
        fprintf(stderr, "Could not read label file header from: %s\n", mnist_labels_path);
        fclose(stream);
        return NULL;
    }

    header.magic_number = map_uint32(header.magic_number);
    header.num_labels = map_uint32(header.num_labels);

    if (header.magic_number != MNIST_LABEL_MAGIC_NUM) {
        fprintf(stderr, "Invalid header, magic number for mnist labels is not correct");
        fclose(stream);
        return NULL;
    }

    *number_of_labels = header.num_labels;
    labels = malloc(*number_of_labels * sizeof(uint8_t));

    if (NULL == labels) {
        fprintf(stderr, "Could not allocate memory for %d labels\n", header.num_labels);
        fclose(stream);
        return NULL;
    }

    if (1 != fread(labels, sizeof(uint8_t) * header.num_labels, 1, stream)) {
        fprintf(stderr, "Could not read labels from file\n");
        fclose(stream);
        return NULL;
    }

    return labels;
}

mnist_image *load_mnist_images(const char *mnist_images_path, uint32_t *number_of_images) {
    FILE *stream;
    mnist_image_file_header header;
    mnist_image *images;

    stream = fopen(mnist_images_path, "rb");

    if (NULL == stream) {
        fprintf(stderr, "Could not open file: %s\n", mnist_images_path);
        return NULL;
    }
    
    if (1 != fread(&header, sizeof(mnist_image_file_header), 1, stream)) {
        fprintf(stderr, "Could not read image file header from: %s\n", mnist_images_path);
        fclose(stream);
        return NULL;
    }

    header.magic_number = map_uint32(header.magic_number);
    header.num_images = map_uint32(header.num_images);
    header.rows = map_uint32(header.rows);
    header.cols = map_uint32(header.cols);

    printf("number of images defined header: %d\n", header.num_images);
    printf("number of rows: %d\n", header.rows);
    printf("number of cols: %d\n", header.cols);

    if (header.magic_number != MNIST_IMAGE_MAGIC_NUM) {
        fprintf(stderr, "Invalid header, magic number for a MNIST image is not correct\n");
        fclose(stream);
        return NULL;
    }

    *number_of_images = header.num_images;
    size_t allocation_size = *number_of_images * sizeof(mnist_image);
    images = malloc(allocation_size);

    if (images == NULL) {
        fprintf(stderr, "Could not allocate memory for %d images\n", header.num_images);
        fclose(stream);
        return NULL;
    }

    if (*number_of_images != fread(images, sizeof(mnist_image), *number_of_images, stream)) {
        fprintf(stderr, "Could not read images from file\n");
        free(images);
        fclose(stream);
        return NULL;
    }

    fclose(stream);

    return images;
}

mnist_dataset *load_mnist_dataset(const char *mnist_label_file, const char *mnist_image_file) {
    uint32_t num_labels, num_images;
    uint8_t *labels = load_mnist_labels(mnist_label_file, &num_labels);
    mnist_image *images = load_mnist_images(mnist_image_file, &num_images);
    for (uint8_t i = 0; i < 10; i++)
    {
        printf("label: %d = %d\n", i, labels[i]);    
    }
    print_mnist_image(&images[0]);
    return NULL;
}


void print_mnist_image(mnist_image *image) {
    float pixel;
    for (size_t i = 0; i < MNIST_IMG_HEIGHT; i++) {
        for (size_t j = 0; j < MNIST_IMG_WIDTH; j++) {
            pixel = image->pixels[i * MNIST_IMG_WIDTH + j] / 255.0;
            printf("%.1f ", pixel); 
        }   
        printf("\n");
    }   
}
