#include "mnist.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>



float random_float() {
    return (float) rand() / (float) RAND_MAX;
}

int main()
{
    load_mnist_dataset("data/t10k-labels.idx1-ubyte", "data/t10k-images.idx3-ubyte");
    return 0;
}
