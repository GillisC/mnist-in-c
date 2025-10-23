#include "operations.h"


size_t max(float *data, size_t length) {
    size_t max_index = 0;
    float max_val = data[0];

    for (size_t i = 1; i < length; i++) {
        if (data[i] > max_val) {
            max_index = i;
            max_val = data[i];
        }
    }
    return max_index;
}

