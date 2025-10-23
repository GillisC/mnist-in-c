#pragma once

#include <stdlib.h>

size_t max(float *data, size_t length);
float cross_entropy_loss(size_t true_label, float* pred, size_t length);
