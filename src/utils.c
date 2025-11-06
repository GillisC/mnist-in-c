#include "utils.h"
#include <stdlib.h>
#include <time.h>


float random_float() {
    return (float) rand() / (float) RAND_MAX;
}

double get_delta_time(clock_t start, clock_t end) {
    return ( (double) (end - start) ) / CLOCKS_PER_SEC;
}
