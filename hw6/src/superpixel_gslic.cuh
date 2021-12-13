#ifndef _SUPERPIXEL_SLIC_CUH
#define _SUPERPIXEL_SLIC_CUH

#include <stdint.h>

#include "types.h"

__global__
void superpixel_gslic(
    Pixel_t *img,
    unsigned int width,
    unsigned int height,
    Seed_t *seeds,
    unsigned int n_seeds,
    ClosestSeed_t *distances
);

#endif
