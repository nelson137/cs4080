#ifndef _SUPERPIXEL_SLIC_CUH
#define _SUPERPIXEL_SLIC_CUH

#include <stdint.h>

typedef struct {
    unsigned char l;
    unsigned char a;
    unsigned char b;
} Pixel_t;

typedef struct {
    double x;
    double y;
    double l;
    double a;
    double b;
} Seed_t;

typedef struct {
    double dist;
    unsigned int label;
} ClosestSeed_t;

__global__
void superpixel_gslic(
    Pixel_t *img,
    unsigned int width,
    unsigned int height,
    Seed_t *seeds,
    unsigned int n_seeds,
    ClosestSeed_t *distances,
    Seed_t *seed_sigmas,
    double *seed_pixel_counts
);

#endif
