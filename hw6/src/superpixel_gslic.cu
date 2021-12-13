#include <float.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "superpixel_gslic.cuh"

#define ITERATIONS 10

#define SUB_KERNEL_BS 32

////////////////////////////////////////////////////////////////////////////////
// Init Seeds
////////////////////////////////////////////////////////////////////////////////

__global__
void superpixel_gslic__init_seeds(
    Pixel_t *img,
    unsigned int width,
    unsigned int height,
    Seed_t *seeds,
    unsigned int n_seeds)
{
    const unsigned int seed_strip_size = sqrt((double) n_seeds);
    const unsigned int seed_side_len = width / seed_strip_size;

    const unsigned int x_off = seed_side_len / 2;
    const unsigned int y_off = seed_side_len / 2;

    const unsigned int err = width - seed_side_len * seed_strip_size;
    const double err_per_strip = ((double) err / (double) seed_strip_size);

    unsigned int seed_i = 0;

    for (unsigned int y = 0; y < seed_strip_size; ++y)
    {
        unsigned int y_err = y * err_per_strip;
        unsigned int Y = y*seed_side_len + y_off + y_err;
        if (Y >= height)
            continue;

        for (unsigned int x = 0; x < seed_strip_size; ++x)
        {
            unsigned int x_err = x * err_per_strip;
            unsigned int X = x*seed_side_len + x_off + x_err;
            if (X >= width)
                continue;

            Pixel_t pixel = img[Y * width + X];
            seeds[seed_i++] = {
                .x = (double) X,
                .y = (double) Y,
                .l = (double) pixel.l,
                .a = (double) pixel.a,
                .b = (double) pixel.b,
            };
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Iterations
////////////////////////////////////////////////////////////////////////////////

__device__
bool pixel_within_seed_2S(
    double x,
    double y,
    unsigned int width,
    unsigned int height,
    Seed_t seed,
    unsigned int seed_side_len)
{
    double y_min = max(0.0,            y - (double)seed_side_len);
    double y_max = min((double)height, y + (double)seed_side_len);
    double x_min = max(0.0,            x - (double)seed_side_len);
    double x_max = min((double)width,  x + (double)seed_side_len);
    return y_min <= seed.y && seed.y < y_max
        && x_min <= seed.x && seed.x < x_max;
}

__global__
void superpixel_gslic__iter__dist(
    Pixel_t *img,
    unsigned int width,
    unsigned int height,
    Seed_t *seeds,
    unsigned int n_seeds,
    unsigned int seed_side_len,
    ClosestSeed_t *distances)
{
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < width * height)
    {
        const double inverse_weight = (double) seed_side_len;

        Pixel_t pixel = img[i];
        double X = (double) (i % width);
        // Use int division to truncate, then cast to double for later calculations
        double Y = (double) (i / width);
        double L = (double) pixel.l;
        double A = (double) pixel.a;
        double B = (double) pixel.b;

        ClosestSeed_t closest_seed = { .dist = DBL_MAX, .label = UINT_MAX };

        for (unsigned int s = 0; s < n_seeds; ++s)
        {
            Seed_t seed = seeds[s];
            if (pixel_within_seed_2S(X, Y, width, height, seed, seed_side_len))
            {
                double dist = (X - seed.x) * (X - seed.x) +
                              (Y - seed.y) * (Y - seed.y);
                dist *= inverse_weight;
                dist += (L - seed.l) * (L - seed.l) +
                        (A - seed.a) * (A - seed.a) +
                        (B - seed.b) * (B - seed.b);
                if (dist < closest_seed.dist)
                    closest_seed = { .dist = dist, .label = s };
            }
        }

        distances[i] = closest_seed;
    }
}

__global__
void superpixel_gslic__iter__recalc_seeds(
    Pixel_t *img,
    unsigned int width,
    unsigned int height,
    unsigned int n_seeds,
    Seed_t *seeds,
    Seed_t *seed_sigmas,
    double *seed_pixel_counts,
    ClosestSeed_t *distances)
{
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n_seeds)
    {
        seed_sigmas[i] = { .x = 0.0, .y = 0.0, .l = 0.0, .a = 0.0, .b = 0.0 };
        seed_pixel_counts[i] = 0.0;

        unsigned int pixel_i = 0;
        for (unsigned int y = 0; y < height; ++y)
        {
            for (unsigned int x = 0; x < width; ++x)
            {
                if (distances[pixel_i].label == i)
                {
                    seed_pixel_counts[i] += 1.0;
                    Seed_t *sigma = seed_sigmas + i;
                    Pixel_t pixel = img[pixel_i];
                    sigma->x += x;
                    sigma->y += y;
                    sigma->l += (double) pixel.l;
                    sigma->a += (double) pixel.a;
                    sigma->b += (double) pixel.b;
                }
                ++pixel_i;
            }
        }

        double s_size = seed_pixel_counts[i];
        if (s_size <= 0.0)
            s_size = 1.0;

        Seed_t sigma = seed_sigmas[i];
        seeds[i] = {
            .x = sigma.x / s_size,
            .y = sigma.y / s_size,
            .l = sigma.l / s_size,
            .a = sigma.a / s_size,
            .b = sigma.b / s_size
        };
    }
}

////////////////////////////////////////////////////////////////////////////////
// Draw Seeds
////////////////////////////////////////////////////////////////////////////////

__global__
void superpixel_gslic__draw_seeds(
    Pixel_t *img,
    unsigned int n_pixels,
    Seed_t *seeds,
    unsigned int n_seeds,
    ClosestSeed_t *distances)
{
    const unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < n_pixels)
    {
        unsigned int s = distances[i].label;
        if (s < n_seeds)
        {
            Seed_t seed = seeds[s];
            img[i] = {
                .l = (unsigned char) seed.l,
                .a = (unsigned char) seed.a,
                .b = (unsigned char) seed.b
            };
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Entry point
////////////////////////////////////////////////////////////////////////////////

__global__
void superpixel_gslic(
    Pixel_t *img,
    unsigned int width,
    unsigned int height,
    Seed_t *seeds,
    unsigned int n_seeds,
    ClosestSeed_t *distances,
    Seed_t *seed_sigmas,
    double *seed_pixel_counts)
{
    const unsigned int n_pixels = width * height;
    const unsigned int seed_size = 0.5 + n_pixels / n_seeds;
    const unsigned int seed_strip_size = (int) sqrt((double) n_seeds);
    const unsigned int seed_side_len = width / seed_strip_size;

    superpixel_gslic__init_seeds
        <<< 1, 1 >>>
    (
        img, width, height, seeds, n_seeds
    );
    cudaDeviceSynchronize();

    for (unsigned int iter = 0; iter < ITERATIONS; ++iter)
    {
        superpixel_gslic__iter__dist
            <<< n_pixels/SUB_KERNEL_BS, SUB_KERNEL_BS >>>
        (
            img, width, height,
            seeds, n_seeds, seed_side_len,
            distances
        );
        cudaDeviceSynchronize();

        superpixel_gslic__iter__recalc_seeds
            <<< n_seeds / SUB_KERNEL_BS, SUB_KERNEL_BS >>>
        (
            img, width ,height,
            n_seeds, seeds,
            seed_sigmas, seed_pixel_counts, distances
        );
        cudaDeviceSynchronize();
    }

    superpixel_gslic__draw_seeds
        <<< n_pixels/SUB_KERNEL_BS, SUB_KERNEL_BS >>>
    (
        img, n_pixels, seeds, n_seeds, distances
    );
    cudaDeviceSynchronize();
}
