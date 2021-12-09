#include "kernels.cuh"

__device__
void selection_sort(unsigned char arr[], int length)
{
    for (int i=0; i<length; i++)
    {
        unsigned char min_val = arr[i];
        int min_idx = i;

        for (int j=i+1; j<length; j++)
        {
            unsigned char val_j = arr[j];
            if (val_j < min_val)
            {
                min_idx = j;
                min_val = val_j;
            }
        }

        if (min_idx != i)
        {
            arr[min_idx] = arr[i];
            arr[i] = min_val;
        }
    }
}

__device__
bool copy_neighborhood(
    unsigned char *img,
    unsigned width,
    unsigned height,
    size_t i,
    unsigned char *neighborhood,
    unsigned radius)
{
    size_t y = i / width;
    size_t x = i % width;

    if (!(radius <= y && y < height-radius && radius <= x && x < width-radius))
        return false;

    // Side length of the neighborhood
    size_t n_side = 2*radius + 1;
    // Index of the first row in the neighborhood (top left pixel)
    size_t n_i     = i - radius - radius*width;
    // Index of the last row in the neighborhood (bot left pixel)
    size_t n_i_end = i - radius + radius*width;
    // Copy pixels of surrounding radius into neighborhood array
    while (n_i <= n_i_end)
    {
        memcpy(neighborhood, img + n_i, n_side);
        neighborhood += n_side;
        n_i += width;
    }

    return true;
}

__global__
void median_filter_r1(
    unsigned char *img_in,
    unsigned char *img_out,
    unsigned width,
    unsigned height)
{
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;

    // Copy neighborhood into local mem
    const size_t n_area = 3 * 3;
    unsigned char neighborhood[n_area];

    // Sort and set output pixel to median
    if (copy_neighborhood(img_in, width, height, i, neighborhood, 1))
    {
        selection_sort(neighborhood, n_area);
        img_out[i] = neighborhood[1*3 + 1];
    }
}

__global__
void median_filter_r3(
    unsigned char *img_in,
    unsigned char *img_out,
    unsigned width,
    unsigned height)
{
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;

    // Copy neighborhood into local mem
    const size_t n_area = 7 * 7;
    unsigned char neighborhood[n_area];

    // Sort and set output pixel to median
    if (copy_neighborhood(img_in, width, height, i, neighborhood, 3))
    {
        selection_sort(neighborhood, n_area);
        img_out[i] = neighborhood[3*7 + 3];
    }
}

__global__
void median_filter_r5(
    unsigned char *img_in,
    unsigned char *img_out,
    unsigned width,
    unsigned height)
{
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;

    // Copy neighborhood into local mem
    const size_t n_area = 11 * 11;
    unsigned char neighborhood[n_area];

    // Sort and set output pixel to median
    if (copy_neighborhood(img_in, width, height, i, neighborhood, 5))
    {
        selection_sort(neighborhood, n_area);
        img_out[i] = neighborhood[5*11 + 5];
    }
}

__global__
void median_filter_r7(
    unsigned char *img_in,
    unsigned char *img_out,
    unsigned width,
    unsigned height)
{
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;

    // Copy neighborhood into local mem
    const size_t n_area = 15 * 15;
    unsigned char neighborhood[n_area];

    // Sort and set output pixel to median
    if (copy_neighborhood(img_in, width, height, i, neighborhood, 7))
    {
        selection_sort(neighborhood, n_area);
        img_out[i] = neighborhood[7*15 + 7];
    }
}
