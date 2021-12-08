#ifndef _KERNELS_CUH
#define _KERNELS_CUH

__global__ void
median_filter(
    unsigned char *img_in,
    unsigned char *img_out,
    unsigned n_pixels,
    unsigned radius
);

#endif
