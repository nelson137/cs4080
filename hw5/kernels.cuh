#ifndef _KERNELS_CUH
#define _KERNELS_CUH

__global__
void median_filter_r1(
    unsigned char *img_in,
    unsigned char *img_out,
    unsigned width,
    unsigned height
);

__global__
void median_filter_r3(
    unsigned char *img_in,
    unsigned char *img_out,
    unsigned width,
    unsigned height
);

__global__
void median_filter_r5(
    unsigned char *img_in,
    unsigned char *img_out,
    unsigned width,
    unsigned height
);

__global__
void median_filter_r7(
    unsigned char *img_in,
    unsigned char *img_out,
    unsigned width,
    unsigned height
);

#endif
