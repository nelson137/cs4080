#include <cstdlib>
#include <iostream>
#include <set>
#include <string>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "util.hpp"

#include "kernels.cuh"

using namespace std;

const char *ARG0 = "homework5";

static const set<int> FILTER_SIZE_CHOICES = { 3, 7, 11, 15 };

static void help_and_exit(int code = 0)
{
    ostream &os = code ? cerr : cout;
    os << "Usage: " << ARG0 << " FILTER_SIZE INFILE OUTFILE" << endl;
    os << endl;
    os << "\
DESCRIPTION\n\
  Perform a median filter on a PGM (Portable Gray Map) image.\n\
\n\
ARGUMENTS\n\
  FILTER_SIZE  The size of the filter window. This is the edge length of the\n\
               window, so a value of 3 means the window will be 3x3, i.e. a\n\
               radius of 1. Must be 3, 7, 11, or 15.\n\
  INFILE       The path of the input file.\n\
  OUTFILE      The path of the output file.\n\
";
    exit(code);
}

#define ERR(...) do { \
    fprintf(stderr, "error: "); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); \
    if (ret) \
        fprintf(stderr, "  %s\n", cudaGetErrorString(ret)); \
    goto err; \
} while(0)

int main(int argc, char *argv[])
{
    int code = 0;

    /**
     * Global initialization
     */

    ARG0 = argv[0];

    /**
     * Argument parsing
     */

    if (argc != 4)
        help_and_exit(1);

    // Parse and validate filter size
    const char *filter_size_str = argv[1];
    int filter_size;
    if (!cstr_to_int(filter_size_str, &filter_size))
        die("invalid filter size: %s", filter_size_str);
    if (FILTER_SIZE_CHOICES.find(filter_size) == FILTER_SIZE_CHOICES.end())
        die("unsupported filter size: %d", filter_size);
    int filter_radius = filter_size / 2;

    // Validate infile
    const char *infile = argv[2];
    if (!file_exists(infile))
        die("no such file: %s", infile);

    // Set outfile
    const char *outfile = argv[3];

    /**
     * Load image, launch kernel
     */

    cudaError_t ret = cudaSuccess;
    unsigned n_threads = 64;
    unsigned int width, height, n_pixels;
    size_t img_size;
    unsigned char *h_img_in = NULL, *h_img_out = NULL;

    unsigned char *d_img_in = NULL, *d_img_out = NULL;

    // Allocate host input image & load infile
    if (!sdkLoadPGM<unsigned char>(infile, &h_img_in, &width, &height))
        ERR("failed to load image: %s", infile);
    n_pixels = width * height;
    img_size = sizeof(unsigned char) * n_pixels;

    // Allocate host output image
    if (!(h_img_out = (unsigned char *) malloc(img_size)))
        ERR("failed to allocate space for output image");

    // Allocate device input image
    if ((ret = cudaMalloc(&d_img_in, img_size)))
        ERR("failed to allocate space for the input image on device");

    // Allocate device output image
    if ((ret = cudaMalloc(&d_img_out, img_size)))
        ERR("failed to allocate space for the output image on device");

    // Copy input image from host to device
    if ((ret = cudaMemcpy(d_img_in, h_img_in, img_size, cudaMemcpyHostToDevice)))
        ERR("failed to copy image data onto device");

    // Launch kernel
    median_filter<<< n_pixels / n_threads, n_threads >>>(
        d_img_in, d_img_out, n_pixels, (unsigned)filter_radius
    );
    if ((ret = cudaDeviceSynchronize()))
        ERR("failed to launch kernel");

    // Copy output image from device back to host
    if ((ret = cudaMemcpy(h_img_out, d_img_in, img_size, cudaMemcpyDeviceToHost)))
        ERR("failed to copy output image from device to host");

    // Write outfile
    if (!sdkSavePGM(outfile, h_img_out, width, height))
        ERR("failed to write to outfile: %s", outfile);

    /**
     * Cleanup
     */

    goto end;

err:
    code = 1;

end:
    cudaFree(d_img_out);
    cudaFree(d_img_in);
    free(h_img_out);
    free(h_img_in);

    return code;
}
