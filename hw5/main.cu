#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_image.h>
#include <helper_timer.h>

#include "gold_standard.hpp"
#include "util.hpp"

#include "kernels.cuh"

using namespace std;

const char *ARG0 = "homework5";

static void help_and_exit(int code = 0)
{
    ostream &os = code ? cerr : cout;
    os << "Usage: " << ARG0 << " FILTER_SIZE INFILE OUTFILE [THREADS]" << endl;
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
  THREADS      The number of threads per block to use for the kernel\n\
               configuration.\n\
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

#define MARK_TIME(__label) do { \
    double __t_ms = timer.getTime(); \
    printf("%s: %.4f ms\n", (__label), __t_ms); \
    (timer).reset(); \
} while (0)

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

    bool test_gold_standard = string(ARG0).rfind("-no-gold") == string::npos;

    if (argc != 4 && argc != 5)
        help_and_exit(1);

    // Parse and validate filter size
    const char *filter_size_str = argv[1];
    int filter_size;
    if (!cstr_to_int(filter_size_str, &filter_size))
        die("invalid filter size: %s", filter_size_str);
    int filter_radius = filter_size / 2;

    // Validate filter size, choose kernel
    void (*kernel)(unsigned char *, unsigned char *, unsigned int, unsigned int);
    switch (filter_size)
    {
        case  3: kernel = median_filter_r1; break;
        case  7: kernel = median_filter_r3; break;
        case 11: kernel = median_filter_r5; break;
        case 15: kernel = median_filter_r7; break;
        default: die("unsupported filter size: %d", filter_size);
    }

    // Validate infile
    const char *infile = argv[2];
    if (!file_exists(infile))
        die("no such file: %s", infile);

    // Set outfile
    const char *outfile = argv[3];

    unsigned n_threads = 64;
    if (argc == 5)
    {
        int t;
        const char *n_threads_str = argv[4];
        if (!cstr_to_int(n_threads_str, &t))
            die("invalid number of threads: %s", n_threads_str);
        if (t <= 0)
            die("unsupported number of threads: %d", t);
        n_threads = (unsigned) t;
    }

    cout << "infile: " << infile << endl;
    cout << "outfile: " << outfile << endl;
    cout << "threads: " << n_threads << endl;

    /**
     * Load image, launch kernel
     */

    cudaError_t ret = cudaSuccess;
    dim3 grid, blocks;
    unsigned int width, height, n_pixels;
    size_t img_size;
    unsigned char *h_img_in = NULL, *h_img_out = NULL;

    unsigned char *d_img_in = NULL, *d_img_out = NULL;

    unique_ptr<unsigned char[]> gold_img_out;
    double matching_pixels = 0.0, percent_match = 0.0;

    StopWatchLinux timer;

    // Allocate host input image & load infile
    timer.start();
    if (!sdkLoadPGM<unsigned char>(infile, &h_img_in, &width, &height))
        ERR("failed to load image: %s", infile);
    timer.stop();
    cout << "image dimensions: " << width << " x " << height << endl;
    MARK_TIME("image load time");
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
    // Initialize device output image (for border)
    if ((ret = cudaMemset(d_img_out, 0x00, img_size)))
        ERR("failed to initialize output image on device");

    // Define configuration
    grid = dim3(n_pixels / n_threads);
    blocks = dim3(n_threads);

    // Warmup
    kernel<<< grid, blocks >>>(d_img_in, d_img_out, width, height);
    if ((ret = cudaDeviceSynchronize()))
        ERR("failed to launch kernel (for warmup");

    timer.start();

    // Copy input image from host to device
    if ((ret = cudaMemcpy(d_img_in, h_img_in, img_size, cudaMemcpyHostToDevice)))
        ERR("failed to copy image data onto device");

    // Launch kernel
    kernel<<< grid, blocks >>>(d_img_in, d_img_out, width, height);
    if ((ret = cudaDeviceSynchronize()))
        ERR("failed to launch kernel");

    // Copy output image from device back to host
    if ((ret = cudaMemcpy(h_img_out, d_img_out, img_size, cudaMemcpyDeviceToHost)))
        ERR("failed to copy output image from device to host");

    timer.stop();
    MARK_TIME("kernel latency");

    if (test_gold_standard)
    {
        // Compute "gold standard"
        timer.start();
        gold_img_out = gold_standard(h_img_in, width, height, filter_radius);
        timer.stop();
        MARK_TIME("gold standard latency");

        // Calculate pixel-wise exact match of device output and gold standard
        for (int i = 0; i < n_pixels; i++)
            if (gold_img_out.get()[i] == h_img_out[i])
                matching_pixels++;
        percent_match = 100.0 * matching_pixels / n_pixels;
        cout << "pixel-wise exact match: "
             << fixed << setprecision(2) << percent_match << '%'
             << " (" << (percent_match == 100.0 ? "PASS" : "FAIL") << ')'
             << endl;
    }

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
