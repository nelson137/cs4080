#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include <cuda_runtime.h>
#include <helper_timer.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "gold_standard.hpp"
#include "superpixel_gslic.cuh"
#include "types.h"
#include "util.hpp"

using namespace std;
using namespace cv;

const char *ARG0 = "homework6";

static void help_and_exit(int code = 0)
{
    ostream &os = code ? cerr : cout;
    os << "Usage: " << ARG0 << " NUM_CLUSTERS INFILE OUTFILE" << endl;
    os << endl;
    os << "\
DESCRIPTION\n\
  Run a GPU implementation of the SLIC Superpixel algorithm on an input\n\
  image. The output image will contain NUM_CLUSTERS clusters.\n\
\n\
ARGUMENTS\n\
  NUM_CLUSTERS  The number of superpixel clusters to create in the output\n\
                image. This number must be a positive integer and a perfect\n\
                square.\n\
  INFILE        The path of the input image.\n\
  OUTFILE       The path to use for the output image file.\n\
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
    timer.reset(); \
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

    bool no_gold = string(ARG0).rfind("-no-gold") != string::npos;

    if (argc != 4)
        help_and_exit(1);

    // Parse & validate the number of clusters
    const char *n_seeds_str = argv[1];
    int n_seeds = 256;
    if (!cstr_to_int(n_seeds_str, &n_seeds))
        die("invalid number of superpixel clusters: %s", n_seeds_str);
    if (!is_perfect_square(n_seeds))
        die("invalid number of superpixel clusters, not a perfect square: %d",
            n_seeds);

    // Validate infile
    const char *infile = argv[2];
    if (!file_exists(infile))
        die("no such file: %s", infile);

    // Validate outfile
    const char *outfile = argv[3];
    if (string(outfile).rfind(".png") == string::npos)
        die("outfile must end with .png: %s", outfile);

    cout << "clusters: " << n_seeds << endl;
    cout << "infile: " << infile << endl;
    cout << "outfile: " << outfile << endl;

    /**
     * Load image, setup memory
     */

    StopWatchLinux timer;

    timer.start();
    Mat img_rgb = imread(infile, IMREAD_COLOR);
    timer.stop();
    MARK_TIME("image load time");

    unsigned int width = img_rgb.cols;
    unsigned int height = img_rgb.rows;
    unsigned int n_pixels = width * height;

    // Convert to CIELAB
    Mat h_img_lab_in, h_img_lab_out;
    cvtColor(img_rgb, h_img_lab_in, COLOR_BGR2Lab);
    h_img_lab_out.create(
        h_img_lab_in.rows, h_img_lab_in.cols, h_img_lab_in.type());

    cudaError_t ret = cudaSuccess;

    Pixel_t *d_img = NULL;
    size_t img_lab_size = sizeof(Pixel_t) * n_pixels;

    Seed_t *d_seeds = NULL;
    size_t seeds_size = sizeof(Seed_t) * n_seeds;

    ClosestSeed_t *d_distances = NULL;
    size_t distances_size = sizeof(ClosestSeed_t) * n_pixels;

    Mat gold_img_lab;
    Seed_t *h_gpu_seeds = NULL, *h_gold_seeds = NULL;
    double percent_match = 0.0;

    if ((ret = cudaMalloc(&d_img, img_lab_size)))
        ERR("failed to allocate space for image on device");

    if ((ret = cudaMalloc(&d_seeds, seeds_size)))
        ERR("failed to allocate space for seeds array on device");

    if ((ret = cudaMalloc(&d_distances, distances_size)))
        ERR("failed to allocate space for distances array on device");

    /**
     * Copy to kernel, run, copy back
     */

    timer.start();

    if ((ret = cudaMemcpy(d_img, h_img_lab_in.data, img_lab_size,
                          cudaMemcpyHostToDevice)))
        ERR("failed to copy image to device");

    superpixel_gslic
        <<< 1, 1 >>>
    (
        d_img, width, height, d_seeds, n_seeds, d_distances
    );

    if ((ret = cudaMemcpy(h_img_lab_out.data, d_img, img_lab_size,
                          cudaMemcpyDeviceToHost)))
        ERR("failed to copy output image to host");

    timer.stop();
    MARK_TIME("kernel latency");

    /**
     * Run gold standard and compare outputs
     */

    if (!no_gold)
    {
        gold_img_lab = h_img_lab_in.clone();
        SuperpixelGSLIC_Gold gold(
            (Pixel_t *)gold_img_lab.data, width, height, n_seeds
        );

        timer.start();
        gold.run();
        timer.stop();
        MARK_TIME("gold standard latency");

        h_gold_seeds = gold.get_seeds();

        if ((h_gpu_seeds = (Seed_t *)malloc(seeds_size)) == NULL)
            ERR("failed to allocate space for output seeds on host");
        if ((cudaMemcpy(h_gpu_seeds, d_seeds, seeds_size,
                        cudaMemcpyDeviceToHost)))
            ERR("failed to copy computed seeds to host");

        for (int i = 0; i < n_seeds; ++i, ++h_gpu_seeds, ++h_gold_seeds)
        {
            if (h_gpu_seeds->x == h_gold_seeds->x) percent_match += 0.2;
            if (h_gpu_seeds->y == h_gold_seeds->y) percent_match += 0.2;
            if (h_gpu_seeds->l == h_gold_seeds->l) percent_match += 0.2;
            if (h_gpu_seeds->a == h_gold_seeds->a) percent_match += 0.2;
            if (h_gpu_seeds->b == h_gold_seeds->b) percent_match += 0.2;
        }

        percent_match /= (double) n_seeds;
        percent_match *= 100.0;
        bool is_match = abs(100.0 - percent_match) < 0.01;
        cout << "exact seed match: "
             << fixed << setprecision(2) << (percent_match) << '%'
             << " (" << (is_match ? "PASS" : "FAIL") << ')'
             << endl;
    }

    /**
     * Write the output image to filesystem
     */

    // Convert back to RGB
    cvtColor(h_img_lab_out, img_rgb, COLOR_Lab2BGR);

    if (!imwrite(outfile, img_rgb))
        ERR("failed to write output image to file: %s", outfile);

    /**
     * Cleanup
     */

    goto end;

err:
    code = 1;

end:
    cudaFree(d_distances);
    cudaFree(d_seeds);
    cudaFree(d_img);

    return code;
}
