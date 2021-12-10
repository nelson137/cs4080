#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "gold_standard.hpp"
#include "util.hpp"

//#include "superpixel_gslic.cuh"

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

    //bool test_gold_standard = string(ARG0).rfind("-no-gold") == string::npos;

    if (argc != 5)
        help_and_exit(1);

    // Parse & validate the number of clusters
    const char *n_clusters_str = argv[1];
    int n_clusters = 256;
    if (!cstr_to_int(n_clusters_str, &n_clusters))
        die("invalid number of superpixel clusters: %s", n_clusters_str);
    if (!is_perfect_square(n_clusters))
        die("invalid number of superpixel clusters, not a perfect square: %d",
            n_clusters);

    // Validate infile
    const char *infile = argv[2];
    if (!file_exists(infile))
        die("no such file: %s", infile);

    // Validate filter size, choose kernel
    /*void (*kernel)(unsigned char *, unsigned char *, unsigned int, unsigned int);
    switch (filter_size)
    {
        case  3: kernel = median_filter_r1; break;
        case  7: kernel = median_filter_r3; break;
        case 11: kernel = median_filter_r5; break;
        case 15: kernel = median_filter_r7; break;
        default: die("unsupported filter size: %d", filter_size);
    }*/

    // Set outfile
    const char *outfile = argv[3];

    unsigned n_threads = 64;
    /*if (argc == 5)
    {
        int t;
        const char *n_threads_str = argv[4];
        if (!cstr_to_int(n_threads_str, &t))
            die("invalid number of threads: %s", n_threads_str);
        if (t <= 0)
            die("unsupported number of threads: %d", t);
        n_threads = (unsigned) t;
    }*/

    cout << "clusters: " << n_clusters << endl;
    cout << "infile: " << infile << endl;
    cout << "outfile: " << outfile << endl;
    cout << "threads: " << n_threads << endl;

    /**
     * Read the input image into memory
     */

    Mat img_rgb, img_lab_in;
    img_rgb = imread(infile, IMREAD_COLOR);

    // Convert to CIELAB
    cvtColor(img_rgb, img_lab_in, COLOR_BGR2Lab);

    Mat img_lab_out = img_lab_in.clone();

    /**
     * Run the algorithm
     */

    /*SuperpixelGSLIC algo(&img_lab_in, &img_lab_out, n_clusters, n_workers);
    algo.init(rank);
    algo.run(rank);*/

    /**
     * Write the output image to filesystem
     */

    // Convert back to RGB
    /*cvtColor(img_lab_out, img_rgb, COLOR_Lab2RGB);

    if (!imwrite(outfile, img_rgb))
        die("failed to write output image to file: %s", outfile);*/

    ////////////////////////////////////////////////////////////////////////////

    /**
     * Load image, launch kernel
     */

    /*cudaError_t ret = cudaSuccess;
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
        ERR("failed to write to outfile: %s", outfile);*/

    /**
     * Cleanup
     */

    goto end;

//err:
    code = 1;

end:
    /*cudaFree(d_img_out);
    cudaFree(d_img_in);
    free(h_img_out);
    free(h_img_in);*/

    return code;
}
