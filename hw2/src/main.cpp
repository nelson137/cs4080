#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "superpixel_slic.hpp"
#include "util.hpp"

using namespace std;
using namespace cv;

const char *ARG0 = "hw2";

static void help_and_exit(int code = 0)
{
    ostream &os = code ? cerr : cout;
    os << "Usage: " << ARG0 << " INFILE NUM_WORKERS NUM_CLUSTERS OUTFILE" << endl;
    os << endl;
    os << "\
DESCRIPTION\n\
  Run a parallelized version of the SLIC Superpixel algorithm on an input\n\
  image. The output image will contain NUM_CLUSTERS clusters and will be\n\
  computed using using NUM_WORKERS workers.\n\
\n\
ARGUMENTS\n\
  INFILE        The path of the input image.\n\
  NUM_WORKERS   The number of worker processes to use in the computation.\n\
                This number must be a positive integer.\n\
  NUM_CLUSTERS  The number of superpixel clusters to create in the output\n\
                image. This number must be a positive integer and a perfect\n\
                square.\n\
  OUTFILE       The path to use for the output image file.\n\
";
    exit(code);
}

int main(int argc, char *argv[])
{
    /**
     * Argument parsing
     */

    ARG0 = argv[0];

    if (argc != 5)
        help_and_exit(1);

    const char *infile = argv[1];
    if (!file_exists(infile))
        die("no such file: %s\n", infile);

    const char *n_workers_str = argv[2];
    int n_workers = 1;
    if (!cstr_to_int(n_workers_str, &n_workers))
        die("invalid number of workers: %s\n", n_workers_str);

    const char *n_clusters_str = argv[3];
    int n_clusters = 256;
    if (!cstr_to_int(n_clusters_str, &n_clusters))
        die("invalid number of superpixel clusters: %s\n", n_clusters_str);
    if (!is_perfect_square(n_clusters))
        die("invalid number of superpixel clusters, not a perfect square: %d\n",
            n_clusters);

    const char *outfile = argv[4];

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

    SuperpixelSLIC algo(&img_lab_in, &img_lab_out, n_clusters, n_workers);
    algo.run();

    /**
     * Write the output image to filesystem
     */

    // Convert back to RGB
    cvtColor(img_lab_out, img_rgb, COLOR_Lab2RGB);

    if (!imwrite(outfile, img_rgb))
        die("failed to write output image to file: %s\n", outfile);

    return 0;
}
