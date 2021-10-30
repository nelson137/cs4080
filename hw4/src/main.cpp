#include <iostream>

#include <mpi.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "superpixel_slic.hpp"
#include "util.hpp"

using namespace std;
using namespace cv;

#ifdef DEBUG
#include <chrono>
#include <cstdlib>
#include <thread>
using namespace chrono;
#endif

#define W_TAG 0

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

#ifdef DEBUG
extern "C" const char *__asan_default_options()
{
    // There is a lot of output for leaks detected in MPI_* functions, so just
    // turn them all off and use Valgrind.
    return "detect_leaks=0";
}
#endif

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
        die("no such file: %s", infile);

    const char *n_workers_str = argv[2];
    int n_workers = 1;
    if (!cstr_to_int(n_workers_str, &n_workers))
        die("invalid number of workers: %s", n_workers_str);

    const char *n_clusters_str = argv[3];
    int n_clusters = 256;
    if (!cstr_to_int(n_clusters_str, &n_clusters))
        die("invalid number of superpixel clusters: %s", n_clusters_str);
    if (!is_perfect_square(n_clusters))
        die("invalid number of superpixel clusters, not a perfect square: %d",
            n_clusters);

    if (n_clusters % n_workers)
        die("number of workers must evenly divide number of superpixel clusters");

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

    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create Seed datatype
    {
        // This is a lot easier than using MPI_Type_create_struct() to create
        // a type that mimics the struct, then MPI_Type_create_resized() to take
        // padding into account. This also continues to work without needing
        // modification if the fields of the type were to change.
        MPI_Type_contiguous(sizeof(Seed), MPI_BYTE, &Seed_T);
        MPI_Type_commit(&Seed_T);
        int size;
        MPI_Type_size(Seed_T, &size);
        assert(size == sizeof(Seed));
    }

    // Create DistChange datatype
    {
        MPI_Type_contiguous(sizeof(DistChange), MPI_BYTE, &DistChange_T);
        MPI_Type_commit(&DistChange_T);
        int size;
        MPI_Type_size(DistChange_T, &size);
        assert(size == sizeof(DistChange));
    }

    // Create DistChange reduction operation
    MPI_Op_create(
        (MPI_User_function *)DistChange_Op_min_impl, true, &DistChange_Op_min);

#ifdef DEBUG
    /**
     * Infinite loop waiting for debugger to attach.
     *
     * When using mpirun, there is no way to launch with a debugger, so it must
     * be attached to the running process. Loop indefinitely until a debugger is
     * attached and the value of the continue flag is manually changed to true.
     *
     * This can be a little flakey, sometimes the debugger attaches but doesn't
     * stop on the breakpoint. Just stop the debugger, interrupt mpirun with
     * ^C, and try again. It has never not worked the second time for me.
     */

    if (getenv("WAIT_FOR_DEBUGGER_ATTACH") != nullptr)
    {
        static const char *const WAIT_FOR_DEBUGGER_ATTACH_NOTE =
            "Infinite loop waiting for debugger to attach; in order to\n"
            "continue, flag must be manually set to true.\n\n"
            "NOTE: VS Code *must* be run with administrative privileges\n"
            "      in order to attach the debugger to a running process.\n\n";
        if (rank == RANK_MAIN)
            cerr << WAIT_FOR_DEBUGGER_ATTACH_NOTE;

        bool a000_continue = false;
        while (!a000_continue)
            this_thread::sleep_for(milliseconds(100));
    }
#endif

    SuperpixelSLIC_MPI algo(&img_lab_in, &img_lab_out, n_clusters, n_workers);
    algo.init(rank);
    algo.run(rank);

    MPI_Finalize();
    if (rank != RANK_MAIN)
        return 0;

    /**
     * Write the output image to filesystem
     */

    // Convert back to RGB
    cvtColor(img_lab_out, img_rgb, COLOR_Lab2RGB);

    if (!imwrite(outfile, img_rgb))
        die("failed to write output image to file: %s", outfile);

    return 0;
}
