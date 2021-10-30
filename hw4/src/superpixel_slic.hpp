#ifndef _SUPERPIXEL_SLIC_HPP
#define _SUPERPIXEL_SLIC_HPP

#include <mutex>

#include <mpi.h>
#include <opencv2/core/mat.hpp>

using namespace std;
using namespace cv;

#define RANK_MAIN 0

struct Seed
{
    double l, a, b, x, y;
};

extern MPI_Datatype Seed_T;

struct DistChange
{
    double dist;
    int label;
};

void DistChange_Op_min_impl(DistChange *, DistChange *, int *, MPI_Datatype *);

extern MPI_Datatype DistChange_T;
extern MPI_Op DistChange_Op_min;

class SuperpixelSLIC_MPI
{
private:
    double m_runtime = 0.0;

    const Mat *const m_img_in;
    Mat *m_img_out;

    int m_k, m_n_workers, m_clusters_per_worker;

    int m_width, m_height, m_img_size;
    int m_cluster_size, m_cluster_strip_size, m_cluster_side_len;

    Seed *m_kseeds = nullptr;
    int *m_labels = nullptr;

public:
    SuperpixelSLIC_MPI(const Mat *const img_in, Mat *img_out, int k, int n_workers);
    ~SuperpixelSLIC_MPI();

    SuperpixelSLIC_MPI(const SuperpixelSLIC_MPI &) = delete;
    SuperpixelSLIC_MPI(SuperpixelSLIC_MPI &&) = delete;
    SuperpixelSLIC_MPI &operator=(const SuperpixelSLIC_MPI &) = delete;

private:
    void _init_seeds();
    void _worker_main(int rank);
    void _enforce_connectivity();
    void _draw_contours();

public:
    void init(int rank);
    void run(int rank);
};

#endif
