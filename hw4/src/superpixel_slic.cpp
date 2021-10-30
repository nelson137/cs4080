#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

#include <unistd.h>

#include <mpi.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "superpixel_slic.hpp"
#include "util.hpp"

#define N_ITERATIONS 10
#define COMPACTNESS 20.0

#define TAG_SEEDS 0

#define _W(__rank) "w(" << (__rank) << ") "
#define _W_DEBUG(__rank, __outstreams)        \
    do                                        \
    {                                         \
        cerr << _W((__rank)) << __outstreams; \
    } while (0)

using namespace std;
using namespace cv;

MPI_Datatype DistChange_T = MPI_DATATYPE_NULL;
MPI_Datatype Seed_T = MPI_DATATYPE_NULL;

MPI_Op DistChange_Op_min = MPI_OP_NULL;

void DistChange_Op_min_impl(
    DistChange *in,
    DistChange *inout,
    int *len,
    MPI_Datatype *datatype)
{
    for (int i = 0; i < *len; i++, in++, inout++)
        if (in->dist < inout->dist)
            *inout = *in;
}

SuperpixelSLIC_MPI::SuperpixelSLIC_MPI(
    const Mat *const img_in,
    Mat *img_out,
    int k,
    int n_workers)
    : m_img_in(img_in), m_img_out(img_out), m_k(k), m_n_workers(n_workers)
{
    m_cluster_strip_size = int(sqrt(k));

    m_width = m_img_in->size().width;
    m_height = m_img_in->size().height;
    m_img_size = m_width * m_height;

    m_clusters_per_worker = m_k / m_n_workers;

    m_cluster_size = 0.5 + double(m_img_size) / double(m_k);
    m_cluster_side_len = m_width / m_cluster_strip_size;
}

SuperpixelSLIC_MPI::~SuperpixelSLIC_MPI()
{
    delete[] m_kseeds;
    delete[] m_labels;
}

void SuperpixelSLIC_MPI::init(int rank)
{
    if (rank != RANK_MAIN)
        return;

    m_kseeds = new Seed[m_k];
    m_labels = new int[m_img_size];

    timer_start();
    _init_seeds();
    double t = timer_end();
    cout << "seed init time: " << t << " ms" << endl;
}

inline void SuperpixelSLIC_MPI::_init_seeds()
{
    int xoff = m_cluster_side_len / 2;
    int yoff = m_cluster_side_len / 2;

    int err = m_width - m_cluster_side_len * m_cluster_strip_size;
    double err_per_strip = int(double(err) / double(m_cluster_strip_size));

    int i = 0;

    for (int y = 0; y < m_cluster_strip_size; y++)
    {
        int y_err = y * err_per_strip;
        int Y = y * m_cluster_side_len + yoff + y_err;
        if (Y > m_height - 1)
            continue;

        for (int x = 0; x < m_cluster_strip_size; x++)
        {
            int x_err = x * err_per_strip;
            int X = x * m_cluster_side_len + xoff + x_err;
            if (X > m_width - 1)
                continue;

            Vec3b point = m_img_in->at<Vec3b>(Y, X);
            m_kseeds[i] = {(double)point(0),
                           (double)point(1),
                           (double)point(2),
                           (double)X,
                           (double)Y};

            i++;
        }
    }
}

void SuperpixelSLIC_MPI::run(int rank)
{
    double t;

    timer_start();
    _worker_main(rank);
    t = timer_end();

    if (rank != RANK_MAIN)
        return;

    cout << "iterations time: " << t << " ms" << endl;
    m_runtime += t;

    timer_start();
    _enforce_connectivity();
    t = timer_end();
    cout << "enforce connectivity time: " << t << " ms" << endl;
    m_runtime += t;

    timer_start();
    _draw_contours();
    t = timer_end();
    cout << "drawing contours time: " << t << " ms" << endl;
    m_runtime += t;

    cout << "total runtime: " << m_runtime << " ms" << endl;
}

inline void SuperpixelSLIC_MPI::_worker_main(int rank)
{
    Seed *worker_seeds = new Seed[m_k];
    DistChange *worker_dists = new DistChange[m_img_size];
    DistChange *min_distchanges = nullptr;
    vector<double> cluster_sizes;
    vector<double> inverses;
    vector<Seed> sigmas;

    if (rank == RANK_MAIN)
    {
        min_distchanges = new DistChange[m_img_size];
        cluster_sizes.reserve(m_img_size);
        inverses.reserve(m_k);
        sigmas.reserve(m_k);
    }

    const int k_start = rank * m_clusters_per_worker;
    const int k_end = k_start + m_clusters_per_worker;
    const double inverse_weight = m_cluster_side_len;

    for (int itr = 0; itr < N_ITERATIONS; itr++)
    {
        for (int i = 0; i < m_img_size; i++)
            worker_dists[i] = DistChange{DBL_MAX, -1};

        //------------------------------------------------------------
        // Distribute latest seeds
        //------------------------------------------------------------

        if (rank == RANK_MAIN)
        {
            // _W_DEBUG(rank, "::: itr " << itr << " :::" << endl);
            // Copy latest seeds into main worker's buffer
            memcpy(worker_seeds, m_kseeds, sizeof(Seed) * m_k);
        }

        // Send latest seeds to all workers
        MPI_Bcast(worker_seeds, m_k, Seed_T, RANK_MAIN, MPI_COMM_WORLD);
        // _W_DEBUG(rank, "broadcast seeds -> " << _W(w) << endl);

        //------------------------------------------------------------
        // Calculate distances
        //------------------------------------------------------------

        // _W_DEBUG(rank, "calc dists" << endl);
        for (int k = k_start; k < k_end; k++)
        {
            Seed seed = worker_seeds[k];

            int y1 = max(0.0, seed.y - m_cluster_side_len);
            int y2 = min((double)m_height, seed.y + m_cluster_side_len);
            int x1 = max(0.0, seed.x - m_cluster_side_len);
            int x2 = min((double)m_width, seed.x + m_cluster_side_len);

            for (int y = y1; y < y2; y++)
            {
                for (int x = x1; x < x2; x++)
                {
                    int i = y * m_width + x;
                    Vec3b point = m_img_in->at<Vec3b>(i);
                    double l = point(0);
                    double a = point(1);
                    double b = point(2);
                    double dist = (l - seed.l) * (l - seed.l) +
                                  (a - seed.a) * (a - seed.a) +
                                  (b - seed.b) * (b - seed.b);
                    double distxy = (x - seed.x) * (x - seed.x) +
                                    (y - seed.y) * (y - seed.y);
                    //--------------------------------------------------
                    dist += distxy * inverse_weight;
                    //--------------------------------------------------
                    if (dist < worker_dists[i].dist)
                        worker_dists[i] = DistChange{dist, k};
                }
            }
        }

        // _W_DEBUG(rank, "reduce" << endl);
        MPI_Reduce(worker_dists, min_distchanges, m_img_size, DistChange_T,
                   DistChange_Op_min, RANK_MAIN, MPI_COMM_WORLD);

        // Wait for all workers to call reduce with their calculated dists
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == RANK_MAIN)
        {
            //------------------------------------------------------------
            // Update labels using those in min dist changes from reduce
            //------------------------------------------------------------

            // _W_DEBUG(rank, "recalc clusters" << endl);
            for (int i = 0; i < m_img_size; i++)
                m_labels[i] = min_distchanges[i].label;

            //------------------------------------------------------------
            // Recalculate the centroid and store in the seed values
            //------------------------------------------------------------

            sigmas.assign(m_k, {0.0, 0.0, 0.0, 0.0, 0.0});
            cluster_sizes.assign(m_k, 0.0);

            {
                int ind = 0;
                for (int r = 0; r < m_height; r++)
                {
                    for (int c = 0; c < m_width; c++)
                    {
                        int l = m_labels[ind];
                        Seed &sigma = sigmas[l];
                        Vec3b point = m_img_in->at<Vec3b>(ind);
                        sigma.l += point(0);
                        sigma.a += point(1);
                        sigma.b += point(2);
                        sigma.x += c;
                        sigma.y += r;
                        cluster_sizes[l] += 1.0;
                        ind++;
                    }
                }
            }

            {
                for (int k = 0; k < m_k; k++)
                {
                    if (cluster_sizes[k] <= 0)
                        cluster_sizes[k] = 1;
                    // computing inverse now to multiply later
                    inverses[k] = 1.0 / cluster_sizes[k];
                }
            }

            {
                for (int k = 0; k < m_k; k++)
                {
                    Seed &sigma = sigmas[k];
                    double inv = inverses[k];
                    m_kseeds[k] = {
                        sigma.l * inv,
                        sigma.a * inv,
                        sigma.b * inv,
                        sigma.x * inv,
                        sigma.y * inv};
                }
            }
        }
    }

    delete[] worker_seeds;
    delete[] worker_dists;
    delete[] min_distchanges;
}

inline void SuperpixelSLIC_MPI::_enforce_connectivity()
{
    const int dx4[4] = {-1, 0, 1, 0};
    const int dy4[4] = {0, -1, 0, 1};

    vector<int> new_labels(m_img_size, -1);

    vector<int> xvec;
    xvec.reserve(m_img_size);
    vector<int> yvec;
    yvec.reserve(m_img_size);

    int label = 0;
    int oindex = 0;
    int adjlabel = 0; //adjacent label
    for (int j = 0; j < m_height; j++)
    {
        for (int k = 0; k < m_width; k++)
        {
            if (0 > new_labels[oindex])
            {
                new_labels[oindex] = label;
                //--------------------
                // Start a new segment
                //--------------------
                xvec[0] = k;
                yvec[0] = j;
                //-------------------------------------------------------
                // Quickly find an adjacent label for use later if needed
                //-------------------------------------------------------
                {
                    for (int n = 0; n < 4; n++)
                    {
                        int x = xvec[0] + dx4[n];
                        int y = yvec[0] + dy4[n];
                        if ((x >= 0 && x < m_width) && (y >= 0 && y < m_height))
                        {
                            int nindex = y * m_width + x;
                            if (new_labels[nindex] >= 0)
                                adjlabel = new_labels[nindex];
                        }
                    }
                }

                int count = 1;
                for (int c = 0; c < count; c++)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        int x = xvec[c] + dx4[n];
                        int y = yvec[c] + dy4[n];

                        if ((x >= 0 && x < m_width) && (y >= 0 && y < m_height))
                        {
                            int nindex = y * m_width + x;

                            if (0 > new_labels[nindex] && m_labels[oindex] == m_labels[nindex])
                            {
                                xvec[count] = x;
                                yvec[count] = y;
                                new_labels[nindex] = label;
                                count++;
                            }
                        }
                    }
                }
                //-------------------------------------------------------
                // If segment size is less then a limit, assign an
                // adjacent label found before, and decrement label count.
                //-------------------------------------------------------
                if (count <= m_cluster_size >> 2)
                {
                    for (int c = 0; c < count; c++)
                    {
                        int ind = yvec[c] * m_width + xvec[c];
                        new_labels[ind] = adjlabel;
                    }
                    label--;
                }
                label++;
            }
            oindex++;
        }
    }

    memcpy(m_labels, new_labels.data(), sizeof(int) * m_img_size);
}

inline void SuperpixelSLIC_MPI::_draw_contours()
{
    static constexpr int dx8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    static constexpr int dy8[8] = {0, -1, -1, -1, 0, 1, 1, 1};
    static const Vec3b color_ffffff = Vec3b(0xff, 0xff, 0xff);
    static const Vec3b color_000000 = Vec3b(0x00, 0x00, 0x00);

    vector<bool> is_taken(m_img_size, false);
    vector<int> contour_x(m_img_size);
    vector<int> contour_y(m_img_size);

    int mainindex = 0;
    int cind = 0;
    for (int j = 0; j < m_height; j++)
    {
        for (int k = 0; k < m_width; k++)
        {
            int np = 0;
            for (int i = 0; i < 8; i++)
            {
                int x = k + dx8[i];
                int y = j + dy8[i];

                if ((x >= 0 && x < m_width) && (y >= 0 && y < m_height))
                {
                    int index = y * m_width + x;
                    if (m_labels[mainindex] != m_labels[index])
                        np++;
                }
            }
            if (np > 1)
            {
                contour_x[cind] = k;
                contour_y[cind] = j;
                is_taken[mainindex] = true;
                cind++;
            }
            mainindex++;
        }
    }

    int numboundpix = cind;
    for (int j = 0; j < numboundpix; j++)
    {
        m_img_out->at<Vec3b>(contour_y[j], contour_x[j]) = color_ffffff;
        for (int n = 0; n < 8; n++)
        {
            int x = contour_x[j] + dx8[n];
            int y = contour_y[j] + dy8[n];
            if ((x >= 0 && x < m_width) && (y >= 0 && y < m_height))
            {
                int ind = y * m_width + x;
                if (!is_taken[ind])
                {
                    m_img_out->at<Vec3b>(contour_y[j], contour_x[j]) = color_000000;
                }
            }
        }
    }
}
