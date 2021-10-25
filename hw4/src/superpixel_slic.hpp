#ifndef _SUPERPIXEL_SLIC_HPP
#define _SUPERPIXEL_SLIC_HPP

#include <mutex>

#include <opencv2/core/mat.hpp>

using namespace std;
using namespace cv;

struct Seed
{
    double l, a, b, x, y;
};

struct DistChange
{
    double dist;
    int label;

    DistChange() : dist(DBL_MAX), label(-1) {}
    DistChange(double d, int l) : dist(d), label(l) {}
};

class SuperpixelSLIC
{
private:
    double m_runtime;

    const Mat *const m_img_in;
    Mat *m_img_out;

    int m_k, m_n_workers;

    int m_width, m_height, m_img_size;
    int m_cluster_size, m_step, m_strip_size, m_cluster_side_len;

    vector<Seed> m_kseeds;

    vector<int> m_labels;

public:
    SuperpixelSLIC(const Mat *const img_in, Mat *img_out, int k, int n_workers);

    SuperpixelSLIC(const SuperpixelSLIC &) = delete;
    SuperpixelSLIC(SuperpixelSLIC &&) = delete;
    SuperpixelSLIC &operator=(const SuperpixelSLIC &) = delete;

private:
    void _init_seeds();
    void _iterations();
    void _iterations_worker(int k_start, int k_end, vector<DistChange> &dists);
    void _enforce_connectivity();
    void _draw_contours();

public:
    void run();
};

#endif
