#ifndef _SUPERPIXEL_SLIC_HPP
#define _SUPERPIXEL_SLIC_HPP

#include <opencv2/core/mat.hpp>

using namespace std;
using namespace cv;

struct Seed
{
    double l, a, b, x, y;
};

class SuperpixelSLIC
{
private:
    double m_runtime;

    Mat *m_img;

    int m_k;

    int m_n_channels;
    vector<Mat> m_channels;

    int m_width, m_height, m_img_size;
    int m_cluster_size, m_step, m_strip_size, m_cluster_side_len;

    vector<Seed> m_kseeds;

    vector<int> m_labels;

public:
    SuperpixelSLIC(Mat *img_in, int k);

    SuperpixelSLIC(const SuperpixelSLIC &) = delete;
    SuperpixelSLIC(SuperpixelSLIC &&) = delete;
    SuperpixelSLIC &operator=(const SuperpixelSLIC &) = delete;

private:
    void _init_seeds();
    void _iterations();
    void _enforce_connectivity();
    void _draw_contours();

public:
    void run();
};

#endif
