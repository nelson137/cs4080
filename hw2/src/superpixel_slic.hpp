#ifndef _SUPERPIXEL_SLIC_HPP
#define _SUPERPIXEL_SLIC_HPP

#include <opencv2/core/mat.hpp>

using namespace std;
using namespace cv;

class SuperpixelSLIC
{
private:
    Mat *m_img;

    int m_k;

    int m_n_channels;
    vector<Mat> m_channels;

    int m_width, m_height, m_img_size;
    int m_cluster_size, m_step, m_strip_size, m_cluster_side_len;

    vector<double> m_kseeds_l;
    vector<double> m_kseeds_a;
    vector<double> m_kseeds_b;

    vector<double> m_kseeds_x;
    vector<double> m_kseeds_y;

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
