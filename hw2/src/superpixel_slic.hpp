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
    int m_strip_size;

    int m_n_channels;

    vector<Mat> m_channels;
    int m_img_size, m_width, m_height, m_cluster_size, m_cluster_side_len;

    vector<double> m_kseeds_l;
    vector<double> m_kseeds_a;
    vector<double> m_kseeds_b;

    vector<double> m_kseeds_x;
    vector<double> m_kseeds_y;

    vector<int> m_klabels, m_klabels_connected;

public:
    SuperpixelSLIC(Mat *img_in, int k);
    ~SuperpixelSLIC();

    SuperpixelSLIC(const SuperpixelSLIC &) = delete;
    SuperpixelSLIC(SuperpixelSLIC &&) = delete;
    SuperpixelSLIC &operator=(const SuperpixelSLIC &) = delete;

private:
    // void _init_seeds();
    // void _iterations();
    // void _enforce_connectivity();
    // void _draw_contours();

    void _init_seeds(
        vector<double> &kseedsl,
        vector<double> &kseedsa,
        vector<double> &kseedsb,
        vector<double> &kseedsx,
        vector<double> &kseedsy,
        const int &STEP);

    void _iterations(
        vector<double> &kseedsl,
        vector<double> &kseedsa,
        vector<double> &kseedsb,
        vector<double> &kseedsx,
        vector<double> &kseedsy,
        int *&klabels,
        const int &STEP,
        const double &M);

    void _enforce_connectivity(
        const int *labels, //input labels that need to be corrected to remove stray labels
        const int width,
        const int height,
        int *&nlabels,  //new labels
        int &numlabels, //the number of labels changes in the end if segments are removed
        const int &K);  //the number of superpixels desired by the user

    void _draw_contours(
        // unsigned int *&ubuff,
        int *&labels,
        const int &width,
        const int &height);

public:
    void run();
};

#endif
