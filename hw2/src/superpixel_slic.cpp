#include <iostream>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "superpixel_slic.hpp"

#define COMPACTNESS 20.0

using namespace std;
using namespace cv;

SuperpixelSLIC::SuperpixelSLIC(Mat *img, int k) : m_img(img), m_k(k)
{
    m_strip_size = int(sqrt(k));
    m_n_channels = img->channels();

    m_width = img->size().width;
    m_height = img->size().height;
    m_img_size = m_width * m_height;

    split(*img, m_channels);

    m_cluster_size = 0.5 + double(m_img_size) / double(m_k);
    m_cluster_side_len = m_width / m_strip_size;

    m_kseeds_l.resize(m_k);
    m_kseeds_a.resize(m_k);
    m_kseeds_b.resize(m_k);
    m_kseeds_x.resize(m_k);
    m_kseeds_y.resize(m_k);

    m_labels.resize(m_img_size);

    _init_seeds();
}

inline void SuperpixelSLIC::_init_seeds()
{
    int xoff = m_cluster_side_len / 2;
    int yoff = m_cluster_side_len / 2;

    int err = m_width - m_cluster_side_len * m_strip_size;
    double err_per_strip = int(double(err) / double(m_strip_size));

    int i = 0;

    for (int y = 0; y < m_strip_size; y++)
    {
        int y_err = y * err_per_strip;
        int Y = y * m_cluster_side_len + yoff + y_err;
        if (Y > m_height - 1)
            continue;

        for (int x = 0; x < m_strip_size; x++)
        {
            int x_err = x * err_per_strip;
            int X = x * m_cluster_side_len + xoff + x_err;
            if (X > m_width - 1)
                continue;

            m_kseeds_l[i] = m_channels[0].at<uchar>(Y, X);
            m_kseeds_a[i] = m_channels[1].at<uchar>(Y, X);
            m_kseeds_b[i] = m_channels[2].at<uchar>(Y, X);

            m_kseeds_x[i] = (double)X;
            m_kseeds_y[i] = (double)Y;

            i++;
        }
    }
}

void SuperpixelSLIC::run()
{
    _iterations();
    _enforce_connectivity();
    _draw_contours();
}

inline void SuperpixelSLIC::_iterations()
{
    vector<double> clustersize(m_k, 0);
    vector<double> inverses(m_k, 0);
    vector<double> sigma_l(m_k, 0);
    vector<double> sigma_a(m_k, 0);
    vector<double> sigma_b(m_k, 0);
    vector<double> sigma_x(m_k, 0);
    vector<double> sigma_y(m_k, 0);
    vector<double> dists(m_img_size, DBL_MAX);

    double inverse_weight = double(m_cluster_side_len * m_cluster_side_len) / double(m_cluster_side_len);
    int x1, y1, x2, y2;
    double l, a, b, dist, distxy;

    for (int itr = 0; itr < 10; itr++)
    {
        dists.assign(m_img_size, DBL_MAX);
        for (int n = 0; n < m_k; n++)
        {
            y1 = max(0.0, m_kseeds_y[n] - m_cluster_side_len);
            y2 = min((double)m_height, m_kseeds_y[n] + m_cluster_side_len);
            x1 = max(0.0, m_kseeds_x[n] - m_cluster_side_len);
            x2 = min((double)m_width, m_kseeds_x[n] + m_cluster_side_len);
            for (int y = y1; y < y2; y++)
            {
                for (int x = x1; x < x2; x++)
                {
                    int i = y * m_width + x;
                    l = m_channels[0].at<uchar>(i);
                    a = m_channels[1].at<uchar>(i);
                    b = m_channels[2].at<uchar>(i);
                    dist = (l - m_kseeds_l[n]) * (l - m_kseeds_l[n]) +
                           (a - m_kseeds_a[n]) * (a - m_kseeds_a[n]) +
                           (b - m_kseeds_b[n]) * (b - m_kseeds_b[n]);
                    distxy = (x - m_kseeds_x[n]) * (x - m_kseeds_x[n]) +
                             (y - m_kseeds_y[n]) * (y - m_kseeds_y[n]);

                    //------------------------------------------------------------------------
                    dist += distxy * inverse_weight;
                    //------------------------------------------------------------------------
                    if (dist < dists[i])
                    {
                        dists[i] = dist;
                        m_labels[i] = n;
                    }
                }
            }
        }

        //-----------------------------------------------------------------
        // Recalculate the centroid and store in the seed values
        //-----------------------------------------------------------------
        //instead of reassigning memory on each iteration, just reset.

        sigma_l.assign(m_k, 0);
        sigma_a.assign(m_k, 0);
        sigma_b.assign(m_k, 0);
        sigma_x.assign(m_k, 0);
        sigma_y.assign(m_k, 0);
        clustersize.assign(m_k, 0);
        int ind = 0;
        for (int r = 0; r < m_height; r++)
        {
            for (int c = 0; c < m_width; c++)
            {
                sigma_l[m_labels[ind]] += m_channels[0].at<uchar>(ind);
                sigma_a[m_labels[ind]] += m_channels[1].at<uchar>(ind);
                sigma_b[m_labels[ind]] += m_channels[2].at<uchar>(ind);
                sigma_x[m_labels[ind]] += c;
                sigma_y[m_labels[ind]] += r;
                clustersize[m_labels[ind]] += 1.0;
                ind++;
            }
        }

        for (int k = 0; k < m_k; k++)
        {
            if (clustersize[k] <= 0)
                clustersize[k] = 1;
            // computing inverse now to multiply, than divide later
            inverses[k] = 1.0 / clustersize[k];
        }

        for (int k = 0; k < m_k; k++)
        {
            m_kseeds_l[k] = sigma_l[k] * inverses[k];
            m_kseeds_a[k] = sigma_a[k] * inverses[k];
            m_kseeds_b[k] = sigma_b[k] * inverses[k];
            m_kseeds_x[k] = sigma_x[k] * inverses[k];
            m_kseeds_y[k] = sigma_y[k] * inverses[k];
        }
    }
}

inline void SuperpixelSLIC::_enforce_connectivity()
{
    const int dx4[4] = {-1, 0, 1, 0};
    const int dy4[4] = {0, -1, 0, 1};

    vector<int> new_labels(m_labels.size(), 0);
    for (int i = 0; i < m_img_size; i++)
        new_labels[i] = -1;

    int label = 0;
    int *xvec = new int[m_img_size];
    int *yvec = new int[m_img_size];
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

    m_labels = new_labels;

    if (xvec)
        delete[] xvec;
    if (yvec)
        delete[] yvec;
}

inline void SuperpixelSLIC::_draw_contours()
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
        m_img->at<Vec3b>(contour_y[j], contour_x[j]) = color_ffffff;
        for (int n = 0; n < 8; n++)
        {
            int x = contour_x[j] + dx8[n];
            int y = contour_y[j] + dy8[n];
            if ((x >= 0 && x < m_width) && (y >= 0 && y < m_height))
            {
                int ind = y * m_width + x;
                if (!is_taken[ind])
                {
                    m_img->at<Vec3b>(contour_y[j], contour_x[j]) = color_000000;
                }
            }
        }
    }
}
