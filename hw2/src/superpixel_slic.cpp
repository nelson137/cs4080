#include <iostream>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "superpixel_slic.hpp"

#define COMPACTNESS 10.0

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

    m_cluster_size = m_img_size / m_k;
    m_cluster_side_len = m_width / m_strip_size;

    m_kseeds_l.resize(m_k);
    m_kseeds_a.resize(m_k);
    m_kseeds_b.resize(m_k);
    m_kseeds_x.resize(m_k);
    m_kseeds_y.resize(m_k);

    m_klabels.resize(m_img_size);
    m_klabels_connected.resize(m_img_size);

    int superpixel_size = 0.5 + double(m_img_size) / double(m_k);
    int step = sqrt(double(superpixel_size)) + 0.5;
    _init_seeds(m_kseeds_l, m_kseeds_a, m_kseeds_b, m_kseeds_x, m_kseeds_y, step);
}

inline void SuperpixelSLIC::_init_seeds(
    vector<double> &kseedsl,
    vector<double> &kseedsa,
    vector<double> &kseedsb,
    vector<double> &kseedsx,
    vector<double> &kseedsy,
    const int &STEP)
{
    int numseeds(0);
    int n(0);

    int xstrips = (0.5 + double(m_width) / double(STEP));
    int ystrips = (0.5 + double(m_height) / double(STEP));

    int xerr = m_width - STEP * xstrips;
    if (xerr < 0)
    {
        xstrips--;
        xerr = m_width - STEP * xstrips;
    }
    int yerr = m_height - STEP * ystrips;
    if (yerr < 0)
    {
        ystrips--;
        yerr = m_height - STEP * ystrips;
    }

    double xerrperstrip = double(xerr) / double(xstrips);
    double yerrperstrip = double(yerr) / double(ystrips);

    int xoff = STEP / 2;
    int yoff = STEP / 2;
    //-------------------------
    numseeds = xstrips * ystrips;
    //-------------------------
    kseedsl.resize(numseeds);
    kseedsa.resize(numseeds);
    kseedsb.resize(numseeds);
    kseedsx.resize(numseeds);
    kseedsy.resize(numseeds);

    for (int y = 0; y < ystrips; y++)
    {
        int ye = y * yerrperstrip;
        for (int x = 0; x < xstrips; x++)
        {
            int xe = x * xerrperstrip;
            int seedx = (x * STEP + xoff + xe);
            int seedy = (y * STEP + yoff + ye);
            int i = seedy * m_width + seedx;

            kseedsl[n] = m_channels[0].at<uchar>(i);
            kseedsa[n] = m_channels[1].at<uchar>(i);
            kseedsb[n] = m_channels[2].at<uchar>(i);
            kseedsx[n] = seedx;
            kseedsy[n] = seedy;
            n++;
        }
    }
}

SuperpixelSLIC::~SuperpixelSLIC()
{
}

void SuperpixelSLIC::run()
{
    int *labels = m_klabels.data();
    int superpixel_size = 0.5 + double(m_img_size) / double(m_k);
    int step = sqrt(double(superpixel_size)) + 0.5;
    _iterations(m_kseeds_l, m_kseeds_a, m_kseeds_b, m_kseeds_x, m_kseeds_y, labels, step, 20.0);
    int *new_labels = m_klabels_connected.data();
    int n_labels = m_kseeds_l.size();
    _enforce_connectivity(labels, m_width, m_height, new_labels, n_labels, m_img_size / (step * step));
    {
        for (int i = 0; i < m_img_size; i++)
            labels[i] = new_labels[i];
    }
    _draw_contours(new_labels, m_width, m_height);
}

inline void SuperpixelSLIC::_iterations(
    vector<double> &kseedsl,
    vector<double> &kseedsa,
    vector<double> &kseedsb,
    vector<double> &kseedsx,
    vector<double> &kseedsy,
    int *&klabels,
    const int &STEP,
    const double &M)
{
    int sz = m_width * m_height;
    const int numk = kseedsl.size();
    int offset = STEP;

    vector<double> clustersize(numk, 0);
    vector<double> inv(numk, 0); //to store 1/clustersize[k] values
    vector<double> sigmal(numk, 0);
    vector<double> sigmaa(numk, 0);
    vector<double> sigmab(numk, 0);
    vector<double> sigmax(numk, 0);
    vector<double> sigmay(numk, 0);
    vector<double> distvec(sz, DBL_MAX);
    double invwt = 1.0 / ((STEP / M) * (STEP / M));
    int x1, y1, x2, y2;
    double l, a, b;
    double dist;
    double distxy;
    for (int itr = 0; itr < 10; itr++)
    {
        distvec.assign(sz, DBL_MAX);
        for (int n = 0; n < numk; n++)
        {
            y1 = max(0.0, kseedsy[n] - offset);
            y2 = min((double)m_height, kseedsy[n] + offset);
            x1 = max(0.0, kseedsx[n] - offset);
            x2 = min((double)m_width, kseedsx[n] + offset);
            for (int y = y1; y < y2; y++)
            {
                for (int x = x1; x < x2; x++)
                {
                    int i = y * m_width + x;
                    l = m_channels[0].at<uchar>(i);
                    a = m_channels[1].at<uchar>(i);
                    b = m_channels[2].at<uchar>(i);
                    dist = (l - kseedsl[n]) * (l - kseedsl[n]) +
                           (a - kseedsa[n]) * (a - kseedsa[n]) +
                           (b - kseedsb[n]) * (b - kseedsb[n]);
                    distxy = (x - kseedsx[n]) * (x - kseedsx[n]) +
                             (y - kseedsy[n]) * (y - kseedsy[n]);

                    //------------------------------------------------------------------------
                    dist += distxy * invwt;
                    //------------------------------------------------------------------------
                    if (dist < distvec[i])
                    {
                        distvec[i] = dist;
                        klabels[i] = n;
                    }
                }
            }
        }
        //-----------------------------------------------------------------
        // Recalculate the centroid and store in the seed values
        //-----------------------------------------------------------------
        //instead of reassigning memory on each iteration, just reset.

        sigmal.assign(numk, 0);
        sigmaa.assign(numk, 0);
        sigmab.assign(numk, 0);
        sigmax.assign(numk, 0);
        sigmay.assign(numk, 0);
        clustersize.assign(numk, 0);
        int ind(0);
        for (int r = 0; r < m_height; r++)
        {
            for (int c = 0; c < m_width; c++)
            {
                sigmal[klabels[ind]] += m_channels[0].at<uchar>(ind);
                sigmaa[klabels[ind]] += m_channels[0].at<uchar>(ind);
                sigmab[klabels[ind]] += m_channels[0].at<uchar>(ind);
                sigmax[klabels[ind]] += c;
                sigmay[klabels[ind]] += r;
                clustersize[klabels[ind]] += 1.0;
                ind++;
            }
        }
        for (int k = 0; k < numk; k++)
        {
            if (clustersize[k] <= 0)
                clustersize[k] = 1;
            // computing inverse now to multiply, than divide later
            inv[k] = 1.0 / clustersize[k];
        }

        for (int k = 0; k < numk; k++)
        {
            kseedsl[k] = sigmal[k] * inv[k];
            kseedsa[k] = sigmaa[k] * inv[k];
            kseedsb[k] = sigmab[k] * inv[k];
            kseedsx[k] = sigmax[k] * inv[k];
            kseedsy[k] = sigmay[k] * inv[k];
        }
    }
}

inline void SuperpixelSLIC::_enforce_connectivity(
    const int *labels, //input labels that need to be corrected to remove stray labels
    const int width,
    const int height,
    int *&nlabels,  //new labels
    int &numlabels, //the number of labels changes in the end if segments are removed
    const int &K)   //the number of superpixels desired by the user
{
    const int dx4[4] = {-1, 0, 1, 0};
    const int dy4[4] = {0, -1, 0, 1};

    const int sz = width * height;
    const int SUPSZ = sz / K;
    for (int i = 0; i < sz; i++)
        nlabels[i] = -1;
    int label(0);
    int *xvec = new int[sz];
    int *yvec = new int[sz];
    int oindex(0);
    int adjlabel(0); //adjacent label
    for (int j = 0; j < height; j++)
    {
        for (int k = 0; k < width; k++)
        {
            if (0 > nlabels[oindex])
            {
                nlabels[oindex] = label;
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
                        if ((x >= 0 && x < width) && (y >= 0 && y < height))
                        {
                            int nindex = y * width + x;
                            if (nlabels[nindex] >= 0)
                                adjlabel = nlabels[nindex];
                        }
                    }
                }

                int count(1);
                for (int c = 0; c < count; c++)
                {
                    for (int n = 0; n < 4; n++)
                    {
                        int x = xvec[c] + dx4[n];
                        int y = yvec[c] + dy4[n];

                        if ((x >= 0 && x < width) && (y >= 0 && y < height))
                        {
                            int nindex = y * width + x;

                            if (0 > nlabels[nindex] && labels[oindex] == labels[nindex])
                            {
                                xvec[count] = x;
                                yvec[count] = y;
                                nlabels[nindex] = label;
                                count++;
                            }
                        }
                    }
                }
                //-------------------------------------------------------
                // If segment size is less then a limit, assign an
                // adjacent label found before, and decrement label count.
                //-------------------------------------------------------
                if (count <= SUPSZ >> 2)
                {
                    for (int c = 0; c < count; c++)
                    {
                        int ind = yvec[c] * width + xvec[c];
                        nlabels[ind] = adjlabel;
                    }
                    label--;
                }
                label++;
            }
            oindex++;
        }
    }
    numlabels = label;

    if (xvec)
        delete[] xvec;
    if (yvec)
        delete[] yvec;
}

inline void SuperpixelSLIC::_draw_contours(
    // unsigned int *&ubuff,
    int *&labels,
    const int &width,
    const int &height)
{
    const int dx8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    const int dy8[8] = {0, -1, -1, -1, 0, 1, 1, 1};

    int sz = width * height;
    vector<bool> istaken(sz, false);
    vector<int> contourx(sz);
    vector<int> contoury(sz);
    int mainindex(0);
    int cind(0);
    for (int j = 0; j < height; j++)
    {
        for (int k = 0; k < width; k++)
        {
            int np(0);
            for (int i = 0; i < 8; i++)
            {
                int x = k + dx8[i];
                int y = j + dy8[i];

                if ((x >= 0 && x < width) && (y >= 0 && y < height))
                {
                    int index = y * width + x;
                    if (labels[mainindex] != labels[index])
                        np++;
                }
            }
            if (np > 1)
            {
                contourx[cind] = k;
                contoury[cind] = j;
                istaken[mainindex] = true;
                cind++;
            }
            mainindex++;
        }
    }

    int numboundpix = cind;
    for (int j = 0; j < numboundpix; j++)
    {
        int ii = contoury[j] * width + contourx[j];
        m_img->at<Vec3b>(ii) = Vec3b(0xff, 0xff, 0xff);

        for (int n = 0; n < 8; n++)
        {
            int x = contourx[j] + dx8[n];
            int y = contoury[j] + dy8[n];
            if ((x >= 0 && x < width) && (y >= 0 && y < height))
            {
                int ind = y * width + x;
                if (!istaken[ind])
                {
                    m_img->at<Vec3b>(ii) = Vec3b(0x00, 0x00, 0x00);
                }
            }
        }
    }
}

/*
SuperpixelSLIC::SuperpixelSLIC(Mat *img, int k) : m_img(img), m_k(k)
{
    m_strip_size = int(sqrt(k));
    m_n_channels = img->channels();

    m_width = img->size().width;
    m_height = img->size().height;
    m_img_size = m_width * m_height;

    split(*img, m_channels);

    m_cluster_size = m_img_size / m_k;
    m_cluster_side_len = m_width / m_strip_size;

    m_kseeds_l.resize(m_k);
    m_kseeds_a.resize(m_k);
    m_kseeds_b.resize(m_k);
    m_kseeds_x.resize(m_k);
    m_kseeds_y.resize(m_k);

    m_klabels.resize(m_img_size);
    m_klabels_connected.resize(m_img_size);

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

SuperpixelSLIC::~SuperpixelSLIC()
{
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
    vector<double> dists(m_img_size, FLT_MAX);

    int img_size = m_img_size;
    double offset = m_cluster_side_len + 0.5;
    double inverse_weight = 1.0 / (m_cluster_side_len / (offset * offset));
    int x1, y1, x2, y2;
    double l, a, b;
    double dist;
    double distxy;

    for (int itr = 0; itr < 10; itr++)
    {
        dists.assign(img_size, DBL_MAX);
        for (int n = 0; n < m_k; n++)
        {
            y1 = max(0.0f, m_kseeds_y[n] - offset);
            y2 = min((double)m_height, m_kseeds_y[n] + offset);
            x1 = max(0.0f, m_kseeds_x[n] - offset);
            x2 = min((double)m_width, m_kseeds_x[n] + offset);
            if (itr == 0)
                cout << '(' << x1 << ',' << y1 << ") -> (" << x2 << ',' << y2 << ')' << endl;
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
                        m_klabels[i] = n;
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
        {
            int ind = 0;
            for (int r = 0; r < m_height; r++)
            {
                for (int c = 0; c < m_width; c++)
                {
                    sigma_l[m_klabels[ind]] += m_channels[0].at<uchar>(ind);
                    sigma_a[m_klabels[ind]] += m_channels[1].at<uchar>(ind);
                    sigma_b[m_klabels[ind]] += m_channels[2].at<uchar>(ind);
                    sigma_x[m_klabels[ind]] += c;
                    sigma_y[m_klabels[ind]] += r;
                    clustersize[m_klabels[ind]] += 1.0;
                    ind++;
                }
            }
        }

        {
            for (int k = 0; k < m_k; k++)
            {
                if (clustersize[k] <= 0)
                    clustersize[k] = 1;
                // computing inverse now to multiply, than divide later
                inverses[k] = 1.0 / clustersize[k];
            }
        }

        {
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
}

inline void SuperpixelSLIC::_enforce_connectivity()
{
    const int dx4[4] = {-1, 0, 1, 0};
    const int dy4[4] = {0, -1, 0, 1};

    for (int i = 0; i < m_img_size; i++)
        m_klabels_connected[i] = -1;

    int label = 0;
    int *xvec = new int[m_img_size];
    int *yvec = new int[m_img_size];
    int oindex = 0;
    int adjlabel = 0; //adjacent label
    for (int j = 0; j < m_height; j++)
    {
        for (int k = 0; k < m_width; k++)
        {
            if (0 > m_klabels_connected[oindex])
            {
                m_klabels_connected[oindex] = label;
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
                            if (m_klabels_connected[nindex] >= 0)
                                adjlabel = m_klabels_connected[nindex];
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

                            if (0 > m_klabels_connected[nindex] && m_klabels[oindex] == m_klabels[nindex])
                            {
                                xvec[count] = x;
                                yvec[count] = y;
                                m_klabels_connected[nindex] = label;
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
                        m_klabels_connected[ind] = adjlabel;
                    }
                    label--;
                }
                label++;
            }
            oindex++;
        }
    }

    if (xvec)
        delete[] xvec;
    if (yvec)
        delete[] yvec;

    // return label;
}

inline void SuperpixelSLIC::_draw_contours()
{
    const int dx8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    const int dy8[8] = {0, -1, -1, -1, 0, 1, 1, 1};

    vector<bool> istaken(m_img_size, false);
    vector<int> contourx(m_img_size);
    vector<int> contoury(m_img_size);
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

                    //if( false == istaken[index] )//comment this to obtain internal contours
                    {
                        if (m_klabels[mainindex] != m_klabels[index])
                            np++;
                    }
                }
            }
            if (np > 1)
            {
                contourx[cind] = k;
                contoury[cind] = j;
                istaken[mainindex] = true;
                cind++;
            }
            mainindex++;
        }
    }

    int numboundpix = cind;
    for (int j = 0; j < numboundpix; j++)
    {
        int ii = contoury[j] * m_width + contourx[j];
        m_img->at<uchar>(ii) = 0xff;

        for (int n = 0; n < 8; n++)
        {
            int x = contourx[j] + dx8[n];
            int y = contoury[j] + dy8[n];
            if ((x >= 0 && x < m_width) && (y >= 0 && y < m_height))
            {
                int ind = y * m_width + x;
                if (!istaken[ind])
                    m_img->at<uchar>(ind) = 0xff;
            }
        }
    }
}
*/
