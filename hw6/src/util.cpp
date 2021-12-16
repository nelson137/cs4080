#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <stdarg.h>
#include <unistd.h>
#include <sys/stat.h>

#include <opencv2/core/mat.hpp>

#include "types.h"
#include "util.hpp"

using namespace std;
using namespace cv;

bool cstr_to_int(const char *s, int *i)
{
    return sscanf(s, "%d", i) == 1;
}

void die(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    fprintf(stderr, "error: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");

    va_end(args);
    exit(1);
}

bool file_exists(const char *path_)
{
    struct stat st;
    return stat(path_, &st) == 0 && S_ISREG(st.st_mode);
}

bool is_perfect_square(int i)
{
    if (i < 0)
        return false;
    double root = sqrt((double)i);
    return root * root == (double)i;
}

void draw_contours(Mat& img_out_contours, ClosestSeed_t *distances)
{
    // Offsets for surrounding 8 pixels
    static constexpr int dx8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    static constexpr int dy8[8] = {0, -1, -1, -1, 0, 1, 1, 1};

    int height = img_out_contours.rows;
    int width = img_out_contours.cols;
    int n_pixels = width * height;

    // Contour colors
    static const Vec3b contour_color = Vec3b(0x26, 0xfb, 0xff); // bright
    //static const Vec3b contour_color = Vec3b(0x2a, 0xe7, 0xeb); // slightly less bright

    vector<bool> is_taken(n_pixels, false);
    vector<int> contour_x;
    vector<int> contour_y;
    contour_x.reserve(n_pixels);
    contour_y.reserve(n_pixels);

    int a_i = 0;
    // For each pixel, determine if pixel is on a boundary
    for (int a_y = 0; a_y < height; a_y++)
    {
        for (int a_x = 0; a_x < width; a_x++)
        {
            // Count how many of the 8 surrounding pixels have a different label
            int np = 0;
            for (int i = 0; i < 8; i++)
            {
                int b_x = a_x + dx8[i];
                int b_y = a_y + dy8[i];
                if ((b_x >= 0 && b_x < width) && (b_y >= 0 && b_y < height))
                {
                    int b_i = b_y * width + b_x;
                    if (distances[a_i].label != distances[b_i].label)
                        np++;
                }
            }
            if (np > 1)
            {
                contour_x.push_back(a_x);
                contour_y.push_back(a_y);
                is_taken[a_i] = true;
            }
            a_i++;
        }
    }

    int n_contour_pix = contour_x.size();
    for (int c = 0; c < n_contour_pix; c++)
    {
        img_out_contours.at<Vec3b>(contour_y[c], contour_x[c]) = contour_color;
        for (int n = 0; n < 8; n++)
        {
            int x = contour_x[c] + dx8[n];
            int y = contour_y[c] + dy8[n];
            if ((x >= 0 && x < width) && (y >= 0 && y < height))
            {
                int ind = y * width + x;
                if (!is_taken[ind])
                {
                    img_out_contours.at<Vec3b>(contour_y[c], contour_x[c]) =
                        contour_color;
                }
            }
        }
    }
}
