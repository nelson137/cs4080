#ifndef _UTIL_HPP
#define _UTIL_HPP

#include <opencv2/core/mat.hpp>

#include "types.h"

using namespace cv;

bool cstr_to_int(const char *s, int *i);

void die(const char *fmt, ...);

bool file_exists(const char *path_);

bool is_perfect_square(int i);

void draw_contours(Mat& img_out_contours, ClosestSeed_t *distances);

#endif
