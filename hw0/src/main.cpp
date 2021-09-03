#include <chrono>
#include <iostream>
#include <map>

#include <unistd.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>

using namespace std;
using namespace chrono;

using namespace cv;

const char *ARG0 = nullptr;

system_clock::time_point _t_start, _t_end;

/**
 * Save the current time as the start of the duration.
 * Used for timing segments of code.
 * WARNING: These timer_* functions are *not* thread-safe.
 */
void timer_start() {
    _t_start = system_clock::now();
}

/**
 * Save the current time as the end of the duration.
 * Used for timing segments of code.
 * WARNING: These timer_* functions are *not* thread-safe.
 */
void timer_end() {
    _t_end = system_clock::now();
}

/**
 * Return the time difference between timer_start() and timer_end()
 * with microsecond precision in milliseconds.
 * Used for timing segments of code.
 * WARNING: These timer_* functions are *not* thread-safe.
 */
double timer_duration() {
    return duration_cast<microseconds>(_t_end - _t_start).count() / 1000.0;
}

/**
 * Print the usage for this program then exit with the given code (default=0).
 */
void help(int code = 0) {
    ostream &s = code ? cerr : cout;
    s << "Usage: " << ARG0 << " IMG" << endl;
    exit(code);
}

int main(int argc, char **argv) {
    ARG0 = argv[0];

    if (argc != 2)
        help(1);

    const string& filename = argv[1];
    Mat img_rgb, img_lab;
    int w, h;
    double t_load = -1, t_convert = -1, t_hist_comp = -1;

    // Load image

    timer_start();
    img_rgb = imread(filename, IMREAD_COLOR);
    timer_end();
    t_load = timer_duration();
    w = img_rgb.cols, h = img_rgb.rows;

    // Convert image from RGB to CIELAB

    timer_start();
    cvtColor(img_rgb, img_lab, COLOR_RGB2Lab);
    timer_end();
    t_convert = timer_duration();

    // Calculate histograms

    vector<Mat> imgs = { img_lab };
    Mat hist_l, hist_a, hist_b;
    Mat *hists[] = { &hist_l, &hist_a, &hist_b };
    int hist_size = 256;
    vector<float> hist_range = { 0.0, 256.0 };
    timer_start();
    for (int i=0; i<3; i++) {
        calcHist(
            imgs, vector<int>{i}, Mat{},
            *hists[i], vector<int>{hist_size}, hist_range
        );
    }
    timer_end();
    t_hist_comp = timer_duration();

    // Print summary

    cout << "              Filename : " << filename << endl;
    cout << "            Dimensions : " << w << " x " << h << endl;
    cout << "             Load time : " << t_load << " ms" << endl;
    cout << "          Convert time : " << t_convert << " ms" << endl;
    cout << "Compute histogram time : " << t_hist_comp << " ms" << endl;

    cout << "    Histogram data (L) :";
    for (int i=0; i<hist_size; i++)
        cout << " (" << i << ',' << hists[0]->at<float>(i) << ')';
    cout << endl;

    cout << "    Histogram data (A) :";
    for (int i=0; i<hist_size; i++)
        cout << " (" << i << ',' << hists[1]->at<float>(i) << ')';
    cout << endl;

    cout << "    Histogram data (B) :";
    for (int i=0; i<hist_size; i++)
        cout << " (" << i << ',' << hists[2]->at<float>(i) << ')';
    cout << endl;

    return 0;
}
