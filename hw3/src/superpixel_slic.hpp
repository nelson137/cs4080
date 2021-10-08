#ifndef _SUPERPIXEL_SLIC_HPP
#define _SUPERPIXEL_SLIC_HPP

#include <memory>
#include <mutex>

#include <opencv2/core/mat.hpp>

using namespace std;
using namespace cv;

struct Seed
{
    double l, a, b, x, y;
};

template <typename T>
class LockableRow : public vector<T>
{
private:
    unique_ptr<mutex> m_mutex;

public:
    // Default constructor
    LockableRow() : vector<T>::vector()
    {
        m_mutex = unique_ptr<mutex>(new mutex{});
    }

    // Delete copy constructors
    LockableRow(const LockableRow &) = delete;
    LockableRow &operator=(const LockableRow &) = delete;

    // Move constructors
    LockableRow(LockableRow &&other) { m_mutex = move(other.m_mutex); }
    LockableRow &operator=(LockableRow &&other) { m_mutex = other.m_mutex; }

    inline lock_guard<mutex> get_lock_guard()
    {
        return lock_guard<mutex>(*m_mutex.get());
    }
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

    vector<LockableRow<int>> m_labels;
    vector<LockableRow<double>> m_dists;

public:
    SuperpixelSLIC(const Mat *const img_in, Mat *img_out, int k, int n_workers);

    SuperpixelSLIC(const SuperpixelSLIC &) = delete;
    SuperpixelSLIC(SuperpixelSLIC &&) = delete;
    SuperpixelSLIC &operator=(const SuperpixelSLIC &) = delete;

private:
    void _init_seeds();
    void _iterations();
    void _iterations_worker(int k_start, int k_end);
    void _enforce_connectivity();
    void _draw_contours();

public:
    void run();
};

struct DistChange
{
    double dist;
    int label;
    int x;
};

#endif
