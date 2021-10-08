#include <iostream>

#include <errno.h>
#include <semaphore.h>
#include <unistd.h>

#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "superpixel_slic.hpp"
#include "util.hpp"

#define N_ITERATIONS 10
#define SHM_KEY 12550087
#define SHM_FLAGS 0644 | IPC_CREAT | IPC_EXCL
#define COMPACTNESS 20.0

using namespace std;
using namespace cv;

SuperpixelSLIC::SuperpixelSLIC(
    const Mat *const img_in,
    Mat *img_out,
    int k,
    int n_workers)
    : m_img_in(img_in), m_img_out(img_out), m_k(k), m_n_workers(n_workers)
{
    m_runtime = 0.0;

    m_strip_size = int(sqrt(k));

    m_width = m_img_in->size().width;
    m_height = m_img_in->size().height;
    m_img_size = m_width * m_height;

    m_cluster_size = 0.5 + double(m_img_size) / double(m_k);
    m_cluster_side_len = m_width / m_strip_size;

    m_kseeds.resize(m_k);

    size_t shm_size = sizeof(sem_t *) + (sizeof(int) * m_img_size) + (sizeof(double) * m_img_size);
    if ((m_shared_mem_id = shmget(SHM_KEY, shm_size, SHM_FLAGS)) < 0)
    {
        if (errno == EEXIST)
        {
            if ((m_shared_mem_id = shmget(SHM_KEY, 0, 0644)) < 0)
                perror_die("shared memory already exists, failed to get its ID");
            if (shmctl(m_shared_mem_id, IPC_RMID, nullptr) < 0)
                perror_die("shared memory already exists, failed to remove it");
            if ((m_shared_mem_id = shmget(SHM_KEY, shm_size, SHM_FLAGS)) < 0)
                perror_die("shared memory already existed and was deleted, failed to create new one");
        }
        else
        {
            perror_die("failed to create shared memory");
        }
    }
    if ((m_shared_mem_buf = shmat(m_shared_mem_id, nullptr, 0)) == (void *)-1)
        perror_die("failed to attach shared memory");

    // Setup mutex for seeds in shared memory
    m_mutex = (sem_t *)m_shared_mem_buf;
    if (sem_init(m_mutex, 1, 1) < 0)
        die("failed to create semaphore: m_mutex");

    m_labels = (int *)(m_mutex + 1);
    m_dists = (double *)(m_labels + m_img_size);

    timer_start();
    _init_seeds();
    timer_end();
    cout << "seed init time: " << timer_duration() << " ms" << endl;
}

SuperpixelSLIC::~SuperpixelSLIC()
{
    sem_destroy(m_mutex);
    shmdt(m_shared_mem_buf);
    shmctl(m_shared_mem_id, IPC_RMID, nullptr);
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

            Vec3b point = m_img_in->at<Vec3b>(Y, X);
            m_kseeds[i] = {(double)point(0),
                           (double)point(1),
                           (double)point(2),
                           (double)X,
                           (double)Y};

            i++;
        }
    }
}

void SuperpixelSLIC::run()
{
    double t;

    timer_start();
    _iterations();
    timer_end();
    t = timer_duration();
    cout << "iterations time: " << t << " ms" << endl;
    m_runtime += t;

    timer_start();
    _enforce_connectivity();
    timer_end();
    t = timer_duration();
    cout << "enforce connectivity time: " << t << " ms" << endl;
    m_runtime += t;

    timer_start();
    _draw_contours();
    timer_end();
    t = timer_duration();
    cout << "drawing contours time: " << t << " ms" << endl;
    m_runtime += t;

    cout << "total runtime: " << m_runtime << " ms" << endl;
}

inline void SuperpixelSLIC::_iterations()
{
    vector<double> clustersize(m_k, 0);
    vector<double> inverses(m_k, 0);
    vector<Seed> sigmas(m_k, {0.0, 0.0, 0.0, 0.0, 0.0});
    // vector<double> m_dists(m_img_size, DBL_MAX);

    double inverse_weight = double(m_cluster_side_len * m_cluster_side_len) / double(m_cluster_side_len);

    // Must be non-zero so the main proc doesn't _exit() when m_n_workers==1
    pid_t pid = 1;

    vector<pid_t> worker_pids;
    worker_pids.reserve(m_n_workers - 1);
    int clusters_per_worker = m_k / m_n_workers;

    int k_start = 0, k_end = clusters_per_worker;

    for (int itr = 0; itr < N_ITERATIONS; itr++)
    {
        // m_dists.assign(m_img_size, DBL_MAX);
        for (int i = 0; i < m_img_size; i++)
            m_dists[i] = DBL_MAX;
        worker_pids.clear();

        // Create workers with fork() and start jobs
        // Note: m_n_workers-1 workers are forked because main thread is a
        // worker too.
        int i;
        for (i = 1; i < m_n_workers; i++)
        {
            pid = fork();
            if (pid < 0)
            {
                // Error
                die("failed to fork");
            }
            else if (pid > 0)
            {
                // Parent
                worker_pids.push_back(pid);
            }
            else
            {
                // Child
                k_start = i * clusters_per_worker;
                k_end = k_start + clusters_per_worker;
                break;
            }
        }

        vector<DistChange> dist_changes(m_width);

        // Do work, different set of `k`s for each worker
        for (int k = k_start; k < k_end; k++)
        {
            Seed &seed = m_kseeds[k];

            int y1 = max(0.0, seed.y - m_cluster_side_len);
            int y2 = min((double)m_height, seed.y + m_cluster_side_len);
            int x1 = max(0.0, seed.x - m_cluster_side_len);
            int x2 = min((double)m_width, seed.x + m_cluster_side_len);

            for (int y = y1; y < y2; y++)
            {
                for (int x = x1; x < x2; x++)
                {
                    int i = y * m_width + x;
                    Vec3b point = m_img_in->at<Vec3b>(i);
                    double l = point(0);
                    double a = point(1);
                    double b = point(2);
                    double dist = (l - seed.l) * (l - seed.l) +
                                  (a - seed.a) * (a - seed.a) +
                                  (b - seed.b) * (b - seed.b);
                    double distxy = (x - seed.x) * (x - seed.x) +
                                    (y - seed.y) * (y - seed.y);

                    //------------------------------------------------------------------------
                    dist += distxy * inverse_weight;
                    //------------------------------------------------------------------------

                    dist_changes[x] = {dist, k, i};
                }

                sem_wait(m_mutex);
                for (int x = x1; x < x2; x++)
                {
                    DistChange dc = dist_changes[x];
                    if (dc.dist < m_dists[dc.i])
                    {
                        m_dists[dc.i] = dc.dist;
                        m_labels[dc.i] = dc.label;
                    }
                }
                sem_post(m_mutex);
            }
        }

        // Kill all _forked_ workers, but not the main one
        // Note: fork() sets `pid` 0 in child procs, but the parent gets the
        // created pid, so the parent is the only one with a non-zero value for
        // `pid`, and, thus, the only one that won't _exit().
        if (pid == 0)
            _exit(0);

        // Wait for all workers to finish
        // Note: should only happen in main because of above _exit().
        for (const pid_t &pid : worker_pids)
        {
            int code;
            waitpid(pid, &code, 0);
        }

        //-----------------------------------------------------------------
        // Recalculate the centroid and store in the seed values
        //-----------------------------------------------------------------
        //instead of reassigning memory on each iteration, just reset.

        sigmas.assign(m_k, {0.0, 0.0, 0.0, 0.0, 0.0});
        clustersize.assign(m_k, 0);
        int ind = 0;
        for (int r = 0; r < m_height; r++)
        {
            for (int c = 0; c < m_width; c++)
            {
                Seed &sigma = sigmas[m_labels[ind]];
                Vec3b point = m_img_in->at<Vec3b>(ind);
                sigma.l += point(0);
                sigma.a += point(1);
                sigma.b += point(2);
                sigma.x += c;
                sigma.y += r;
                clustersize[m_labels[ind]] += 1.0;
                ind++;
            }
        }

        {
            for (int k = 0; k < m_k; k++)
            {
                if (clustersize[k] <= 0)
                    clustersize[k] = 1;
                // computing inverse now to multiply later
                inverses[k] = 1.0 / clustersize[k];
            }
        }

        {
            for (int k = 0; k < m_k; k++)
            {
                Seed &sigma = sigmas[k];
                double inv = inverses[k];
                m_kseeds[k] = {
                    sigma.l * inv,
                    sigma.a * inv,
                    sigma.b * inv,
                    sigma.x * inv,
                    sigma.y * inv};
            }
        }
    }
}

inline void SuperpixelSLIC::_enforce_connectivity()
{
    const int dx4[4] = {-1, 0, 1, 0};
    const int dy4[4] = {0, -1, 0, 1};

    int *new_labels = new int[m_img_size];
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

    memcpy(m_labels, new_labels, sizeof(int) * m_img_size);

    delete[] new_labels;
    delete[] xvec;
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
        m_img_out->at<Vec3b>(contour_y[j], contour_x[j]) = color_ffffff;
        for (int n = 0; n < 8; n++)
        {
            int x = contour_x[j] + dx8[n];
            int y = contour_y[j] + dy8[n];
            if ((x >= 0 && x < m_width) && (y >= 0 && y < m_height))
            {
                int ind = y * m_width + x;
                if (!is_taken[ind])
                {
                    m_img_out->at<Vec3b>(contour_y[j], contour_x[j]) = color_000000;
                }
            }
        }
    }
}
