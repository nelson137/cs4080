#ifndef _GOLD_STANDARD_HPP
#define _GOLD_STANDARD_HPP

#include <vector>

#include "types.h"

using namespace std;

class SuperpixelGSLIC_Gold
{
private:
    // Input parameters
    Pixel_t *_img;
    unsigned int _width;
    unsigned int _height;
    unsigned int _n_seeds;

    // Values for calculations
    unsigned int _n_pixels;
    unsigned int _seed_size;
    unsigned int _seed_strip_size;
    unsigned int _seed_side_len;

    // "Global" memory
    vector<Seed_t> _seeds;
    vector<ClosestSeed_t> _distances;
    vector<Seed_t> _seed_sigmas;
    vector<double> _seed_pixel_counts;

    // Sub kernels
    void _init_seeds();
    bool _pixel_within_seed_2S(double x, double y, Seed_t seed);
    void _iter_once__dist();
    void _iter_once__recalc_seeds();
    void _draw();

public:
    SuperpixelGSLIC_Gold(
        Pixel_t *img,
        unsigned int width,
        unsigned int height,
        unsigned int n_seeds
    );

    // Main kernel
    void run();

    // Get pointer to seeds for comparison
    Seed_t *get_seeds();
};

#endif
