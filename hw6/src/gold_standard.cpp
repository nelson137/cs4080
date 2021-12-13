#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <vector>

#include "gold_standard.hpp"

#define ITERATIONS 10

SuperpixelGSLIC_Gold::SuperpixelGSLIC_Gold(
    Pixel_t *img,
    unsigned int width,
    unsigned int height,
    unsigned int n_seeds)
    : _img(img), _width(width), _height(height), _n_seeds(n_seeds)
{
    _n_pixels = width * height;
    _seed_size = 0.5 + _n_pixels / n_seeds;
    _seed_strip_size = (int) sqrt((double) n_seeds);
    _seed_side_len = width / _seed_strip_size;

    _seeds.reserve(n_seeds);
    _distances.reserve(_n_pixels);
    _seed_sigmas.reserve(n_seeds);
    _seed_pixel_counts.reserve(n_seeds);
}

void SuperpixelGSLIC_Gold::_init_seeds()
{
    const unsigned int x_off = _seed_side_len / 2;
    const unsigned int y_off = _seed_side_len / 2;

    const unsigned int err = _width - _seed_side_len * _seed_strip_size;
    const double err_per_strip = ((double) err / (double) _seed_strip_size);

    unsigned int seed_i = 0;

    for (unsigned int y = 0; y < _seed_strip_size; ++y)
    {
        unsigned int y_err = y * err_per_strip;
        unsigned int Y = y*_seed_side_len + y_off + y_err;
        if (Y >= _height)
            continue;

        for (unsigned int x = 0; x < _seed_strip_size; ++x)
        {
            unsigned int x_err = x * err_per_strip;
            unsigned int X = x*_seed_side_len + x_off + x_err;
            if (X >= _width)
                continue;

            Pixel_t pixel = _img[Y * _width + X];
            _seeds[seed_i++] = {
                .x = (double) X,
                .y = (double) Y,
                .l = (double) pixel.l,
                .a = (double) pixel.a,
                .b = (double) pixel.b,
            };
        }
    }
}

bool SuperpixelGSLIC_Gold::_pixel_within_seed_2S(
    double x,
    double y,
    Seed_t seed)
{
    double y_min = max(0.0,             y - (double)_seed_side_len);
    double y_max = min((double)_height, y + (double)_seed_side_len);
    double x_min = max(0.0,             x - (double)_seed_side_len);
    double x_max = min((double)_width,  x + (double)_seed_side_len);
    return y_min <= seed.y && seed.y < y_max
        && x_min <= seed.x && seed.x < x_max;
}

void SuperpixelGSLIC_Gold::_iter_once__dist()
{
    const double inverse_weight = (double) _seed_side_len;

    for (unsigned int i = 0; i < _n_pixels; ++i)
    {
        Pixel_t pixel = _img[i];
        double X = (double) (i % _width);
        // Use int division to truncate, then cast to double for later calculations
        double Y = (double) (i / _width);
        double L = (double) pixel.l;
        double A = (double) pixel.a;
        double B = (double) pixel.b;

        ClosestSeed_t closest_seed = { .dist = DBL_MAX, .label = UINT_MAX };

        for (unsigned int s = 0; s < _n_seeds; ++s)
        {
            Seed_t seed = _seeds[s];
            if (_pixel_within_seed_2S(X, Y, seed))
            {
                double dist = (X - seed.x) * (X - seed.x) +
                              (Y - seed.y) * (Y - seed.y);
                dist *= inverse_weight;
                dist += (L - seed.l) * (L - seed.l) +
                        (A - seed.a) * (A - seed.a) +
                        (B - seed.b) * (B - seed.b);
                if (dist < closest_seed.dist)
                    closest_seed = { .dist = dist, .label = s };
            }
        }

        _distances[i] = closest_seed;
    }
}

void SuperpixelGSLIC_Gold::_iter_once__recalc_seeds()
{
    for (unsigned int s = 0; s < _n_seeds; ++s)
    {
        _seed_sigmas[s] =
            { .x = 0.0, .y = 0.0, .l = 0.0, .a = 0.0, .b = 0.0 };
        _seed_pixel_counts[s] = 0.0;

        unsigned int pixel_i = 0;
        for (unsigned int y = 0; y < _height; ++y)
        {
            for (unsigned int x = 0; x < _width; ++x)
            {
                if (_distances[pixel_i].label == s)
                {
                    _seed_pixel_counts[s] += 1.0;
                    Seed_t& sigma = _seed_sigmas[s];
                    Pixel_t pixel = _img[pixel_i];
                    sigma.x += x;
                    sigma.y += y;
                    sigma.l += (double) pixel.l;
                    sigma.a += (double) pixel.a;
                    sigma.b += (double) pixel.b;
                }
                ++pixel_i;
            }
        }

        double s_size = _seed_pixel_counts[s];
        if (s_size <= 0.0)
            s_size = 1.0;

        Seed_t sigma = _seed_sigmas[s];
        _seeds[s] = {
            .x = sigma.x / s_size,
            .y = sigma.y / s_size,
            .l = sigma.l / s_size,
            .a = sigma.a / s_size,
            .b = sigma.b / s_size
        };
    }
}

void SuperpixelGSLIC_Gold::_draw()
{
    for (unsigned int i = 0; i < _n_pixels; ++i)
    {
        unsigned int s = _distances[i].label;
        if (s < _n_seeds)
        {
            Seed_t seed = _seeds[s];
            _img[i] = {
                .l = (unsigned char) seed.l,
                .a = (unsigned char) seed.a,
                .b = (unsigned char) seed.b
            };
        }
    }
}

void SuperpixelGSLIC_Gold::run()
{
    _init_seeds();

    for (unsigned int iter = 0; iter < ITERATIONS; ++iter)
    {
        _iter_once__dist();
        _iter_once__recalc_seeds();
    }

    _draw();
}

Seed_t *SuperpixelGSLIC_Gold::get_seeds()
{
    return &_seeds[0];
}
