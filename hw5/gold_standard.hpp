#ifndef _GOLD_STANDARD_HPP
#define _GOLD_STANDARD_HPP

#include <memory>

using namespace std;

unique_ptr<unsigned char[]> gold_standard(
    unsigned char *img,
    unsigned width,
    unsigned height,
    unsigned radius
);

#endif