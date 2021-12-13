#ifndef _TYPES_H
#define _TYPES_H

typedef struct {
    unsigned char l;
    unsigned char a;
    unsigned char b;
} Pixel_t;

typedef struct {
    double x;
    double y;
    double l;
    double a;
    double b;
} Seed_t;

typedef struct {
    double dist;
    unsigned int label;
} ClosestSeed_t;

#endif
