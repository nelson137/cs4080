#include <cmath>
#include <chrono>
#include <iostream>
#include <string>

#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

#include "util.hpp"

using namespace std;
using namespace chrono;

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

void perror_die(const char *msg)
{
    fprintf(stderr, "error: ");
    perror(msg);
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

system_clock::time_point _t_start, _t_end;

void timer_start()
{
    _t_start = system_clock::now();
}

void timer_end()
{
    _t_end = system_clock::now();
}

double timer_duration()
{
    return duration_cast<microseconds>(_t_end - _t_start).count() / 1000.0;
}
