#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

vector<system_clock::time_point> _t_times;

void timer_start()
{
    auto it = _t_times.emplace(_t_times.cend());
    *it = system_clock::now();
}

double timer_end()
{
    system_clock::time_point end = system_clock::now();
    if (_t_times.size() == 0)
        throw runtime_error(
            "called timer_end() without a previous call of timer_start()");
    system_clock::time_point start = _t_times.back();
    _t_times.pop_back();
    return duration_cast<microseconds>(end - start).count() / 1000.0;
}
