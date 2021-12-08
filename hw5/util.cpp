#include <cstdio>
#include <cstdlib>

#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "util.hpp"

using namespace std;

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

bool file_exists(const char *path_)
{
    struct stat st;
    return stat(path_, &st) == 0 && S_ISREG(st.st_mode);
}
