#ifndef _UTIL_HPP
#define _UTIL_HPP

bool cstr_to_int(const char *s, int *i);

void die(const char *fmt, ...);

bool file_exists(const char *path_);

bool is_perfect_square(int i);

#endif
