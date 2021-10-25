#ifndef _UTIL_HPP
#define _UTIL_HPP

bool cstr_to_int(const char *s, int *i);

void die(const char *fmt, ...);

void perror_die(const char *msg);

bool file_exists(const char *path_);

bool is_perfect_square(int i);

/**
 * Save the current time as the start of the duration.
 * Used for timing segments of code.
 * WARNING: The timer_* functions are *not* thread-safe.
 */
void timer_start();

/**
 * Return the time difference between the previous call to timer_start() and now
 * with precision in milliseconds.
 * Used for timing segments of code.
 * WARNING: The timer_* functions are *not* thread-safe.
 */
double timer_end();

#endif
