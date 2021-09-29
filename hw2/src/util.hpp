#ifndef _UTIL_HPP
#define _UTIL_HPP

bool cstr_to_int(const char *s, int *i);

void die(const char *fmt, ...);

bool file_exists(const char *path_);

bool is_perfect_square(int i);

/**
 * Save the current time as the start of the duration.
 * Used for timing segments of code.
 * WARNING: These timer_* functions are *not* thread-safe.
 */
void timer_start();

/**
 * Save the current time as the end of the duration.
 * Used for timing segments of code.
 * WARNING: These timer_* functions are *not* thread-safe.
 */
void timer_end();

/**
 * Return the time difference between timer_start() and timer_end()
 * with microsecond precision in milliseconds.
 * Used for timing segments of code.
 * WARNING: These timer_* functions are *not* thread-safe.
 */
double timer_duration();

#endif
