#ifndef ROCKNROLL_UTILS_HPP
#define ROCKNROLL_UTILS_HPP

#include <cstdint>
#include <cstddef>
#include <string>


template <typename T>
constexpr T POW2(T x) {
    return x*x;
}

std::string to_uppercase(const std::string& str);


// Round up to a multiple of a know power of up
// n: number to be rounded up
// m: must be a power of two
constexpr size_t roundup_kpow2(size_t n, int m) {
    return (n + (m - 1)) & -m;
}

constexpr int roundup_kn(int n, int m) {
    int remainder = n % m;
    remainder = remainder == 0 ? m : remainder; // don't round up if n already a multiple of m
    return n + m - remainder;
}

constexpr uint32_t ilog2(uint32_t n) {
    return 31 - __builtin_clz(n);
}


inline int isqrt(int n) {
    int i = 0;

    // (Very) Naive approach. This shouldn't be used for performance critical functions
    do {
	i++;
    } while (i * i < n);
    
    return i - 1;
}


#endif // ROCKNROLL_UTILS_HPP
