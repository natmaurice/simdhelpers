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

#endif // ROCKNROLL_UTILS_HPP
