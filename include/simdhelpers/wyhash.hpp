#ifndef SIMDHELPERS_WYHASH_HPP
#define SIMDHELPERS_WYHASH_HPP

#include <cstdint>
#include <cstddef>

static inline uint64_t _mum(uint64_t A, uint64_t B) {
    __uint128_t c = (__uint128_t)A * B;
    return (c >> 64) ^ c;
}

static inline uint64_t wyrand(uint64_t& seed) {
    seed += 0xa0761d6478bd642full;
    return _mum(seed ^ 0xe7037ed1a0b428dbull, seed);
}

static inline uint64_t wyhash(uint64_t seed) {
    return wyrand(seed);
}

#endif // SIMDHELPERS_WYHASH_HPP
