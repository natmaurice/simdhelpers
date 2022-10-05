#ifndef SIMDHELPERS_EXPAND_EXPAND_SSE_HPP_
#define SIMDHELPERS_EXPAND_EXPAND_SSE_HPP_

#include <immintrin.h>
#include "simdhelpers/utils-sse.hpp"


namespace expand {
namespace sse {

extern unsigned char expand_LUT32x4[256 * 256][16];

inline int get_mask_32x4(__m128i mask, __m128i& shuffle_mask) {
    int m = ::sse::movemask_32x4(mask);
    shuffle_mask = _mm_load_si128((__m128i*)(expand_LUT32x4[m]));
    return __builtin_popcount(m);
}

inline int expand_32x4(__m128i a, __m128i mask, __m128i& res) {
    __m128i shuffle_mask;
    int count = get_mask_32x4(mask, shuffle_mask);
    res = _mm_shuffle_epi8(a, shuffle_mask);
    return count;
}

}
}


#endif // SIMDHELPERS_EXPAND_EXPAND_SSE_HPP_
