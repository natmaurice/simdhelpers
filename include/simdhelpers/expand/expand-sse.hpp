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

inline int expand_select_2x32x4(__m128i a, __m128i b, __m128i mask, __m128i& res) {
    __m128i shufm0, shufm1;
    int m0 = ::sse::movemask_32x4(mask);
    int m1 = (~m0) & 0xf;
    
    shufm0 = _mm_load_si128((__m128i*)(expand_LUT32x4[m0]));
    shufm1 = _mm_load_si128((__m128i*)(expand_LUT32x4[m1]));

    __m128i r0 = _mm_shuffle_epi8(a, shufm0);
    __m128i r1 = _mm_shuffle_epi8(b, shufm1);

    res = _mm_blendv_epi8(r1, r0, mask);
    
    return __builtin_popcount(m0);
}

}
}


#endif // SIMDHELPERS_EXPAND_EXPAND_SSE_HPP_
