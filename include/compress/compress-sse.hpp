#ifndef ROCKNROLL_COMPRESS_COMPRESS_SSE_HPP
#define ROCKNROLL_COMPRESS_COMPRESS_SSE_HPP


#include "restrict.hpp"
#include "utils-sse.hpp"


extern unsigned char LUT16x8[256][16];
extern unsigned char LUT32x4[16][16];

namespace compress {
namespace sse {

inline int get_mask_32x4(__m128i mask, __m128i& restrict shuffle_mask) {
    
    int m = ::sse::movemask_32x4(mask);
    
    shuffle_mask = ((__m128i*)LUT32x4)[m];
    return __builtin_popcount(m);
}

inline int get_mask_16x8(__m128i mask, __m128i& restrict shuffle_mask) {
    
    int m = ::sse::movemask_16x8(mask);    
    shuffle_mask = ((__m128i*)LUT16x8)[m];
    return __builtin_popcount(m);
}


inline int compress_32x4(__m128i a, __m128i mask, __m128i &res) {
    __m128i shuffle_mask;
    int count = get_mask_32x4(mask, shuffle_mask);
    res = _mm_shuffle_epi8(a, shuffle_mask);
    return count;
}

inline int compress_16x8(__m128i a, __m128i mask, __m128i &res) {
    __m128i shuffle_mask;
    int count = get_mask_16x8(mask, shuffle_mask);
    res = _mm_shuffle_epi8(a, shuffle_mask);
    return count;
}


}
}


#endif // ROCKNROLL_COMPRESS_COMPRESS_SSE_HPP
