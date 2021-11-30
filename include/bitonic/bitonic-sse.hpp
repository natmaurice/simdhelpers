#ifndef ROCKNROLL_BITONIC_HPP
#define ROCKNROLL_BITONIC_HPP

#include "utils-sse.hpp"

enum class MemLayout {
    SEP,
    MIX
};


namespace bitonic {
namespace sse {


// [32 bits * 4]
inline __m128i bitonic_sse_32x4(__m128i data);
inline int bitonic_sse_32x2x2(__m128i& lo, __m128i& hi);


__m128i bitonic_sse_32x4(__m128i data) {
    // [a b c d] -> [a b d c]
    data = _mm_shuffle_epi32(data, 0b10'11'01'00);
    
    __m128i hi = _mm_blend_epi16(data, _mm_set1_epi16(0), 0xF0);
    __m128i lo = _mm_blend_epi16(_mm_set1_epi16(0), data , 0xF0);

    __m128i mask = _mm_cmpgt_epi32(hi, lo);

    __m128i hi1 = _mm_blendv_epi8(hi, lo, mask);
    __m128i lo1 = _mm_blendv_epi8(lo, hi, mask);

    hi = _mm_unpackhi_epi32(hi1, lo1);
    lo = _mm_unpacklo_epi32(hi1, lo1);

    mask = _mm_cmpgt_epi32(hi, lo);
    hi1 = _mm_blendv_epi8(hi, lo, mask);
    lo1 = _mm_blendv_epi8(lo, hi, mask);

    // Re-order data    
    hi = _mm_unpackhi_epi32(hi1, lo1);
    lo = _mm_unpacklo_epi32(hi1, lo1);
    
    hi = _mm_bslli_si128(hi, 2);

    data = _mm_or_epi32(lo, hi);
    return data;
}



int bitonic_sse_32x2x2(__m128i& lo, __m128i& hi) {
    
    // First network
    // 1. Invert lo
    hi = ::sse::invert_32x4(hi);
    
    // 2. First network
    __m128i mask = _mm_cmpgt_epi32(lo, hi);
    __m128i lo0 = _mm_blendv_epi8(lo, hi, mask);
    __m128i hi0 = _mm_blendv_epi8(hi, lo, mask);

    // Optionnal: We count the number of elements from hi that moved into lo
    int moved = __builtin_popcount(::sse::movemask_32x4(mask));
    
    // 2nd net
    __m128i lo1 = _mm_unpacklo_epi32(lo0, hi0);
    __m128i hi1 = _mm_unpackhi_epi32(lo0, hi0);

    mask = _mm_cmpgt_epi32(lo1, hi1);
    hi0 = _mm_blendv_epi8(lo1, hi1, mask);
    lo0 = _mm_blendv_epi8(hi1, lo1, mask);
    
    // 3rd network
    lo1 = _mm_unpacklo_epi32(lo0, hi0);
    hi1 = _mm_unpackhi_epi32(lo0, hi0);

    mask = _mm_cmpgt_epi32(lo1, hi1);
    hi0 = _mm_blendv_epi8(lo1, hi1, mask);
    lo0 = _mm_blendv_epi8(hi1, lo1, mask);

    // Write back
    lo = _mm_unpacklo_epi32(lo0, hi0);
    hi = _mm_unpackhi_epi32(lo0, hi0);

    lo = ::sse::invert_32x4(lo);
    hi = ::sse::invert_32x4(hi);

    return moved;
}

inline void bitonic_dual_sse_32x2x2(__m128i& lo, __m128i& hi) {
    __m128i lo0, hi0;
    __m128i lo1, hi1;
    __m128i lo2, hi2;
    __m128i lo3, hi3;

    //std::cout << "=== bitonic_dual_sse_32x2x2 ===" << std::endl;
    
    //std::cout << "lo = " << lo << ", hi = " << hi << std::endl;
    
    lo0 = _mm_min_epi32(lo, hi);
    hi0 = _mm_max_epi32(lo, hi);

    //std::cout << "mins = " << lo0 << ", maxs = " << hi0 << "\n\n";

    lo1 = _mm_shuffle_epi32(lo0, 0b01'00'11'10);
    hi1 = _mm_shuffle_epi32(hi0, 0b01'00'11'10);

    //std::cout << "lo1 = " << lo1 << ", hi1 = " << hi1 << std::endl;
    
    lo2 = _mm_min_epi32(lo0, lo1);
    lo3 = _mm_max_epi32(lo0, lo1);
    lo = _mm_blend_epi16(lo2, lo3, 0b11110000);

    //std::cout << "lo_min = " << lo2 << ", lo_max = " << lo3 << "\n";
    //std::cout << "lo = " << lo << "\n\n";
    
    hi2 = _mm_min_epi32(hi0, hi1);
    hi3 = _mm_max_epi32(hi0, hi1);
    hi = _mm_blend_epi16(hi2, hi3, 0b00001111);
    //std::cout << "hi_min = " << hi2 << ", hi_max = " << hi3 << "\n";
    //std::cout << "hi = " << hi << "\n\n";
}


inline void bitonic_mix_sse_16x2x4(__m128i& lo, __m128i& hi) {
    __m128i lo0, hi0;
    __m128i lo1, hi1;

    //std::cout << "--- bitonic_mix_sse_16x2x4 --- \n";

    //std::cout << "(0) lo = " << SIMDWrapper<2>(lo) << ", hi = " << SIMDWrapper<2>(hi) << "\n";
    
    hi = _mm_shuffle_epi8(hi, _mm_set_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12));

    //std::cout << "(1) lo = " << SIMDWrapper<2>(lo) << ", hi = " << SIMDWrapper<2>(hi) << "\n";
    
    __m128i mask = _mm_cmpgt_epi16(lo, hi);
    lo0 = _mm_blendv_epi8(lo, hi, mask);
    hi0 = _mm_blendv_epi8(hi, lo, mask);

    //std::cout << "(2) lo0 = " << SIMDWrapper<2>(lo0) << ", hi0 = " << SIMDWrapper<2>(hi0) << "\n";
    
    lo1 = _mm_unpacklo_epi16(lo0, hi0);
    hi1 = _mm_unpackhi_epi16(lo0, hi0);

    //std::cout << "(2.5) lo1 = " << SIMDWrapper<2>(lo1) << ", hi1 = " << SIMDWrapper<2>(hi1) << "\n";
    
    mask = _mm_cmpgt_epi16(lo1, hi1);
    lo0 = _mm_blendv_epi8(lo1, hi1, mask);
    hi0 = _mm_blendv_epi8(hi1, lo1, mask);

    //std::cout << "(3) lo0 = " << SIMDWrapper<2>(lo0) << ", hi0 = " << SIMDWrapper<2>(hi0) << "\n";
    
    lo1 = _mm_unpacklo_epi16(lo0, hi0);
    hi1 = _mm_unpackhi_epi16(lo0, hi0);

    //std::cout << "(3.5) lo1 = " << SIMDWrapper<2>(lo1) << ", hi1 = " << SIMDWrapper<2>(hi1) << "\n";

    mask = _mm_cmpgt_epi16(lo1, hi1);
    lo0 = _mm_blendv_epi8(lo1, hi1, mask);
    hi0 = _mm_blendv_epi8(hi1, lo1, mask);

    //std::cout << "(4) lo0 = " << SIMDWrapper<2>(lo0) << ", hi0 = " << SIMDWrapper<2>(hi0) << "\n";
    
    // Re-arrange data
    lo1 = _mm_unpacklo_epi16(lo0, hi0);
    hi1 = _mm_unpackhi_epi16(lo0, hi0);

    //std::cout << "(4.5) lo1 = " << SIMDWrapper<2>(lo1) << ", hi1 = " << SIMDWrapper<2>(hi1) << "\n";
    
    // Re-mix data
    lo = _mm_unpacklo_epi16(lo1, hi1);
    hi = _mm_unpackhi_epi16(lo1, hi1);
}

} // namespace sse
} // namespace bitonic

#endif // ROCKNROLL_BITONIC_HPP
