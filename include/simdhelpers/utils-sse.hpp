#ifndef ROCKNROLL_UTILS_SSE_HPP
#define ROCKNROLL_UTILS_SSE_HPP

#ifndef __SSE4_2__
#error "SSE4 not supported"
#endif // __SSE4_2__


#include <immintrin.h>
#include <emmintrin.h>

#include <ostream>
#include <cassert>

#include "simdhelpers/array_utils.hpp"


typedef int8_t  s8x16 __attribute__((vector_size (16)));
typedef int16_t s16x8 __attribute__((vector_size (16)));
typedef int32_t s32x4 __attribute__((vector_size (16)));

union int128 {
    __m128i mm;
    s8x16 i8x16;
    s16x8 i16x8;
    s32x4 i4x32;
};

// Note: operations that use the afore-defined types (s8x16, ...) have their sizes reversed compared
// to intrinsincs-based operations


namespace sse {

inline __m128i invert_16x8(__m128i in) {
    return _mm_shuffle_epi8(in, _mm_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14));
}

inline __m128i invert_32x4(__m128i in) {
    return _mm_shuffle_epi32(in, 0b00'01'10'11);
}

inline __m128i invert_16x2x4(__m128i in) {
    return invert_32x4(in);
}

inline __m128i invert_32x2x2(__m128i in) {
    return _mm_shuffle_epi32(in, 0b01'00'11'10);
}


inline __m128i vec_right_8x16(__m128i u, __m128i v) {
    return _mm_alignr_epi8(v, u, 15);
}

inline __m128i vec_right_16x8(__m128i u, __m128i v) {
    return _mm_alignr_epi8(v, u, 14);
}

inline __m128i vec_right_32x4(__m128i u, __m128i v) {
    return _mm_alignr_epi8(v, u, 12);
}


inline __m128i vec_left_8x16(__m128i u, __m128i v) {
    return _mm_alignr_epi8(v, u, 1);
}

inline __m128i vec_left_16x8(__m128i u, __m128i v) {
    return _mm_alignr_epi8(v, u, 2);
}

inline __m128i vec_left_32x4(__m128i u, __m128i v) {    
    return _mm_alignr_epi8(v, u, 4);
}


inline __m128i interleave_lo_32x4(__m128i u, __m128i v) {
    v = _mm_bslli_si128(v, 4);
    v = _mm_blend_epi16(u, v, 0b11001100);
    return v;
}

inline __m128i interleave_hi_32x4(__m128i u, __m128i v) {
    u = _mm_bsrli_si128(u, 4);
    v = _mm_blend_epi16(u, v, 0b11001100);
    return v;
}

inline __m128i filter_lo_32x4(__m128i u, __m128i v) {
    __m128i r = interleave_lo_32x4(u, v);
    r = _mm_shuffle_epi32(r, 0b11'01'10'00);
    return r;
}

inline __m128i filter_hi_32x4(__m128i u, __m128i v) {
    __m128i r = interleave_hi_32x4(u, v);
    r = _mm_shuffle_epi32(r, 0b11'01'10'00);
    return r;    
}

inline int movemask_8x16(__m128i a) {
    // Function is not very useful but is here for consistency
    return _mm_movemask_epi8(a);
}

inline int movemask_16x8(__m128i a) {
    const __m128i shuffle_mask = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
					      14, 12, 10, 8, 6, 4, 2, 0);
    __m128i mask = _mm_shuffle_epi8(a, shuffle_mask);
    int m = _mm_movemask_epi8(mask);
    return m;
}

inline int movemask_32x4(__m128i a) {
    const __m128i shuffle_mask = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
					      -1, -1, -1, -1, 12, 8, 4, 0);
    __m128i mask = _mm_shuffle_epi8(a, shuffle_mask);
    int m = _mm_movemask_epi8(mask);
    return m;
}


// [a b c d] => [d d d d]
inline __m128i bcast_last(__m128i a) {
    return _mm_shuffle_epi32(a, 0b11111111);
}

inline __m128i bcast_first(__m128i a) {
    return _mm_shuffle_epi32(a, 0b00000000);

}

inline int count_leq_32x4(__m128i a, __m128i b) {
    // a <= b
    __m128i mask = _mm_cmplt_epi32(a, b);
    int m = movemask_32x4(mask);
    return __builtin_popcount(m);
}

// Debug helpers
inline __m128i to_32x4_sse(std::initializer_list<int> input) {
    assert(input.size() == 4);

    size_t len;
    int data[4];
    build_array(data, 4, len, input);

    return _mm_set_epi32(data[3], data[2], data[1], data[0]);
}

inline void print_16x8(__m128i in) {
    alignas(16) int16_t vals[8];
    _mm_store_si128((__m128i*)vals, in);
    std::cout << vals[0];
    for (size_t i = 1; i < 16; i++) {
	std::cout << ", " << vals[i];
    }
}

inline void print_32x8(__m128i in) {
    alignas(16) int32_t vals[4];
    _mm_store_si128((__m128i*)vals, in);
    std::cout << vals[0] << ", " << vals[1] << ", " << vals[2] << ", " << vals[3] << ", " << std::endl;
}

inline void print_8x16(__m128i in) {
    alignas(16) uint8_t vals[16];
    _mm_store_si128((__m128i*)vals, in);
    std::cout << (int)vals[0];
    for (int i = 1; i < 16; i++) {
	std::cout << ", " << (int)vals[i];
    }
}

}

inline std::ostream& operator<<(std::ostream& out, __m128i in) {
    alignas(16) int32_t vals[4];
    _mm_store_si128((__m128i*)vals, in);
    out << "[" << vals[0] << ", " << vals[1] << ", " << vals[2] << ", " << vals[3] << "]";
    return out;
}

#endif // ROCKNROLL_UTILS_SSE_HPP
