#ifndef ROCKNROLL_SIMD_WRAPPER_HPP
#define ROCKNROLL_SIMD_WRAPPER_HPP

#include <cstdint>
#include <cstddef>
#include <iostream>

#ifdef __x86_64__
#include <immintrin.h>
#endif // __x86_64__

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "array_utils.hpp"


// Make Catch2 integration with simd types easier
template <size_t WS = 4>
struct SIMDWrapper {

#ifdef __x86_64__
    SIMDWrapper(__m128i a);
    SIMDWrapper(__m256i a);
    SIMDWrapper(__m512i a);
#endif // __x86_64__

#ifdef __ARM_NEON
    SIMDWrapper(int32x4_t a);
#endif // __ARM_NEON
    
    bool operator==(const SIMDWrapper& other) const;
    bool operator!=(const SIMDWrapper& other) const;

    alignas(64) uint8_t data[64];
    size_t word_count;
};

#ifdef __x86_64__
template <size_t N>
SIMDWrapper<N>::SIMDWrapper(__m128i u) {
    word_count = sizeof(__m128i) / N;
    _mm_store_si128((__m128i*)data, u);
}

template <size_t N>
SIMDWrapper<N>::SIMDWrapper(__m256i a) {
    word_count = sizeof(__m256i) / N;
    _mm256_store_si256((__m256i*)data, a);
}

template <size_t N>
SIMDWrapper<N>::SIMDWrapper(__m512i a) {
    word_count = sizeof(__m512i) / N;
    _mm512_store_si512((__m512i*)data, a);
}

#endif // __x86_64__


#ifdef __ARM_NEON

template <size_t N>
SIMDWrapper<N>::SIMDWrapper(int32x4_t a) {
    static_assert(N == 4, "Incompatible types");
    word_count = 4;
    vst1q_u8(data, vreinterpretq_u8_s32(a));
}

#endif // __ARM_NEON

template <size_t N>
bool SIMDWrapper<N>::operator==(const SIMDWrapper& other) const {
    if (word_count != other.word_count) {
	return false;
    }
    
    return std::equal(data, data + N * word_count, other.data);
}

template <size_t N>
bool SIMDWrapper<N>::operator!=(const SIMDWrapper& other) const {
    return !(*this == other);
}

template <size_t N>
std::ostream& operator<<(std::ostream& out, const SIMDWrapper<N>& wrapper) {
    switch (N) {
    case 8:
	print_array(out, (int64_t*)wrapper.data, wrapper.word_count);
	break;
    case 4:
	print_array(out, (int32_t*)wrapper.data, wrapper.word_count);
	break;
    case 2:
	print_array(out, (int16_t*)wrapper.data, wrapper.word_count);
	break;
    case 1:
	print_array(out, (int8_t*)wrapper.data, wrapper.word_count);
    default:	
	break;
    }
    return out;
}




#endif // ROCKNROLL_SIMD_WRAPPER_HPP
