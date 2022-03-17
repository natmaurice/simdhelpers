#ifndef SIMDHELPERS_DEFS_HPP
#define SIMDHELPERS_DEFS_HPP

<<<<<<< HEAD
=======

#if defined(__i386__) || defined(__x86_64__)
#define __HELPER_X86_ANY__
#endif // __i386__ || __x86_64__

// SIMD instructions sets


>>>>>>> d0daaa4 (Added macro for X86 detection)
// AVX256 instructions require both flags
#if defined(__AVX512F__) && defined(__AVX512VL__)
#define __HELPER_AVX256__
#endif // __AVX512__

// For consistency
#ifdef __AVX512F__
#define __HELPER_AVX512__
#endif // __AVX512F__


#endif // SIMDHELPERS_DEFS_HPP
