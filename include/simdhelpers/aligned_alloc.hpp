#ifndef CCL_ALIGNED_ALLOC_HPP
#define CCL_ALIGNED_ALLOC_HPP

#include <cstddef>
#include <cstdlib>

#include <simdhelpers/utils.hpp>

template <typename T>
T* aligned_new(size_t size, size_t alignment);

template <typename T>
void aligned_delete(T* ptr, size_t alignment);



// Implementations
template <typename T>
T* aligned_new(size_t size, size_t alignment) {
#ifdef _ISOC11_SOURCE
  return (T*)aligned_alloc(alignment, roundup_kpow2(size * sizeof(T), alignment));
#elif _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600
  void* ptr;
  posix_memalign(&ptr, alignment, roundup_kpow2(size * sizeof(T), alignment));
  return reinterpret_cast<T*>(ptr);

#else
  return NULL;
#endif // 
}

template <typename T>
void aligned_delete(T* data, size_t alignment) {
    free(data);
}
 

#endif // CCL_ALIGNED_ALLOC_HPP
