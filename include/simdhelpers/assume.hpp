#ifndef CCL_ASSUME_HPP
#define CCL_ASSUME_HPP

__attribute__((always_inline))
inline void assume(bool b) {
    if (!b) {
	__builtin_unreachable();
    }
}

#endif // CCL_ASSUME_HPP
