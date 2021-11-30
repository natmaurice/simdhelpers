#ifndef ROCKNROLL_TESTS_TESTS_UTILS_HPP
#define ROCKNROLL_TESTS_TESTS_UTILS_HPP

#include <cstddef>
#include <cstdint>
#include <utility>
#include <iostream>


template <typename T>
void print_array(std::ostream& out, const T* arr, size_t len) {

    out << "[";
    if (len > 0) {
	if constexpr (std::is_signed<T>()) {
	    out << static_cast<int64_t>(arr[0]);
	} else if (std::is_unsigned<T>()) {
	    out << static_cast<uint64_t>(arr[0]);
	} else {
	    out << arr[0];
	}
    }    
    for (size_t i = 1; i < len; i++) {
	// Ensure that char types will be printed as integer
	if constexpr (std::is_signed<T>()) {
	    out << ", " << static_cast<int64_t>(arr[i]);
	} else if (std::is_unsigned<T>()) {
	    out << ", " << static_cast<uint64_t>(arr[i]);
	} else {	
	    out <<  ", " << arr[i];
	}
    }
    out << "]";
}

template <typename T>
void print_intervals(std::ostream& out, const T* arr, size_t len) {
    T beg = 0, end = 0;
    for (size_t i = 0; i < len; i += 2) {
	beg = arr[i];
	for (size_t j = end; j < beg; j++) {
	    out << "_";
	}
	end = arr[i + 1];
	for (size_t j = beg; j < end; j++) {
	    out << "x";
	}
    }
}

double calc_pmerge(size_t len0, size_t len1, size_t union_count);
double calc_pmerge(double avglen0, double avglen1, double pmerge);

template <typename T, typename U>
bool build_array(T* arr, size_t max_len, size_t &len, std::initializer_list<U> elements) {
    size_t i = 0;
    if (elements.size() > max_len) {
	return false;
    }
    for (const auto& elem: elements) {
	arr[i++] = elem;
    }
    len = i;
    return true;
}

// Convert an initializer list of pairs [(a, b), (c, d), ...] into a C-style array [a, b, c, d, ...]
template <typename T, typename U>
bool build_array_from_pairs(T* arr, size_t max_len, size_t &len, std::initializer_list<std::pair<U, U>> elements) {
    size_t i = 0;
    if (elements.size() * 2 > max_len) {
	return false;
    }
    for (const auto& elem: elements) {
	arr[i++] = elem.first;
	arr[i++] = elem.second;
    }
    len = i;
    return true;
}

// This structure is meant to be used with Catch, when comparing arrays.
// std::equals() is the C++ way of comparing arrays. However, catch does not print the arrays
// automatically with it (because it can't).
// This structs is a wrapper around a C array that provides both the comparaison operator and the
// equality operator.
// Note: It only keep references on the array. It should therefore not be used after free.
template <typename T>
struct ArrayWrapper {

    ArrayWrapper(T* arr, size_t len);

    bool operator==(const ArrayWrapper<T>& other) const;
    bool operator!=(const ArrayWrapper<T>& other) const;
    
    T* arr = nullptr;
    size_t len = 0;    
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const ArrayWrapper<T>& checker);

// ==================================================
// Implementations
// ==================================================

template <typename T>
ArrayWrapper<T>::ArrayWrapper(T* arr, size_t len) :
    arr(arr), len(len) {
}

template <typename T>
bool ArrayWrapper<T>::operator==(const ArrayWrapper<T>& other) const {
    if (this->len != other.len) {
	return false;
    }
    return std::equal(this->arr, this->arr + this->len, other.arr);
}

template <typename T>
bool ArrayWrapper<T>::operator!=(const ArrayWrapper<T>& other) const {
    return !(*this == other);
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const ArrayWrapper<T>& checker) {
    print_array(out, checker.arr, checker.len);
    return out;
}

#endif // ROCKNROLL_TESTS_TESTS_UTILS_HPP
