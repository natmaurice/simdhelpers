#include "utils.hpp"

#include <algorithm>



std::string to_uppercase(const std::string& str) {
    std::string dest = str;
    std::transform(dest.begin(), dest.end(), dest.begin(), [](unsigned char c) {
	return std::toupper(c);
    });
    return dest;
}
