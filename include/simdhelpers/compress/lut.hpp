#ifndef SIMDHELPERS_COMPRESS_LUT_HPP
#define SIMDHELPERS_COMPRESS_LUT_HPP

#include <cstddef>

#include "simdhelpers/utils.hpp"


namespace compress {

template <size_t LUTEntries, size_t EntrySize>
inline int lut_init(unsigned char LUT[LUTEntries][EntrySize]) {

    const int BitsCount = ilog2(LUTEntries);
    for (int i = 0; i < LUTEntries; i++) {
	unsigned char* lut = LUT[i];
	
	int mask = i;
	int j = 0;
	for (int k = 0; k < BitsCount; k++) {
	    lut[j] = k;
	    if (mask & 1) {
		j++;
	    }
	    mask >>= 1;	    
	}
	for (; j < EntrySize; j++) {
	    lut[j] = 255;
	}
    }
    return 0;
}

}

#endif // SIMDHELPRES_COMPRESS_LUT_HPP
