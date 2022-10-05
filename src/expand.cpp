#include <simdhelpers/expand/expand-sse.hpp>

#include "simdhelpers/utils.hpp"


namespace expand {

template <size_t LUTEntries, size_t EntrySize>
inline int lut_init(unsigned char LUT[LUTEntries][EntrySize]) {

    const int BitsCount = ilog2(LUTEntries);
    const int WordBytes = EntrySize / BitsCount;
    
    for (size_t i = 0; i < LUTEntries; i++) {

	unsigned char* lut = LUT[i];

	int mask = i;

	int j = 0;
	for (int k = 0; k < BitsCount; k++) {
	    
	    for (int bi = 0; bi < WordBytes; bi++) {
		if (mask & 1) {
		    lut[k * WordBytes + bi] = j;
		    j++;
		} else {
		    lut[k * WordBytes + bi] = -1;
		}
	    }
	    mask >>= 1;
	}
    }
    return 0;
}

namespace sse {


alignas(16) unsigned char expand_LUT32x4[256 * 256][16];




}

int lut_sse_32x4 = ::expand::lut_init<16, 16>(sse::expand_LUT32x4);

}



