#ifdef __x86_64__

#include <catch2/catch.hpp>

#include <immintrin.h>
#include <emmintrin.h>

#include <simdhelpers/utils-sse.hpp>
#include <simdhelpers/compress/compress-sse.hpp>
#include <simdhelpers/array_utils.hpp>
#include <simdhelpers/simd-wrapper.hpp>
#include <simdhelpers/expand/expand-sse.hpp>

#ifdef __SSE4_2__
using SSEFun1 = std::function<__m128i(__m128i)>;
using SSEFun2 = std::function<__m128i(__m128i, __m128i)>;

using BitonicFun = std::function<int(__m128i&, __m128i&)>;
using BitonicFunV = std::function<void(__m128i&, __m128i&)>;
#endif // __SSE4_2__


TEST_CASE("SSE - invert_16x8") {
    __m128i u = _mm_set_epi16(1008, 1007, 1006, 1005, 1004, 1003, 1002, 1001);
    auto invert = sse::invert_16x8(u);
    auto expected = _mm_set_epi16(1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008);

    REQUIRE(SIMDWrapper<4>(invert) == SIMDWrapper<4>(expected));
}

TEST_CASE("SSE - vec_right_16x8()") {
    __m128i u = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
    __m128i v = _mm_set_epi16(16, 15, 14, 13, 12, 11, 10, 9);
    
    auto right = sse::vec_right_16x8(u, v);

    auto expected = _mm_set_epi16(15, 14, 13, 12, 11, 10, 9, 8);

    REQUIRE(SIMDWrapper<2>(right) == SIMDWrapper<2>(expected));
}

TEST_CASE("SSE - vec_right_32x4()") {
    __m128i u = _mm_set_epi32(4, 3, 2, 1);
    __m128i v = _mm_set_epi32(8, 7, 6, 5);

    auto right = sse::vec_right_32x4(u, v);

    auto expected = _mm_set_epi32(7, 6, 5, 4);

    REQUIRE(SIMDWrapper(right) == SIMDWrapper(expected));
}

TEST_CASE("SSE - vec_left_16x8()") {
    __m128i u = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
    __m128i v = _mm_set_epi16(16, 15, 14, 13, 12, 11, 10, 9);
    
    auto left = sse::vec_left_16x8(u, v);

    auto expected = _mm_set_epi16(9, 8, 7, 6, 5, 4, 3, 2);

    REQUIRE(SIMDWrapper<2>(left) == SIMDWrapper<2>(expected));
}

TEST_CASE("SSE - vec_left_32x4()") {
    __m128i u = _mm_set_epi32(4, 3, 2, 1);
    __m128i v = _mm_set_epi32(8, 7, 6, 5);

    auto right = sse::vec_left_32x4(u, v);

    // [1 2 3 4] [5 6 7 8] => [2 3 4 5]
    auto expected = _mm_set_epi32(5, 4, 3, 2);

    REQUIRE(SIMDWrapper(right) == SIMDWrapper(expected));
}


using CompressFun = std::function<int(__m128i, __m128i, __m128i&)>;

template <typename T, size_t Entries>
void test_compress_case(std::initializer_list<int> input, std::initializer_list<int> expected,
			const CompressFun& fun) {
    
    alignas(16) T input_arr[16];
    alignas(16) T expected_arr[16];
    alignas(16) T mask_arr[16];
    
    size_t len;

    std::fill(input_arr, input_arr + 16, 0);
    std::fill(expected_arr, expected_arr + 16, 0);
	
    build_array(input_arr, Entries, len, input);
    build_array(expected_arr, Entries, len, expected);
    
    __m128i u = _mm_load_si128((__m128i*)input_arr);
    __m128i expected_v = _mm_load_si128((__m128i*)expected_arr);

    int nonzero_cnt = 0;
    for (size_t i = 0; i < Entries; i++) {
	if (input_arr[i] != 0) {
	    mask_arr[i] = -1;
	    nonzero_cnt++;
	} else {
	    mask_arr[i] = 0;
	}
    }
    
    __m128i mask = _mm_load_si128((__m128i*)mask_arr);
    
    __m128i res;
    int popcnt = fun(u, mask, res);

    REQUIRE(popcnt == nonzero_cnt);
    REQUIRE(SIMDWrapper<sizeof(T)>(res) == SIMDWrapper<sizeof(T)>(expected_v));
}


void test_compress_16x8(std::initializer_list<int16_t> input, std::initializer_list<int16_t> expected) {
    
    int16_t in[8];
    int16_t ex[8];

    assert(input.size() == 8);
    
    size_t len;
    build_array(in, 8, len, input);
    build_array(ex, 8, len, expected);

    int nonzero_cnt = 0;
    for (auto val: expected) {
	if (val != 0) {
	    nonzero_cnt++;
	}
    }
    
    __m128i u = _mm_set_epi16(in[7], in[6], in[5], in[4], in[3], in[2], in[1], in[0]);
    __m128i expected_v = _mm_set_epi16(ex[7], ex[6], ex[5], ex[4], ex[3], ex[2], ex[1],
				       ex[0]);

    __m128i zero = _mm_set1_epi16(0);
    __m128i mask = _mm_cmpgt_epi16(u, zero);
    
    __m128i res;
    int popcnt = compress::sse::compress_16x8(u, mask, res);

    REQUIRE(popcnt == nonzero_cnt);
    REQUIRE(SIMDWrapper<2>(res) == SIMDWrapper<2>(expected_v));
}


TEST_CASE("SSE - Compress 32x4") {
    SECTION("Compress nothing") {
	test_compress_case<int32_t, 4>({1, 2, 3, 4}, {1, 2, 3, 4},
				       compress::sse::compress_32x4);
    }
    SECTION ("Compress everything") {
	test_compress_case<int32_t, 4>({0, 0, 0, 0}, {0, 0, 0, 0},
				       compress::sse::compress_32x4);
    }
    SECTION("Compress some") {
	test_compress_case<int32_t, 4>({1, 3, 0, 4}, {1, 3, 4, 0},
				       compress::sse::compress_32x4);
    }
    SECTION("Compress begining") {
	test_compress_case<int32_t, 4>({0, 0, 1, 2}, {1, 2, 0, 0},
				       compress::sse::compress_32x4);
    }
    SECTION ("Compress end") {
	test_compress_case<int32_t, 4>({1, 2, 3, 0}, {1, 2, 3, 0},
				       compress::sse::compress_32x4);
    }
}


TEST_CASE("SSE - Compress 16x8") {
    SECTION("Compress nothing") {
	test_compress_16x8({1, 2, 3, 4, 5, 6, 7, 8},
			   {1, 2, 3, 4, 5, 6, 7, 8});	
    }
    SECTION ("Compress everything") {
	test_compress_16x8({0, 0, 0, 0, 0, 0, 0, 0},
			   {0, 0, 0, 0, 0, 0, 0, 0});
    }
    SECTION("Compress some") {
        test_compress_16x8({1, 3, 0, 4, 0, 0, 6, 7},
			   {1, 3, 4, 6, 7, 0, 0, 0});
    }
    SECTION("Compress begining") {
        test_compress_16x8({0, 0, 0, 0, 0, 0, 1, 2},
			   {1, 2, 0, 0, 0, 0, 0, 0});
    }
    SECTION ("Compress end") {
        test_compress_16x8({1, 2, 3, 4, 5, 0, 0, 0},
			   {1, 2, 3, 4, 5, 0, 0, 0});
    }
}


void test_simdfun1_32x4(const SSEFun1& fun, std::initializer_list<int> in0,
		  std::initializer_list<int> expected_result) {
       
    __m128i a = sse::to_32x4_sse(in0);
    __m128i expected = sse::to_32x4_sse(expected_result);

    __m128i res = fun(a);

    REQUIRE(SIMDWrapper(res) == SIMDWrapper(expected));
}

void test_simdfun2_32x4(const SSEFun2& fun, std::initializer_list<int> in0, std::initializer_list<int> in1,
		  std::initializer_list<int> expected_result) {
       
    __m128i a = sse::to_32x4_sse(in0);
    __m128i b = sse::to_32x4_sse(in1);
    __m128i expected = sse::to_32x4_sse(expected_result);

    __m128i res = fun(a, b);

    REQUIRE(SIMDWrapper(res) == SIMDWrapper(expected));
}

void test_simdfun2_16x8(const SSEFun2& fun, std::initializer_list<int16_t> in0,
			std::initializer_list<int16_t> in1,
			std::initializer_list<int16_t> expected_results) {

    __m128i a = sse::to_16x8(in0);
    __m128i b = sse::to_16x8(in1);
    __m128i expected = sse::to_16x8(expected_results);

    __m128i res = fun(a, b);

    REQUIRE(SIMDWrapper<2>(res) == SIMDWrapper<2>(expected));
}

TEST_CASE("SSE - Interleave") {
    SECTION ("interleave_hi_32x4") {
	test_simdfun2_32x4(sse::interleave_lo_32x4, {1, 2, 3, 4}, {5, 6, 7, 8}, {1, 5, 3, 7});
    }
    SECTION ("interleave_lo_32x4") {
	test_simdfun2_32x4(sse::interleave_hi_32x4, {1, 2, 3, 4}, {5, 6, 7, 8}, {2, 6, 4, 8});
    }
}


TEST_CASE("SSE - Filter") {
    SECTION ("filter_lo_32x4") {
	test_simdfun2_32x4(sse::filter_lo_32x4, {1, 2, 3, 4}, {5, 6, 7, 8}, {1, 3, 5, 7});
    }
    SECTION ("filter_hi_32x4") {
	test_simdfun2_32x4(sse::filter_hi_32x4, {1, 2, 3, 4}, {5, 6, 7, 8}, {2, 4, 6, 8});
    }
}

TEST_CASE("SSE - movemask_32x4") {
    SECTION("All set") {
	int expected = 0b1111;
	__m128i a = _mm_set_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
	int res = sse::movemask_32x4(a);
	REQUIRE(res == expected);
    }
    SECTION ("None set") {
	int expected = 0b0000;
	__m128i a = _mm_set_epi32(0, 0, 0, 0);
	int res = sse::movemask_32x4(a);
	REQUIRE(res == expected);
    }
    SECTION ("Some set") {
	int expected = 0b0101;
	__m128i a = _mm_set_epi32(0, 0xFFFFFFFF, 0, 0xFFFFFFFF);
	int res = sse::movemask_32x4(a);
	REQUIRE(res == expected);
    }
}


TEST_CASE("SSE - movemask_16x8") {
    SECTION("All set") {
	int expected = 0b1111'1111;
	__m128i a = _mm_set_epi16(-1, -1, -1, -1, -1, -1, -1, -1);
	int res = sse::movemask_16x8(a);
	REQUIRE(res == expected);
    }
    
    SECTION ("None set") {
	int expected = 0b0000;
	__m128i a = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);
	int res = sse::movemask_16x8(a);
	REQUIRE(res == expected);
    }
    
    SECTION ("Some set") {
	int expected = 0b0101'1101;
	__m128i a = _mm_set_epi16(0, -1, 0, -1, -1, -1, 0, -1);
	int res = sse::movemask_16x8(a);
	REQUIRE(res == expected);
    }
    
    SECTION ("Some set #2") {
	int expected = 0b1101'0011;
	__m128i a = _mm_set_epi16(-1, -1, 0, -1, 0, 0, -1, -1);
	int res = sse::movemask_16x8(a);
	REQUIRE(res == expected);
    }
    
    SECTION ("Some set #3") {
	int expected = 0b1111'1000;
	__m128i a = _mm_set_epi16(-1, -1, -1, -1, -1, 0, 0, 0);
	int res = sse::movemask_16x8(a);
	REQUIRE(res == expected);
    }
    
    SECTION ("Some set #4") {
	int expected = 0b0001'1111;
	__m128i a = _mm_set_epi16(0, 0, 0, -1, -1, -1, -1, -1);
	int res = sse::movemask_16x8(a);
	REQUIRE(res == expected);
    }
}


TEST_CASE("bcast_last") {
    test_simdfun1_32x4(sse::bcast_last, {1, 2, 3, 4}, {4, 4, 4, 4});
}


TEST_CASE("bcast_first") {
    test_simdfun1_32x4(sse::bcast_first, {1, 2, 3, 4}, {1, 1, 1, 1});
}


TEST_CASE("Test count leq") {
    SECTION("All lower") {
	int expected = 4;
	__m128i a = _mm_set_epi32(4, 3, 2, 1);
	__m128i limit = _mm_set_epi32(5, 5, 5, 5);
	int res = sse::count_leq_32x4(a, limit);

	REQUIRE(res == expected);
    }
    SECTION ("All higher") {
	int expected = 0;
	__m128i a = _mm_set_epi32(5, 4, 3, 2);
	__m128i limit = _mm_set_epi32(1, 1, 1, 1);
	int res = sse::count_leq_32x4(a, limit);
	REQUIRE(res == expected);
    }
    SECTION ("Some lower") {
	int expected = 3;
	__m128i a = _mm_set_epi32(8, 6, 4, 2);
	__m128i limit = _mm_set_epi32(7, 7, 7, 7);
	int res = sse::count_leq_32x4(a, limit);
	REQUIRE(res == expected);
    }
    SECTION ("All equal") {
	int expected = 0;
	__m128i a = _mm_set_epi32(3, 3, 3, 3);
	__m128i limit = _mm_set_epi32(3, 3, 3, 3);
	int res = sse::count_leq_32x4(a, limit);
	REQUIRE(res == expected);
    }
}

bool is_sorted_32x4(__m128i a) {
    auto left = sse::vec_left_32x4(a, _mm_set1_epi32(std::numeric_limits<int>::max()));
    auto mask = _mm_cmpgt_epi32(a, left);
    
    int m = _mm_movemask_epi8(mask);
    return m == 0;
}

bool is_sorted_32x2x2(__m128i a, __m128i b) {
    auto mask = _mm_cmpgt_epi32(a, b);
    int m = _mm_movemask_epi8(mask);

    return is_sorted_32x4(a) && is_sorted_32x4(b) && m == 0;
}


bool is_sorted_sep_32x4(__m128i a) {
    alignas(16) int32_t vals[4];
    _mm_store_si128((__m128i*)vals, a);

    return vals[0] <= vals[1] && vals[2] <= vals[3];
}


bool is_sorted_sep_32x2x2(__m128i a, __m128i b) {
    return is_sorted_32x4(a) && is_sorted_32x4(b);
}

bool is_sorted_mix_32x4(__m128i a) {
    alignas(16) int32_t vals[4];
    _mm_store_si128((__m128i*)vals, a);

    return vals[0] <= vals[2] && vals[1] <= vals[3];
}

bool is_sorted_mix_16x8(__m128i a) {
    alignas(16) int16_t vals[8];
    _mm_store_si128((__m128i*)vals, a);

    return vals[0] <= vals[2] && vals[1] <= vals[3]
	&& vals[2] <= vals[4] && vals[3] <= vals[5]
	&& vals[4] <= vals[6] && vals[5] <= vals[7];
}

bool is_sorted_mix_32x2x2(__m128i a, __m128i b) {
    auto mask = _mm_cmpgt_epi32(a, b);
    int m = _mm_movemask_epi8(mask);

    return is_sorted_mix_32x4(a) && is_sorted_mix_32x4(b) && m == 0;
}

bool is_sorted_mix_16x2x4(__m128i a, __m128i b) {
    auto mask = _mm_cmpgt_epi16(a, b);
    int m = _mm_movemask_epi8(mask);

    return is_sorted_mix_16x8(a) && is_sorted_mix_16x8(b) && m == 0;
}


bool is_hlower_32x4(__m128i a) {
    auto mask = _mm_cmpgt_epi32(a, _mm_set1_epi32(255));
    int m = _mm_movemask_epi8(mask);
    return m == 0;
}

__m128i n_one_to_32x4(int n) {
    assert(n < 5);
    alignas(16) static uint32_t lut[5][4] = {
	{0, 0, 0, 0},
	{1, 0, 0, 0},
	{1, 1, 0, 0},
	{1, 1, 1, 0},
	{1, 1, 1, 1}};
    return ((__m128i*)lut)[n];    
}


__m128i nm_one_to_32x4(int n, int m) {
    assert(n < 3 && m < 3);
    alignas(16) uint32_t data[4];

    switch (m) {
    case 0:
	data[0] = 0;
	data[2] = 0;
	break;
    case 1:
	data[0] = 0;
	data[2] = 1;
	break;
    case 2:
	data[0] = 1;
	data[2] = 1;
    default:
	break;
    }

    switch (n) {
    case 0:
	data[1] = 0;
	data[3] = 0;
	break;
    case 1:
	data[1] = 0;
	data[3] = 1;
	break;
    case 2:
	data[1] = 1;
	data[3] = 1;
	break;
    default:
	break;    
    }

    __m128i res = *((__m128i*)data);
    return res;
}

__m128i nm_one_to_16x8(int n, int m) {
    assert(n < 5 && m < 5);

    alignas(16) int16_t data[8];

    for (int i = 0; i < 8; i++) {
	data[i] = 0;
    }
    
    for (int i = 0; i < 4; i++) {
	data[2 * i + 1] = (n > 0 ? 1 : 0);
	data[2 * i] = (m > 0 ? 1 : 0);
	n--;
	m--;
    }

    __m128i res = *((__m128i*)data);
    return res;
}

void test_sort_case(const BitonicFunV& fun, std::initializer_list<int> in0,
		    std::initializer_list<int> in1,
		    std::initializer_list<int> expected0, std::initializer_list<int> expected1) {
    __m128i a = sse::to_32x4_sse(in0);
    __m128i b = sse::to_32x4_sse(in1);
    __m128i exp0 = sse::to_32x4_sse(expected0);
    __m128i exp1 = sse::to_32x4_sse(expected1);


    fun(a, b);

    REQUIRE(SIMDWrapper(a) == SIMDWrapper(exp0));
    REQUIRE(SIMDWrapper(b) == SIMDWrapper(exp1));
}

TEST_CASE("Test is_sorted_32x4") {
    REQUIRE(is_sorted_32x4(_mm_set_epi32(0, 0, 0, 0)) == true);
    REQUIRE(is_sorted_32x4(_mm_set_epi32(4, 3, 2, 1)) == true);
    REQUIRE(is_sorted_32x4(_mm_set_epi32(1, 4, 3, 2)) == false);
    REQUIRE(is_sorted_32x4(_mm_set_epi32(4, 2, 3, 1)) == false);
    REQUIRE(is_sorted_32x4(_mm_set_epi32(1, 1, 0, 0)) == true);
    REQUIRE(is_sorted_32x4(_mm_set_epi32(1, 0, 0, 0)) == true);
    REQUIRE(is_sorted_32x4(_mm_set_epi32(0, 0, 0, 1)) == false);
}

TEST_CASE("Test is_sorted_sep_32x4") {
    REQUIRE(is_sorted_sep_32x4(_mm_set_epi32(0, 0, 0, 0)) == true);
    REQUIRE(is_sorted_sep_32x4(_mm_set_epi32(8, 2, 6, 1)) == true);
    REQUIRE(is_sorted_sep_32x4(_mm_set_epi32(6, 2, 1, 8)) == false);
    REQUIRE(is_sorted_sep_32x4(_mm_set_epi32(2, 6, 8, 1)) == false);
    REQUIRE(is_sorted_sep_32x4(_mm_set_epi32(2, 6, 8, 1)) == false);

    REQUIRE(is_sorted_sep_32x4(_mm_set_epi32(1, 0, 1, 0)) == true);
    REQUIRE(is_sorted_sep_32x4(_mm_set_epi32(1, 0, 0, 0)) == true);
}

TEST_CASE("Test is_sorted_mix_32x4") {
    REQUIRE(is_sorted_mix_32x4(_mm_set_epi32(0, 0, 0, 0)) == true);
    REQUIRE(is_sorted_mix_32x4(_mm_set_epi32(0, 1, 0, 0)) == true);
    REQUIRE(is_sorted_mix_32x4(_mm_set_epi32(1, 1, 0, 0)) == true);
    
    REQUIRE(is_sorted_mix_32x4(_mm_set_epi32(0, 1, 0, 1)) == true);
    REQUIRE(is_sorted_mix_32x4(_mm_set_epi32(1, 0, 1, 0)) == true);

    REQUIRE(is_sorted_mix_32x4(_mm_set_epi32(0, 0, 0, 1)) == false);
    REQUIRE(is_sorted_mix_32x4(_mm_set_epi32(0, 0, 1, 0)) == false);
}

TEST_CASE("Test n_one_to_32x4") {
    REQUIRE(SIMDWrapper(n_one_to_32x4(0)) == SIMDWrapper(_mm_set_epi32(0, 0, 0, 0)));
    REQUIRE(SIMDWrapper(n_one_to_32x4(1)) == SIMDWrapper(_mm_set_epi32(0, 0, 0, 1)));
    REQUIRE(SIMDWrapper(n_one_to_32x4(2)) == SIMDWrapper(_mm_set_epi32(0, 0, 1, 1)));
    REQUIRE(SIMDWrapper(n_one_to_32x4(3)) == SIMDWrapper(_mm_set_epi32(0, 1, 1, 1)));
    REQUIRE(SIMDWrapper(n_one_to_32x4(4)) == SIMDWrapper(_mm_set_epi32(1, 1, 1, 1)));	
}

TEST_CASE("Test nm_one_to_32x4") {
    REQUIRE(SIMDWrapper(nm_one_to_32x4(0, 0)) == SIMDWrapper(_mm_set_epi32(0, 0, 0, 0)));
    REQUIRE(SIMDWrapper(nm_one_to_32x4(0, 2)) == SIMDWrapper(_mm_set_epi32(0, 1, 0, 1)));
    REQUIRE(SIMDWrapper(nm_one_to_32x4(1, 0)) == SIMDWrapper(_mm_set_epi32(1, 0, 0, 0)));
    REQUIRE(SIMDWrapper(nm_one_to_32x4(1, 1)) == SIMDWrapper(_mm_set_epi32(1, 1, 0, 0)));
    REQUIRE(SIMDWrapper(nm_one_to_32x4(1, 2)) == SIMDWrapper(_mm_set_epi32(1, 1, 0, 1)));
    
    REQUIRE(SIMDWrapper(nm_one_to_32x4(2, 2)) == SIMDWrapper(_mm_set_epi32(1, 1, 1, 1)));
    REQUIRE(SIMDWrapper(nm_one_to_32x4(2, 1)) == SIMDWrapper(_mm_set_epi32(1, 1, 1, 0)));

    REQUIRE(!is_hlower_32x4(_mm_set1_epi32(500)));
    for (int i = 0; i < 3; i++) {
	for (int j = 0; j < 3; j++) {
	    __m128i v = nm_one_to_32x4(i, j);
	    REQUIRE(is_sorted_mix_32x4(v));
	    REQUIRE(is_hlower_32x4(v));
	}
    }
}

TEST_CASE("SSE - nm_one_to_16x8") {
    REQUIRE(SIMDWrapper<2>(nm_one_to_16x8(0, 0)) == SIMDWrapper<2>(_mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0)));
    REQUIRE(SIMDWrapper<2>(nm_one_to_16x8(0, 2)) == SIMDWrapper<2>(_mm_set_epi16(0, 0, 0, 0, 0, 1, 0, 1)));
    REQUIRE(SIMDWrapper<2>(nm_one_to_16x8(0, 3)) == SIMDWrapper<2>(_mm_set_epi16(0, 0, 0, 1, 0, 1, 0, 1)));
    REQUIRE(SIMDWrapper<2>(nm_one_to_16x8(1, 0)) == SIMDWrapper<2>(_mm_set_epi16(0, 0, 0, 0, 0, 0, 1, 0)));
    REQUIRE(SIMDWrapper<2>(nm_one_to_16x8(2, 0)) == SIMDWrapper<2>(_mm_set_epi16(0, 0, 0, 0, 1, 0, 1, 0)));

    REQUIRE(SIMDWrapper<2>(nm_one_to_16x8(4, 0)) == SIMDWrapper<2>(_mm_set_epi16(1, 0, 1, 0, 1, 0, 1, 0)));

    REQUIRE(SIMDWrapper<2>(nm_one_to_16x8(2, 3)) == SIMDWrapper<2>(_mm_set_epi16(0, 0, 0, 1, 1, 1, 1, 1)));
    REQUIRE(SIMDWrapper<2>(nm_one_to_16x8(4, 4)) == SIMDWrapper<2>(_mm_set_epi16(1, 1, 1, 1, 1, 1, 1, 1)));

    for (int i = 0; i < 5; i++) {
	for (int j = 0; j < 5; j++) {
	    __m128i v = nm_one_to_16x8(i, j);
	    //REQUIRE(is_sorted_mix_16x8(v));
	}
    }
}

void test_case_expand_32x4(__m128i mask) {
    __m128i vals = _mm_set_epi32(4, 3, 2, 1);
    __m128i res;
    ::expand::sse::expand_32x4(vals, mask, res);

    alignas(16) int m[4];    
    alignas(16) int32_t arr[4];

    _mm_store_si128((__m128i*)(m), mask);
    int j = 1;
    for (int i = 0; i < 4; i++) {
	if (m[i]) {
	    arr[i] = j;
	    j++;
	} else {
	    arr[i] = 0;
	}
    }

    __m128i expected = _mm_load_si128((__m128i*)(arr));
    
    REQUIRE(SIMDWrapper<4>(res) == SIMDWrapper<4>(expected));
}

TEST_CASE("SSE - expand_32x4")  {
    SECTION ("None set") {
	test_case_expand_32x4(_mm_set_epi32(0, 0, 0, 0));
    }
    SECTION ("All set") {
	test_case_expand_32x4(_mm_set_epi32(-1, -1, -1, -1));
    }
    SECTION("Some set") {
	test_case_expand_32x4(_mm_set_epi32(-1, 0, -1, 0));
    }

    for (int i = 0; i < 16; i++) {
	__m128i mask = ::sse::from_mask_32x4(i);
	test_case_expand_32x4(mask);
    }
}

/*
TEST_CASE("SSE - cmp_le_epi16") {
    SECTION ("Lower") {
	test_simdfun2_16x8(sse::cmp_le_epi16, {1, 1, 1, 1, 1, 1, 1, 1}, {2, 2, 2, 2, 2, 2, 2, 2},
			   {-1, -1, -1, -1, -1, -1, -1, -1});
    }
    SECTION ("Greater") {
	test_simdfun2_16x8(sse::cmp_le_epi16, {2, 2, 2, 2, 2, 2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1}, 
			   {0, 0, 0, 0, 0, 0, 0, 0});

    }
    SECTION("Equal") {
	test_simdfun2_16x8(sse::cmp_le_epi16, {1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1},
			   {-1, -1, -1, -1, -1, -1, -1, -1});
    }
    }*/

// Exhaustive tests for bitonic merge
void test_bitonic_32x4(const BitonicFun& fun) {

    for (int i = 0; i <= 4; i++ ) {
	__m128i a = n_one_to_32x4(i);
	
	for (int j = 0; j <= 4; j++) {
	    __m128i b = n_one_to_32x4(j);
	    
	    fun(a, b);
	    REQUIRE(is_sorted_32x2x2(a, b));
	}
    }
}

void test_bitonic_sep_32x4(const BitonicFunV& fun) {

    test_sort_case(fun, {1, 2, 5, 6}, {3, 3, 7, 8}, {1, 3, 5, 7}, {2, 3, 6, 8});
    test_sort_case(fun, {1, 2, 3, 4}, {5, 6, 7, 8}, {1, 3, 5, 7}, {2, 4, 6, 8});
    test_sort_case(fun, {1, 2, 5, 6}, {3, 4, 7, 8}, {1, 3, 5, 7}, {2, 4, 6, 8});

    
    for (int i0 = 0; i0 < 3; i0 ++) {
	for (int j0 = 0; j0 < 3; j0++) {

	    __m128i a = nm_one_to_32x4(j0, i0);

	    REQUIRE(is_sorted_mix_32x4(a));

	    for (int i1 = 0; i1 < 3; i1++) {
		for (int j1 = 0; j1 < 3; j1++) {		    
		    __m128i b = nm_one_to_32x4(j1, i1);

		    REQUIRE(is_sorted_mix_32x4(b));

		    fun(a, b);
		    REQUIRE(is_sorted_sep_32x2x2(a, b));
		}
	    }	    
	}
    }
}


void test_bitonic_mix_32x4(const BitonicFunV& fun) {
    
    test_sort_case(fun, {1, 2, 5, 6}, {7, 8, 3, 3}, {1, 2, 3, 3}, {7, 8, 5, 6});
    test_sort_case(fun, {1, 2, 3, 4}, {7, 8, 5, 6}, {1, 2, 3, 4}, {7, 8, 5, 6});
    test_sort_case(fun, {1, 2, 5, 6}, {7, 8, 3, 4}, {1, 2, 3, 4}, {7, 8, 5, 6});
    
    for (int i0 = 0; i0 < 3; i0 ++) {
	for (int j0 = 0; j0 < 3; j0++) {

	    __m128i a = nm_one_to_32x4(j0, i0);
	    
	    for (int i1 = 0; i1 < 3; i1++) {
		for (int j1 = 0; j1 < 3; j1++) {		    
		    __m128i b = nm_one_to_32x4(j1, i1);

		    b = sse::invert_32x4(b);
		    fun(a, b);
		    REQUIRE(is_sorted_mix_32x2x2(a, sse::invert_32x4(b)));
		}
	    }
	}
    }
}


void test_bitonic_16x8(const BitonicFunV& fun) {
    for (int i0 = 0; i0 < 5; i0++) {
	for (int j0 = 0; j0 < 5; j0++) {
	    
	    __m128i a = nm_one_to_16x8(j0, i0);

	    for (int i1 = 0; i1 < 5; i1++) {
		for (int j1 = 0; j1 < 5; j1++) {
		    __m128i b = nm_one_to_16x8(j1, i1);

		    fun(a, b);
		    REQUIRE(is_sorted_mix_16x2x4(a, b));
		}
	    }
	}
    }
}

#endif // __x86_64__
