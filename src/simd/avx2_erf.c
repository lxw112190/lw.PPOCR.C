#include "simd_kernels.h"

/* AVX2 Erf approximation used by GELU-shaped model graphs. */

#include <math.h>
#include <stddef.h>

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#  include <immintrin.h>
#  define LW_COMPILES_AVX2_ERF 1
#else
#  define LW_COMPILES_AVX2_ERF 0
#endif

#if LW_COMPILES_AVX2_ERF
#  if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,no-fma")))
#  endif
static __m256
erf_small_magnitude_f32(__m256 absolute, __m256 squared) {
    __m256 polynomial = _mm256_set1_ps(1.0590875083315439e-6f);
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(-1.3906452410711274e-5f), _mm256_mul_ps(squared, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(1.1955437252428243e-4f), _mm256_mul_ps(squared, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(-8.5424759600797662e-4f), _mm256_mul_ps(squared, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(5.2237718994271529e-3f), _mm256_mul_ps(squared, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(-2.6866128888288668e-2f), _mm256_mul_ps(squared, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(1.1283791226625459e-1f), _mm256_mul_ps(squared, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(-3.7612638883043881e-1f), _mm256_mul_ps(squared, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(1.1283791670929921f), _mm256_mul_ps(squared, polynomial));
    return _mm256_mul_ps(absolute, polynomial);
}

#  if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,no-fma")))
#  endif
static __m256
erf_middle_magnitude_f32(__m256 absolute) {
    __m256 shifted = _mm256_sub_ps(absolute, _mm256_set1_ps(1.5f));
    __m256 polynomial = _mm256_set1_ps(-2.4006675278365739e-3f);
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(-3.8855162788028288e-3f), _mm256_mul_ps(shifted, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(1.9332860298601401e-2f), _mm256_mul_ps(shifted, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(-1.5013607234871428e-2f), _mm256_mul_ps(shifted, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(-4.4599369472787913e-2f), _mm256_mul_ps(shifted, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(1.3876136033191724e-1f), _mm256_mul_ps(shifted, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(-1.7839541988688759e-1f), _mm256_mul_ps(shifted, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(1.1893013063163335e-1f), _mm256_mul_ps(shifted, polynomial));
    return _mm256_add_ps(_mm256_set1_ps(9.6610514641406819e-1f),
                         _mm256_mul_ps(shifted, polynomial));
}

#  if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,no-fma")))
#  endif
static __m256
erf_large_magnitude_f32(__m256 absolute) {
    __m256 shifted = _mm256_sub_ps(absolute, _mm256_set1_ps(3.0f));
    __m256 polynomial = _mm256_set1_ps(-8.8750765320565021e-5f);
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(3.8805285630072217e-4f), _mm256_mul_ps(shifted, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(-7.7812010711428433e-4f), _mm256_mul_ps(shifted, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(1.0255980996254569e-3f), _mm256_mul_ps(shifted, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(-1.0307062838910627e-3f), _mm256_mul_ps(shifted, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(7.8586080129729565e-4f), _mm256_mul_ps(shifted, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(-4.1936053855221229e-4f), _mm256_mul_ps(shifted, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(1.3951109720745927e-4f), _mm256_mul_ps(shifted, polynomial));
    return _mm256_add_ps(_mm256_set1_ps(9.9997793886838715e-1f),
                         _mm256_mul_ps(shifted, polynomial));
}
#endif

#if LW_COMPILES_AVX2_ERF && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,no-fma")))
#endif
void lw_avx2_erf_f32(const float* input, float* output, uint64_t element_count) {
#if LW_COMPILES_AVX2_ERF
    const __m256 absolute_mask = _mm256_castsi256_ps(_mm256_set1_epi32((int)UINT32_C(0x7fffffff)));
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32((int)UINT32_C(0x80000000)));
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 two = _mm256_set1_ps(2.0f);
    const __m256 four = _mm256_set1_ps(4.0f);
    uint64_t index = 0u;
    for (; index + 8u <= element_count; index += 8u) {
        __m256 value = _mm256_loadu_ps(input + (size_t)index);
        __m256 absolute = _mm256_and_ps(value, absolute_mask);
        __m256 squared = _mm256_mul_ps(absolute, absolute);
        __m256 small = erf_small_magnitude_f32(absolute, squared);
        __m256 middle = erf_middle_magnitude_f32(absolute);
        __m256 large = erf_large_magnitude_f32(absolute);
        __m256 mask;
        __m256 result;

        mask = _mm256_cmp_ps(absolute, two, _CMP_LT_OQ);
        result = _mm256_blendv_ps(large, middle, mask);
        mask = _mm256_cmp_ps(absolute, one, _CMP_LT_OQ);
        result = _mm256_blendv_ps(result, small, mask);
        mask = _mm256_cmp_ps(absolute, four, _CMP_GE_OQ);
        result = _mm256_blendv_ps(result, one, mask);
        result = _mm256_xor_ps(result, _mm256_and_ps(value, sign_mask));
        mask = _mm256_cmp_ps(value, value, _CMP_UNORD_Q);
        result = _mm256_blendv_ps(result, value, mask);
        _mm256_storeu_ps(output + (size_t)index, result);
    }
    for (; index < element_count; ++index) {
        output[(size_t)index] = erff(input[(size_t)index]);
    }
#else
    uint64_t index;
    for (index = 0u; index < element_count; ++index) {
        output[(size_t)index] = erff(input[(size_t)index]);
    }
#endif
}
