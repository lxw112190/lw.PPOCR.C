#include "simd_kernels.h"

/* AVX2 Softmax for the long, contiguous REC classification axis. */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#  include <immintrin.h>
#  define LW_COMPILES_AVX2_SOFTMAX 1
#else
#  define LW_COMPILES_AVX2_SOFTMAX 0
#endif

#if LW_COMPILES_AVX2_SOFTMAX
#  if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,no-fma")))
#  endif
static __m256
exp_approximation_f32(__m256 value) {
    /* This is the standard base-2 range reduction used by vector math
     * kernels. The clamp matches the finite range accepted by expf and the
     * polynomial error stays well below the public graph tolerances. */
    const __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    const __m256 c1 = _mm256_set1_ps(0.693359375f);
    const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 minimum = _mm256_set1_ps(-88.3762626647949f);
    const __m256 maximum = _mm256_set1_ps(88.3762626647949f);
    const __m256 half = _mm256_set1_ps(0.5f);
    __m256 scaled;
    __m256 integer_part;
    __m256 reduced;
    __m256 square;
    __m256 polynomial;
    __m256i exponent;
    __m256 power;

    value = _mm256_max_ps(value, minimum);
    value = _mm256_min_ps(value, maximum);
    scaled = _mm256_add_ps(_mm256_mul_ps(value, log2e), half);
    integer_part = _mm256_cvtepi32_ps(_mm256_cvttps_epi32(scaled));
    /* Truncation toward zero rounds negative values in the wrong direction. */
    integer_part =
        _mm256_sub_ps(integer_part, _mm256_and_ps(_mm256_cmp_ps(integer_part, scaled, _CMP_GT_OQ),
                                                  _mm256_set1_ps(1.0f)));
    reduced = _mm256_sub_ps(value, _mm256_mul_ps(integer_part, c1));
    reduced = _mm256_sub_ps(reduced, _mm256_mul_ps(integer_part, c2));
    square = _mm256_mul_ps(reduced, reduced);
    /* Taylor expansion around zero. Range reduction keeps |reduced| below
     * ln(2)/2, so the omitted ninth-order term is below 1e-8 in FP32. */
    polynomial = _mm256_set1_ps(2.4801587302E-5f); /* 1 / 40320 */
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(1.9841269841E-4f), _mm256_mul_ps(reduced, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(1.3888888889E-3f), _mm256_mul_ps(reduced, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(8.3333333333E-3f), _mm256_mul_ps(reduced, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(4.1666666667E-2f), _mm256_mul_ps(reduced, polynomial));
    polynomial =
        _mm256_add_ps(_mm256_set1_ps(1.6666666667E-1f), _mm256_mul_ps(reduced, polynomial));
    polynomial = _mm256_add_ps(_mm256_set1_ps(5.0E-1f), _mm256_mul_ps(reduced, polynomial));
    polynomial = _mm256_add_ps(reduced, _mm256_mul_ps(square, polynomial));
    polynomial = _mm256_add_ps(_mm256_set1_ps(1.0f), polynomial);
    exponent = _mm256_add_epi32(_mm256_cvttps_epi32(integer_part), _mm256_set1_epi32(127));
    power = _mm256_castsi256_ps(_mm256_slli_epi32(exponent, 23));
    return _mm256_mul_ps(polynomial, power);
}
#endif

#if LW_COMPILES_AVX2_SOFTMAX && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,no-fma")))
#endif
void lw_avx2_softmax_contiguous_f32(const float* input, float* output, uint64_t row_count,
                                    uint64_t axis_count) {
#if LW_COMPILES_AVX2_SOFTMAX
    uint64_t row;
    for (row = 0u; row < row_count; ++row) {
        const float* input_row = input + (size_t)(row * axis_count);
        float* output_row = output + (size_t)(row * axis_count);
        float maximum = input_row[0];
        float sum;
        uint64_t index;
        __m256 maximum_values = _mm256_set1_ps(input_row[0]);
        for (index = 0u; index + 8u <= axis_count; index += 8u) {
            maximum_values =
                _mm256_max_ps(maximum_values, _mm256_loadu_ps(input_row + (size_t)index));
        }
        for (; index < axis_count; ++index) {
            if (input_row[(size_t)index] > maximum) {
                maximum = input_row[(size_t)index];
            }
        }
        {
            float lanes[8];
            _mm256_storeu_ps(lanes, maximum_values);
            for (index = 0u; index < 8u; ++index) {
                if (lanes[index] > maximum) {
                    maximum = lanes[index];
                }
            }
        }
        sum = 0.0f;
        index = 0u;
        {
            __m256 maximum_vector = _mm256_set1_ps(maximum);
            for (; index + 8u <= axis_count; index += 8u) {
                __m256 values = exp_approximation_f32(
                    _mm256_sub_ps(_mm256_loadu_ps(input_row + (size_t)index), maximum_vector));
                _mm256_storeu_ps(output_row + (size_t)index, values);
            }
        }
        for (index = axis_count - (axis_count % 8u); index < axis_count; ++index) {
            float value = expf(input_row[(size_t)index] - maximum);
            output_row[(size_t)index] = value;
        }
        /* Keep the established left-to-right accumulation order so callers
         * that consume the maximum probability retain their score contract. */
        for (index = 0u; index < axis_count; ++index) {
            sum += output_row[(size_t)index];
        }
        {
            __m256 inverse_sum = _mm256_set1_ps(1.0f / sum);
            for (index = 0u; index + 8u <= axis_count; index += 8u) {
                _mm256_storeu_ps(
                    output_row + (size_t)index,
                    _mm256_mul_ps(_mm256_loadu_ps(output_row + (size_t)index), inverse_sum));
            }
        }
        for (index = axis_count - (axis_count % 8u); index < axis_count; ++index) {
            output_row[(size_t)index] /= sum;
        }
    }
#else
    uint64_t row;
    for (row = 0u; row < row_count; ++row) {
        const float* input_row = input + (size_t)(row * axis_count);
        float* output_row = output + (size_t)(row * axis_count);
        float maximum = input_row[0];
        float sum = 0.0f;
        uint64_t index;
        for (index = 1u; index < axis_count; ++index) {
            if (input_row[(size_t)index] > maximum) {
                maximum = input_row[(size_t)index];
            }
        }
        for (index = 0u; index < axis_count; ++index) {
            float value = expf(input_row[(size_t)index] - maximum);
            output_row[(size_t)index] = value;
            sum += value;
        }
        for (index = 0u; index < axis_count; ++index) {
            output_row[(size_t)index] /= sum;
        }
    }
#endif
}
