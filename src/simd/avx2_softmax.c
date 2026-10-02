#include "simd_kernels.h"

/* AVX2 Softmax for the long, contiguous REC classification axis. */

#include "../kernels/packed_matmul_internal.h"

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

#if LW_COMPILES_AVX2_SOFTMAX && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,no-fma")))
#endif
void lw_avx2_ctc_emitted_softmax_contiguous_f32(const float* input,
                                                const uint32_t* best_indices,
                                                float* emitted_probabilities,
                                                uint64_t row_count, uint64_t axis_count) {
#if LW_COMPILES_AVX2_SOFTMAX
    uint64_t row;
    for (row = 0u; row < row_count; ++row) {
        uint32_t best_index = best_indices[(size_t)row];
        if (best_index != 0u &&
            (row == 0u || best_index != best_indices[(size_t)row - 1u])) {
            const float* input_row = input + (size_t)(row * axis_count);
            float maximum = input_row[best_index];
            float sum = 0.0f;
            uint64_t index;
            __m256 maximum_vector = _mm256_set1_ps(maximum);
            for (index = 0u; index + 8u <= axis_count; index += 8u) {
                float lanes[8];
                __m256 values = exp_approximation_f32(
                    _mm256_sub_ps(_mm256_loadu_ps(input_row + (size_t)index), maximum_vector));
                uint32_t lane;
                _mm256_storeu_ps(lanes, values);
                for (lane = 0u; lane < 8u; ++lane) {
                    sum += lanes[lane];
                }
            }
            for (; index < axis_count; ++index) {
                sum += expf(input_row[(size_t)index] - maximum);
            }
            emitted_probabilities[(size_t)row] = 1.0f / sum;
        }
    }
#else
    uint64_t row;
    for (row = 0u; row < row_count; ++row) {
        uint32_t best_index = best_indices[(size_t)row];
        if (best_index != 0u &&
            (row == 0u || best_index != best_indices[(size_t)row - 1u])) {
            const float* input_row = input + (size_t)(row * axis_count);
            float maximum = input_row[best_index];
            float sum = 0.0f;
            uint64_t index;
            for (index = 0u; index < axis_count; ++index) {
                sum += expf(input_row[(size_t)index] - maximum);
            }
            emitted_probabilities[(size_t)row] = 1.0f / sum;
        }
    }
#endif
}

#if LW_COMPILES_AVX2_SOFTMAX && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma")))
#endif
void lw_avx2_ctc_row_probabilities_f32(
    const float* input, const float* packed_weights, const float* bias,
    const uint32_t* best_indices, const float* scores, float* probabilities,
    uint32_t rows, uint32_t inner_dimension, uint32_t columns) {
#if LW_COMPILES_AVX2_SOFTMAX
    const uint32_t column_panels =
        (columns + LW_PACKED_MATMUL_COLUMN_TILE - 1u) / LW_PACKED_MATMUL_COLUMN_TILE;
    uint32_t row;
    /* Only emitted rows need a probability: recompute that row's logits as a
     * single packed matvec and fold the softmax denominator into the same
     * pass, so the [rows x columns] logit tensor never exists. The 8-lane
     * batches and per-lane accumulation order match
     * lw_avx2_ctc_emitted_softmax_contiguous_f32 for bit-comparable sums. */
    for (row = 0u; row < rows; ++row) {
        uint32_t best_index = best_indices[(size_t)row];
        if (best_index != 0u &&
            (row == 0u || best_index != best_indices[(size_t)row - 1u])) {
            const float maximum = scores[(size_t)row];
            const __m256 maximum_vector = _mm256_set1_ps(maximum);
            const float* input_row = input + (size_t)row * inner_dimension;
            float sum = 0.0f;
            uint32_t column_panel;
            for (column_panel = 0u; column_panel < column_panels; ++column_panel) {
                const float* packed =
                    packed_weights +
                    (size_t)(column_panel * inner_dimension *
                             LW_PACKED_MATMUL_COLUMN_TILE);
                __m256 accumulator_low = _mm256_setzero_ps();
                __m256 accumulator_high = _mm256_setzero_ps();
                uint32_t inner;
                for (inner = 0u; inner < inner_dimension; ++inner) {
                    const __m256 weight_low = _mm256_loadu_ps(packed);
                    const __m256 weight_high = _mm256_loadu_ps(packed + 8u);
                    const __m256 input_value = _mm256_set1_ps(input_row[inner]);
                    accumulator_low =
                        _mm256_fmadd_ps(input_value, weight_low, accumulator_low);
                    accumulator_high =
                        _mm256_fmadd_ps(input_value, weight_high, accumulator_high);
                    packed += LW_PACKED_MATMUL_COLUMN_TILE;
                }
                {
                    const uint32_t column_base =
                        column_panel * LW_PACKED_MATMUL_COLUMN_TILE;
                    const uint32_t valid_columns =
                        columns - column_base < LW_PACKED_MATMUL_COLUMN_TILE
                            ? columns - column_base
                            : LW_PACKED_MATMUL_COLUMN_TILE;
                    if (bias != NULL) {
                        __m256 bias_low;
                        __m256 bias_high;
                        if (valid_columns == LW_PACKED_MATMUL_COLUMN_TILE) {
                            bias_low = _mm256_loadu_ps(bias + column_base);
                            bias_high = _mm256_loadu_ps(bias + column_base + 8u);
                        } else {
                            /* Packed weights are padded to 16 lanes, but
                             * model bias tensors contain only real classes. */
                            float bias_values[LW_PACKED_MATMUL_COLUMN_TILE] = {0.0f};
                            uint32_t lane;
                            for (lane = 0u; lane < valid_columns; ++lane) {
                                bias_values[lane] = bias[column_base + lane];
                            }
                            bias_low = _mm256_loadu_ps(bias_values);
                            bias_high = _mm256_loadu_ps(bias_values + 8u);
                        }
                        accumulator_low = _mm256_add_ps(
                            accumulator_low, bias_low);
                        accumulator_high = _mm256_add_ps(
                            accumulator_high, bias_high);
                    }
                    if (valid_columns == LW_PACKED_MATMUL_COLUMN_TILE) {
                        float lanes[8];
                        __m256 values;
                        uint32_t lane;
                        values = exp_approximation_f32(
                            _mm256_sub_ps(accumulator_low, maximum_vector));
                        _mm256_storeu_ps(lanes, values);
                        for (lane = 0u; lane < 8u; ++lane) sum += lanes[lane];
                        values = exp_approximation_f32(
                            _mm256_sub_ps(accumulator_high, maximum_vector));
                        _mm256_storeu_ps(lanes, values);
                        for (lane = 0u; lane < 8u; ++lane) sum += lanes[lane];
                    } else {
                        float lanes[LW_PACKED_MATMUL_COLUMN_TILE];
                        uint32_t lane = 0u;
                        _mm256_storeu_ps(lanes, accumulator_low);
                        _mm256_storeu_ps(lanes + 8u, accumulator_high);
                        while (lane + 8u <= valid_columns) {
                            float batch[8];
                            __m256 values = exp_approximation_f32(_mm256_sub_ps(
                                _mm256_loadu_ps(lanes + lane), maximum_vector));
                            uint32_t sub_lane;
                            _mm256_storeu_ps(batch, values);
                            for (sub_lane = 0u; sub_lane < 8u; ++sub_lane) {
                                sum += batch[sub_lane];
                            }
                            lane += 8u;
                        }
                        for (; lane < valid_columns; ++lane) {
                            sum += expf(lanes[lane] - maximum);
                        }
                    }
                }
            }
            probabilities[(size_t)row] = 1.0f / sum;
        }
    }
#else
    uint32_t row;
    for (row = 0u; row < rows; ++row) {
        uint32_t best_index = best_indices[(size_t)row];
        if (best_index != 0u &&
            (row == 0u || best_index != best_indices[(size_t)row - 1u])) {
            const float maximum = scores[(size_t)row];
            const float* input_row = input + (size_t)row * inner_dimension;
            float sum = 0.0f;
            uint32_t column;
            for (column = 0u; column < columns; ++column) {
                float value = bias != NULL ? bias[column] : 0.0f;
                uint32_t inner;
                const float* weights = packed_weights +
                    (size_t)((column / LW_PACKED_MATMUL_COLUMN_TILE) *
                             inner_dimension * LW_PACKED_MATMUL_COLUMN_TILE +
                             (column % LW_PACKED_MATMUL_COLUMN_TILE));
                for (inner = 0u; inner < inner_dimension; ++inner) {
                    value += input_row[inner] *
                             weights[(size_t)inner * LW_PACKED_MATMUL_COLUMN_TILE];
                }
                sum += expf(value - maximum);
            }
            probabilities[(size_t)row] = 1.0f / sum;
        }
    }
#endif
}
