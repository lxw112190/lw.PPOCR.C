#include "simd_kernels.h"

#include "packed_matmul_internal.h"

#include <stddef.h>

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#  include <immintrin.h>
#  define LW_COMPILES_AVX2_FMA_MATMUL 1
#else
#  define LW_COMPILES_AVX2_FMA_MATMUL 0
#endif

#if LW_COMPILES_AVX2_FMA_MATMUL && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma")))
#endif
void lw_avx2_fma_packed_matmul_bias_argmax_f32(
    const float* input, const float* packed_weights, const float* bias, float* output,
    uint32_t* best_indices, uint32_t batch_count, uint32_t rows,
    uint32_t inner_dimension, uint32_t columns) {
#if LW_COMPILES_AVX2_FMA_MATMUL
    const uint32_t column_panels =
        (columns + LW_PACKED_MATMUL_COLUMN_TILE - 1u) / LW_PACKED_MATMUL_COLUMN_TILE;
    uint32_t batch;
    for (batch = 0u; batch < batch_count; ++batch) {
        uint32_t row;
        for (row = 0u; row + 4u <= rows; row += 4u) {
            uint32_t column_panel;
            for (column_panel = 0u; column_panel < column_panels; ++column_panel) {
                __m256 accumulators[8] = {
                    _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                    _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                    _mm256_setzero_ps(), _mm256_setzero_ps()};
                uint32_t inner;
                for (inner = 0u; inner < inner_dimension; ++inner) {
                    const float* packed =
                        packed_weights +
                        (size_t)(((uint64_t)column_panel * inner_dimension + inner) *
                                 LW_PACKED_MATMUL_COLUMN_TILE);
                    const uint64_t input_base =
                        ((uint64_t)batch * rows + row) * inner_dimension + inner;
                    const __m256 weight_low = _mm256_loadu_ps(packed);
                    const __m256 weight_high = _mm256_loadu_ps(packed + 8u);
                    __m256 input_value = _mm256_set1_ps(input[(size_t)input_base]);
                    accumulators[0] = _mm256_fmadd_ps(input_value, weight_low, accumulators[0]);
                    accumulators[1] = _mm256_fmadd_ps(input_value, weight_high, accumulators[1]);
                    input_value = _mm256_set1_ps(input[(size_t)(input_base + inner_dimension)]);
                    accumulators[2] = _mm256_fmadd_ps(input_value, weight_low, accumulators[2]);
                    accumulators[3] = _mm256_fmadd_ps(input_value, weight_high, accumulators[3]);
                    input_value =
                        _mm256_set1_ps(input[(size_t)(input_base + 2u * inner_dimension)]);
                    accumulators[4] = _mm256_fmadd_ps(input_value, weight_low, accumulators[4]);
                    accumulators[5] = _mm256_fmadd_ps(input_value, weight_high, accumulators[5]);
                    input_value =
                        _mm256_set1_ps(input[(size_t)(input_base + 3u * inner_dimension)]);
                    accumulators[6] = _mm256_fmadd_ps(input_value, weight_low, accumulators[6]);
                    accumulators[7] = _mm256_fmadd_ps(input_value, weight_high, accumulators[7]);
                }
                {
                    const uint32_t column_base = column_panel * LW_PACKED_MATMUL_COLUMN_TILE;
                    const uint32_t valid_columns =
                        columns - column_base < LW_PACKED_MATMUL_COLUMN_TILE
                            ? columns - column_base
                            : LW_PACKED_MATMUL_COLUMN_TILE;
                    uint32_t current_row;
                    for (current_row = 0u; current_row < 4u; ++current_row) {
                        const uint64_t result_row = (uint64_t)batch * rows + row + current_row;
                        float* destination =
                            output + (size_t)(result_row * columns + column_base);
                        float values[LW_PACKED_MATMUL_COLUMN_TILE];
                        uint32_t best_index = column_base == 0u
                                                  ? 0u
                                                  : best_indices[(size_t)result_row];
                        float best_value = column_base == 0u
                                               ? 0.0f
                                               : output[(size_t)(result_row * columns +
                                                                 best_index)];
                        uint32_t lane;
                        _mm256_storeu_ps(values, accumulators[current_row * 2u]);
                        _mm256_storeu_ps(values + 8u, accumulators[current_row * 2u + 1u]);
                        for (lane = 0u; lane < valid_columns; ++lane) {
                            const float value = values[lane] + bias[column_base + lane];
                            destination[lane] = value;
                            if ((column_base != 0u || lane != 0u) && value > best_value) {
                                best_value = value;
                                best_index = column_base + lane;
                            } else if (column_base == 0u && lane == 0u) {
                                best_value = value;
                            }
                        }
                        best_indices[(size_t)result_row] = best_index;
                    }
                }
            }
        }
    }
#else
    (void)input;
    (void)packed_weights;
    (void)bias;
    (void)output;
    (void)best_indices;
    (void)batch_count;
    (void)rows;
    (void)inner_dimension;
    (void)columns;
#endif
}
