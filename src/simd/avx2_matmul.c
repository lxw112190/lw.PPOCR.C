#include "simd_kernels.h"

#include "packed_matmul_internal.h"

#include <stddef.h>

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#  include <immintrin.h>
#  define LW_COMPILES_AVX2 1
#else
#  define LW_COMPILES_AVX2 0
#endif

#if LW_COMPILES_AVX2
#  if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,no-fma")))
#  endif
static void lw_avx2_matmul_rows4_tiled_f32(
    const float* input,
    const float* weights,
    float* output,
    uint32_t batch_count,
    uint32_t rows,
    uint32_t inner_dimension,
    uint32_t columns) {
    uint32_t batch;
    for (batch = 0u; batch < batch_count; ++batch) {
        uint32_t row;
        for (row = 0u; row + 4u <= rows; row += 4u) {
            uint32_t column;
            for (column = 0u; column < columns; column += 8u) {
                uint32_t width = columns - column < 8u ? columns - column : 8u;
                __m256 accumulators[4] = {
                    _mm256_setzero_ps(), _mm256_setzero_ps(),
                    _mm256_setzero_ps(), _mm256_setzero_ps()};
                uint32_t inner;
                for (inner = 0u; inner < inner_dimension; ++inner) {
                    const uint64_t weight_base = (uint64_t)inner * columns + column;
                    const uint64_t input_base = ((uint64_t)batch * rows + row) * inner_dimension + inner;
                    __m256 weight_values;
                    if (width == 8u) {
                        weight_values = _mm256_loadu_ps(weights + (size_t)weight_base);
                    } else {
                        /* The final tile may contain fewer than eight columns.
                         * Copy only valid weights before loading the vector so
                         * this fast path never reads past the matrix. */
                        float weight_tail[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                                                0.0f, 0.0f, 0.0f, 0.0f};
                        for (uint32_t lane = 0u; lane < width; ++lane) {
                            weight_tail[lane] = weights[(size_t)weight_base + lane];
                        }
                        weight_values = _mm256_loadu_ps(weight_tail);
                    }
                    uint32_t current_row;
                    for (current_row = 0u; current_row < 4u; ++current_row) {
                        __m256 input_value = _mm256_set1_ps(
                            input[(size_t)(input_base + (uint64_t)current_row * inner_dimension)]);
                        accumulators[current_row] = _mm256_add_ps(
                            accumulators[current_row], _mm256_mul_ps(input_value, weight_values));
                    }
                }
                {
                    uint32_t current_row;
                    for (current_row = 0u; current_row < 4u; ++current_row) {
                        float* destination = output + (size_t)(((uint64_t)batch * rows + row + current_row) * columns + column);
                        if (width == 8u) {
                            _mm256_storeu_ps(destination, accumulators[current_row]);
                        } else {
                            float values[8];
                            _mm256_storeu_ps(values, accumulators[current_row]);
                            for (uint32_t lane = 0u; lane < width; ++lane) {
                                destination[lane] = values[lane];
                            }
                        }
                    }
                }
            }
        }
    }
}
#endif

#if LW_COMPILES_AVX2 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,no-fma")))
#endif
void lw_avx2_matmul_shared_f32(
    const float* input,
    const float* weights,
    float* output,
    uint32_t batch_count,
    uint32_t rows,
    uint32_t inner_dimension,
    uint32_t columns) {
#if LW_COMPILES_AVX2
    if (rows >= 4u && (rows % 4u) == 0u && inner_dimension >= 64u && columns >= 1024u) {
        lw_avx2_matmul_rows4_tiled_f32(input, weights, output, batch_count, rows,
                                       inner_dimension, columns);
        return;
    }
#endif
    uint32_t batch;
    uint32_t row;
    for (batch = 0u; batch < batch_count; ++batch) {
        for (row = 0u; row < rows;) {
            uint32_t row_end = rows - row < 4u ? rows : row + 4u;
            uint32_t inner;
            uint32_t current_row;
            for (current_row = row; current_row < row_end; ++current_row) {
                uint64_t output_base = ((uint64_t)batch * rows + current_row) * columns;
                uint32_t column;
                for (column = 0u; column < columns; ++column) {
                    output[(size_t)(output_base + column)] = 0.0f;
                }
            }
            for (inner = 0u; inner < inner_dimension; ++inner) {
                uint64_t weight_base = (uint64_t)inner * columns;
                for (current_row = row; current_row < row_end; ++current_row) {
                    uint64_t input_base = ((uint64_t)batch * rows + current_row) * inner_dimension;
                    uint64_t output_base = ((uint64_t)batch * rows + current_row) * columns;
                    float input_value = input[(size_t)(input_base + inner)];
                    uint32_t column = 0u;
#if LW_COMPILES_AVX2
                    __m256 input_values = _mm256_set1_ps(input_value);
                    for (; column + 8u <= columns; column += 8u) {
                        __m256 output_values =
                            _mm256_loadu_ps(output + (size_t)(output_base + column));
                        __m256 weight_values =
                            _mm256_loadu_ps(weights + (size_t)(weight_base + column));
                        output_values = _mm256_add_ps(output_values,
                                                      _mm256_mul_ps(input_values, weight_values));
                        _mm256_storeu_ps(output + (size_t)(output_base + column), output_values);
                    }
#endif
                    for (; column < columns; ++column) {
                        output[(size_t)(output_base + column)] +=
                            input_value * weights[(size_t)(weight_base + column)];
                    }
                }
            }
            row = row_end;
        }
    }
}

#if LW_COMPILES_AVX2 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,no-fma")))
#endif
void lw_avx2_packed_matmul_shared_f32(
    const float* input,
    const float* packed_weights,
    float* output,
    uint32_t batch_count,
    uint32_t rows,
    uint32_t inner_dimension,
    uint32_t columns) {
#if LW_COMPILES_AVX2 && (defined(_M_X64) || defined(__x86_64__))
    uint32_t batch;
    const uint32_t column_panels =
        (columns + LW_PACKED_MATMUL_COLUMN_TILE - 1u) / LW_PACKED_MATMUL_COLUMN_TILE;
    if (rows < 4u || rows % 4u != 0u) {
        lw_scalar_packed_matmul_shared_f32(input, packed_weights, output, batch_count, rows,
                                           inner_dimension, columns);
        return;
    }
    for (batch = 0u; batch < batch_count; ++batch) {
        uint32_t row;
        for (row = 0u; row + 4u <= rows; row += 4u) {
            uint32_t column_panel;
            for (column_panel = 0u; column_panel < column_panels; ++column_panel) {
                __m256 accumulator0_low = _mm256_setzero_ps();
                __m256 accumulator0_high = _mm256_setzero_ps();
                __m256 accumulator1_low = _mm256_setzero_ps();
                __m256 accumulator1_high = _mm256_setzero_ps();
                __m256 accumulator2_low = _mm256_setzero_ps();
                __m256 accumulator2_high = _mm256_setzero_ps();
                __m256 accumulator3_low = _mm256_setzero_ps();
                __m256 accumulator3_high = _mm256_setzero_ps();
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
                    accumulator0_low = _mm256_add_ps(
                        accumulator0_low, _mm256_mul_ps(input_value, weight_low));
                    accumulator0_high = _mm256_add_ps(
                        accumulator0_high, _mm256_mul_ps(input_value, weight_high));
                    input_value =
                        _mm256_set1_ps(input[(size_t)(input_base + inner_dimension)]);
                    accumulator1_low = _mm256_add_ps(
                        accumulator1_low, _mm256_mul_ps(input_value, weight_low));
                    accumulator1_high = _mm256_add_ps(
                        accumulator1_high, _mm256_mul_ps(input_value, weight_high));
                    input_value =
                        _mm256_set1_ps(input[(size_t)(input_base + 2u * inner_dimension)]);
                    accumulator2_low = _mm256_add_ps(
                        accumulator2_low, _mm256_mul_ps(input_value, weight_low));
                    accumulator2_high = _mm256_add_ps(
                        accumulator2_high, _mm256_mul_ps(input_value, weight_high));
                    input_value =
                        _mm256_set1_ps(input[(size_t)(input_base + 3u * inner_dimension)]);
                    accumulator3_low = _mm256_add_ps(
                        accumulator3_low, _mm256_mul_ps(input_value, weight_low));
                    accumulator3_high = _mm256_add_ps(
                        accumulator3_high, _mm256_mul_ps(input_value, weight_high));
                }
                {
                    const uint32_t column_base = column_panel * LW_PACKED_MATMUL_COLUMN_TILE;
                    const uint32_t valid_columns =
                        columns - column_base < LW_PACKED_MATMUL_COLUMN_TILE
                            ? columns - column_base
                            : LW_PACKED_MATMUL_COLUMN_TILE;
                    __m256 accumulators[8] = {
                        accumulator0_low, accumulator0_high, accumulator1_low, accumulator1_high,
                        accumulator2_low, accumulator2_high, accumulator3_low, accumulator3_high};
                    uint32_t current_row;
                    for (current_row = 0u; current_row < 4u; ++current_row) {
                        float* destination =
                            output + (size_t)(((uint64_t)batch * rows + row + current_row) *
                                                  columns +
                                              column_base);
                        if (valid_columns == LW_PACKED_MATMUL_COLUMN_TILE) {
                            _mm256_storeu_ps(destination, accumulators[current_row * 2u]);
                            _mm256_storeu_ps(destination + 8u,
                                             accumulators[current_row * 2u + 1u]);
                        } else {
                            float values[LW_PACKED_MATMUL_COLUMN_TILE];
                            _mm256_storeu_ps(values, accumulators[current_row * 2u]);
                            _mm256_storeu_ps(values + 8u, accumulators[current_row * 2u + 1u]);
                            for (uint32_t lane = 0u; lane < valid_columns; ++lane) {
                                destination[lane] = values[lane];
                            }
                        }
                    }
                }
            }
        }
    }
#else
    lw_scalar_packed_matmul_shared_f32(input, packed_weights, output, batch_count, rows,
                                       inner_dimension, columns);
#endif
}
