#include "simd_kernels.h"

#include <stddef.h>

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#  include <emmintrin.h>
#  define LW_COMPILES_SSE2 1
#else
#  define LW_COMPILES_SSE2 0
#endif

#if LW_COMPILES_SSE2 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("sse2")))
#endif
void lw_sse2_matmul_shared_f32(
    const float* input,
    const float* weights,
    float* output,
    uint32_t batch_count,
    uint32_t rows,
    uint32_t inner_dimension,
    uint32_t columns) {
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
#if LW_COMPILES_SSE2
                    __m128 input_values = _mm_set1_ps(input_value);
                    for (; column + 4u <= columns; column += 4u) {
                        __m128 output_values =
                            _mm_loadu_ps(output + (size_t)(output_base + column));
                        __m128 weight_values =
                            _mm_loadu_ps(weights + (size_t)(weight_base + column));
                        output_values =
                            _mm_add_ps(output_values, _mm_mul_ps(input_values, weight_values));
                        _mm_storeu_ps(output + (size_t)(output_base + column), output_values);
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
