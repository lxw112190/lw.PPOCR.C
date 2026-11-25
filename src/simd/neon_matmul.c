#include "simd_kernels.h"
#include "packed_matmul_internal.h"

#include <stddef.h>
#include <string.h>

#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#include <arm_neon.h>
#define LW_COMPILES_NEON_MATMUL 1
#else
#define LW_COMPILES_NEON_MATMUL 0
#endif

void lw_neon_packed_matmul_shared_f32(const float* input, const float* packed_weights,
                                      const float* bias, float* output, uint32_t batch_count,
                                      uint32_t rows, uint32_t inner_dimension, uint32_t columns) {
#if LW_COMPILES_NEON_MATMUL
    for (uint32_t batch = 0u; batch < batch_count; ++batch) {
        const float* batch_input = input + (size_t)batch * rows * inner_dimension;
        float* batch_output = output + (size_t)batch * rows * columns;
        const float* panel = packed_weights;
        for (uint32_t column = 0u; column < columns;) {
            const uint32_t remaining = columns - column;
            const uint32_t lanes = remaining < 16u ? remaining : 16u;
            /* Reuse the contiguous panel across every row pair before moving to
             * the next panel. Each pair keeps both output vectors in registers. */
            for (uint32_t row = 0u; row < rows;) {
                const int has_second = row + 1u < rows;
                const float* x0 = batch_input + (size_t)row * inner_dimension;
                const float* x1 = has_second ? x0 + inner_dimension : x0;
                float* y0 = batch_output + (size_t)row * columns + column;
                float* y1 = has_second ? y0 + columns : y0;
                const float* weights = panel;
                const float32x4_t zero = vdupq_n_f32(0.0f);
                float32x4_t a00 = zero, a01 = zero, a02 = zero, a03 = zero;
                float32x4_t a10 = zero, a11 = zero, a12 = zero, a13 = zero;
                for (uint32_t k = 0u; k < inner_dimension; ++k) {
                    const float32x4_t w0 = vld1q_f32(weights);
                    const float32x4_t w1 = vld1q_f32(weights + 4u);
                    const float32x4_t w2 = vld1q_f32(weights + 8u);
                    const float32x4_t w3 = vld1q_f32(weights + 12u);
                    const float32x4_t v0 = vdupq_n_f32(x0[k]);
                    const float32x4_t v1 = vdupq_n_f32(x1[k]);
                    /* Use the same multiply-add expression and FP contraction
                     * policy as the canonical shared-weight MatMul. */
                    a00 = w0 * v0 + a00;
                    a01 = w1 * v0 + a01;
                    a02 = w2 * v0 + a02;
                    a03 = w3 * v0 + a03;
                    a10 = w0 * v1 + a10;
                    a11 = w1 * v1 + a11;
                    a12 = w2 * v1 + a12;
                    a13 = w3 * v1 + a13;
                    weights += LW_PACKED_MATMUL_COLUMN_TILE;
                }
                if (lanes == 16u) {
                    if (bias != NULL) {
                        const float32x4_t b0 = vld1q_f32(bias + column);
                        const float32x4_t b1 = vld1q_f32(bias + column + 4u);
                        const float32x4_t b2 = vld1q_f32(bias + column + 8u);
                        const float32x4_t b3 = vld1q_f32(bias + column + 12u);
                        a00 += b0;
                        a01 += b1;
                        a02 += b2;
                        a03 += b3;
                        a10 += b0;
                        a11 += b1;
                        a12 += b2;
                        a13 += b3;
                    }
                    vst1q_f32(y0, a00);
                    vst1q_f32(y0 + 4u, a01);
                    vst1q_f32(y0 + 8u, a02);
                    vst1q_f32(y0 + 12u, a03);
                    if (has_second) {
                        vst1q_f32(y1, a10);
                        vst1q_f32(y1 + 4u, a11);
                        vst1q_f32(y1 + 8u, a12);
                        vst1q_f32(y1 + 12u, a13);
                    }
                } else {
                    float first[16], second[16];
                    vst1q_f32(first, a00);
                    vst1q_f32(first + 4u, a01);
                    vst1q_f32(first + 8u, a02);
                    vst1q_f32(first + 12u, a03);
                    vst1q_f32(second, a10);
                    vst1q_f32(second + 4u, a11);
                    vst1q_f32(second + 8u, a12);
                    vst1q_f32(second + 12u, a13);
                    if (bias != NULL) {
                        for (uint32_t lane = 0u; lane < lanes; ++lane) {
                            first[lane] += bias[column + lane];
                            second[lane] += bias[column + lane];
                        }
                    }
                    memcpy(y0, first, (size_t)lanes * sizeof(float));
                    if (has_second) {
                        memcpy(y1, second, (size_t)lanes * sizeof(float));
                    }
                }
                row += has_second ? 2u : 1u;
            }
            column += lanes;
            panel += (size_t)inner_dimension * LW_PACKED_MATMUL_COLUMN_TILE;
        }
    }
#else
    lw_scalar_packed_matmul_shared_f32(input, packed_weights, output, batch_count, rows,
                                       inner_dimension, columns);
    if (bias != NULL) {
        for (size_t row = 0u; row < (size_t)batch_count * rows; ++row) {
            for (uint32_t col = 0u; col < columns; ++col) {
                output[row * columns + col] += bias[col];
            }
        }
    }
#endif
}
