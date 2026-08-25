#include "scalar_kernels.h"

#include <stddef.h>
#include <stdint.h>

static int multiply_fits(uint64_t left, uint64_t right, uint64_t* result) {
    if (right != 0u && left > UINT64_MAX / right) {
        return 0;
    }
    *result = left * right;
    return 1;
}

static int buffer_fits(uint32_t first, uint32_t second, uint32_t third) {
    uint64_t count;
    if (!multiply_fits(first, second, &count) ||
        !multiply_fits(count, third, &count)) {
        return 0;
    }
    return count <= (uint64_t)(SIZE_MAX / sizeof(float));
}

lw_status lw_scalar_matmul_shared_f32(
    const float* input,
    const float* weights,
    float* output,
    uint32_t batch_count,
    uint32_t rows,
    uint32_t inner_dimension,
    uint32_t columns) {
    uint32_t batch;
    uint32_t row;
    uint32_t column;
    if (input == NULL || weights == NULL || output == NULL ||
        output == input || output == weights) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (batch_count == 0u || rows == 0u || inner_dimension == 0u || columns == 0u) {
        return LW_STATUS_INVALID_SHAPE;
    }
    if (!buffer_fits(batch_count, rows, inner_dimension) ||
        !buffer_fits(1u, inner_dimension, columns) ||
        !buffer_fits(batch_count, rows, columns)) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }
    for (batch = 0u; batch < batch_count; ++batch) {
        for (row = 0u; row < rows; ++row) {
            uint64_t input_base =
                ((uint64_t)batch * rows + row) * inner_dimension;
            uint64_t output_base = ((uint64_t)batch * rows + row) * columns;
            for (column = 0u; column < columns; ++column) {
                uint32_t inner;
                float sum = 0.0f;
                for (inner = 0u; inner < inner_dimension; ++inner) {
                    sum += input[(size_t)(input_base + inner)] *
                           weights[(size_t)((uint64_t)inner * columns + column)];
                }
                output[(size_t)(output_base + column)] = sum;
            }
        }
    }
    return LW_STATUS_OK;
}
