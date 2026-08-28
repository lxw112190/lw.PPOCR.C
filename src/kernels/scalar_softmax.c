#include "scalar_kernels.h"
#include "cpu_features.h"
#include "simd_kernels.h"

/* Numerically stable softmax: subtract the row maximum before exponentiation. */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

lw_status lw_scalar_softmax_f32(const float* input, float* output, uint32_t rank,
                                const int32_t* dimensions, int32_t axis) {
    uint64_t outer_count = 1u;
    uint64_t inner_count = 1u;
    uint64_t axis_count;
    uint64_t outer;
    uint64_t inner;
    uint32_t normalized_axis;
    uint32_t index;

    if (input == NULL || output == NULL) {
        return LW_STATUS_INVALID_ARGUMENT;
    }
    if (rank == 0u || rank > LW_MAX_DIMS || dimensions == NULL || axis < -(int32_t)rank ||
        axis >= (int32_t)rank) {
        return LW_STATUS_INVALID_SHAPE;
    }
    normalized_axis = axis < 0 ? (uint32_t)((int32_t)rank + axis) : (uint32_t)axis;
    for (index = 0u; index < rank; ++index) {
        uint64_t dimension;
        if (dimensions[index] <= 0) {
            return LW_STATUS_INVALID_SHAPE;
        }
        dimension = (uint32_t)dimensions[index];
        if (index < normalized_axis) {
            if (outer_count > UINT64_MAX / dimension) {
                return LW_STATUS_OUT_OF_BOUNDS;
            }
            outer_count *= dimension;
        } else if (index > normalized_axis) {
            if (inner_count > UINT64_MAX / dimension) {
                return LW_STATUS_OUT_OF_BOUNDS;
            }
            inner_count *= dimension;
        }
    }
    axis_count = (uint32_t)dimensions[normalized_axis];
    if (outer_count > UINT64_MAX / axis_count ||
        outer_count * axis_count > UINT64_MAX / inner_count ||
        outer_count * axis_count * inner_count > (uint64_t)(SIZE_MAX / sizeof(float))) {
        return LW_STATUS_OUT_OF_BOUNDS;
    }

    /* The REC classifier axis is the contiguous final dimension. Direct row
     * pointers remove the general strided offset arithmetic while preserving
     * the maximum, expf, accumulation and division order exactly. */
    if (inner_count == 1u) {
        if (axis_count >= 256u && lw_detect_simd_level() >= LW_SIMD_LEVEL_AVX2) {
            lw_avx2_softmax_contiguous_f32(input, output, outer_count, axis_count);
            return LW_STATUS_OK;
        }
        for (outer = 0u; outer < outer_count; ++outer) {
            const float* input_row = input + (size_t)(outer * axis_count);
            float* output_row = output + (size_t)(outer * axis_count);
            float maximum = input_row[0];
            float sum = 0.0f;
            uint64_t axis_index;
            for (axis_index = 1u; axis_index < axis_count; ++axis_index) {
                if (input_row[(size_t)axis_index] > maximum) {
                    maximum = input_row[(size_t)axis_index];
                }
            }
            for (axis_index = 0u; axis_index < axis_count; ++axis_index) {
                float value = expf(input_row[(size_t)axis_index] - maximum);
                output_row[(size_t)axis_index] = value;
                sum += value;
            }
            for (axis_index = 0u; axis_index < axis_count; ++axis_index) {
                output_row[(size_t)axis_index] /= sum;
            }
        }
        return LW_STATUS_OK;
    }

    for (outer = 0u; outer < outer_count; ++outer) {
        for (inner = 0u; inner < inner_count; ++inner) {
            uint64_t axis_index;
            uint64_t base = outer * axis_count * inner_count + inner;
            /* Shifting by the maximum preserves the result and prevents expf
             * from overflowing on large logits. */
            float maximum = input[(size_t)base];
            float sum = 0.0f;
            for (axis_index = 1u; axis_index < axis_count; ++axis_index) {
                float value = input[(size_t)(base + axis_index * inner_count)];
                if (value > maximum) {
                    maximum = value;
                }
            }
            for (axis_index = 0u; axis_index < axis_count; ++axis_index) {
                size_t offset = (size_t)(base + axis_index * inner_count);
                float value = expf(input[offset] - maximum);
                output[offset] = value;
                sum += value;
            }
            for (axis_index = 0u; axis_index < axis_count; ++axis_index) {
                size_t offset = (size_t)(base + axis_index * inner_count);
                output[offset] /= sum;
            }
        }
    }
    return LW_STATUS_OK;
}
