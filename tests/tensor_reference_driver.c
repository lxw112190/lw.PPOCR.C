#include "scalar_kernels.h"
#include "cpu_features.h"
#include "simd_kernels.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void print_values(const char* name, const float* values, uint64_t count) {
    uint64_t index;
    printf("%s %" PRIu64, name, count);
    for (index = 0u; index < count; ++index) {
        printf(" %.9g", (double)values[index]);
    }
    putchar('\n');
}

static int expect_status(const char* name, lw_status actual, lw_status expected) {
    if (actual != expected) {
        fprintf(stderr, "%s: expected %s, got %s\n", name, lw_status_string(expected),
                lw_status_string(actual));
        return 0;
    }
    return 1;
}

static int is_32_bit_process(void) {
    volatile size_t pointer_size = sizeof(size_t);
    return pointer_size == 4u;
}

int main(void) {
    const int32_t transpose_input_dimensions[3] = {2, 3, 4};
    const int32_t transpose_output_dimensions[3] = {2, 4, 3};
    const int32_t invalid_transpose_output[3] = {2, 3, 4};
    const int32_t permutation[3] = {0, 2, 1};
    const int32_t duplicate_permutation[3] = {0, 1, 1};
    const int32_t squeeze_input_dimensions[4] = {2, 1, 3, 1};
    const int32_t squeeze_output_dimensions[2] = {2, 3};
    const int32_t squeeze_axes[2] = {1, -1};
    const int32_t invalid_squeeze_axis[1] = {2};
    const int32_t unsqueeze_input_dimensions[2] = {2, 3};
    const int32_t unsqueeze_output_dimensions[4] = {1, 2, 3, 1};
    const int32_t reshape_output_dimensions[2] = {2, 3};
    const int32_t invalid_reshape_output_dimensions[2] = {2, 4};
    const int32_t unsqueeze_axes[2] = {0, -1};
    const int32_t duplicate_unsqueeze_axes[2] = {0, -4};
    const int32_t reduce_input_dimensions[3] = {2, 3, 4};
    const int32_t reduce_output_dimensions[3] = {2, 1, 1};
    const int32_t reduce_axes[2] = {1, -1};
    const int32_t duplicate_reduce_axes[2] = {1, -2};
    const int32_t reduce_spatial_input_dimensions[4] = {2, 2, 2, 3};
    const int32_t reduce_spatial_output_dimensions[4] = {2, 2, 1, 1};
    const int32_t reduce_spatial_axes[2] = {2, 3};
    const int32_t pool_input_dimensions[4] = {1, 2, 3, 5};
    const int32_t pool_output_dimensions[4] = {1, 2, 3, 3};
    const int32_t invalid_pool_output[4] = {1, 2, 2, 3};
    const int32_t pool_kernel[2] = {2, 3};
    const int32_t pool_strides[2] = {1, 2};
    const int32_t pool_pads[4] = {1, 1, 0, 1};
    const int32_t max_pool_same_input_dimensions[4] = {2, 2, 2, 3};
    const int32_t max_pool_same_output_dimensions[4] = {2, 2, 2, 3};
    const int32_t max_pool_same_kernel[2] = {2, 2};
    const int32_t max_pool_same_strides[2] = {1, 1};
    const int32_t max_pool_same_pads[4] = {0, 0, 1, 1};
    const int32_t concat_left_dimensions[3] = {1, 2, 3};
    const int32_t concat_right_dimensions[3] = {1, 1, 3};
    const int32_t concat_output_dimensions[3] = {1, 3, 3};
    const uint32_t concat_ranks[2] = {3u, 3u};
    const int32_t resize_input_dimensions[4] = {1, 1, 2, 3};
    const int32_t resize_output_dimensions[4] = {1, 1, 4, 6};
    const float resize_scales[4] = {1.0f, 1.0f, 2.0f, 2.0f};
    const int32_t resize_multi_input_dimensions[4] = {2, 2, 2, 3};
    const int32_t resize_multi_output_dimensions[4] = {2, 2, 4, 9};
    const float resize_multi_scales[4] = {1.0f, 1.0f, 2.0f, 3.0f};
    const int32_t resize_fractional_output_dimensions[4] = {1, 1, 3, 6};
    const float resize_fractional_scales[4] = {1.0f, 1.0f, 1.5f, 2.0f};
    float tensor_input[30];
    float transpose_output[24];
    float reshape_output[6];
    float reduce_output[24];
    float reduce_spatial_output[4];
    float pool_output[18];
    float max_pool_same_output[24];
    float concat_left[6] = {-3.0f, -2.0f, -1.0f, 1.0f, 2.0f, 3.0f};
    float concat_right[3] = {4.0f, 5.0f, 6.0f};
    const float* concat_inputs[2] = {concat_left, concat_right};
    const int32_t* concat_dimensions[2] = {concat_left_dimensions, concat_right_dimensions};
    float concat_output[9];
    float resize_output[24];
    float resize_multi_output[144];
    float resize_fractional_output[18];
    float matmul_input[24];
    float matmul_weights[20];
    float matmul_output[30];
    float matmul_dispatched_output[30];
    float matmul_simd_output[30];
    uint32_t index;
    lw_simd_level simd_level;
    lw_status status;

    for (index = 0u; index < 30u; ++index) {
        tensor_input[index] = (float)((int32_t)((index * 5u) % 17u) - 8) / 3.0f;
    }
    status = lw_scalar_transpose_f32(tensor_input, transpose_output, 3u, transpose_input_dimensions,
                                     3u, permutation, transpose_output_dimensions);
    if (!expect_status("transpose", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("transpose", transpose_output, 24u);

    status = lw_scalar_squeeze_f32(tensor_input, reshape_output, 4u, squeeze_input_dimensions, 2u,
                                   squeeze_axes, 2u, squeeze_output_dimensions);
    if (!expect_status("squeeze", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("squeeze", reshape_output, 6u);

    status = lw_scalar_squeeze_f32(tensor_input, reshape_output, 4u, squeeze_input_dimensions, 0u,
                                   NULL, 2u, squeeze_output_dimensions);
    if (!expect_status("squeeze_all", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("squeeze_all", reshape_output, 6u);

    status = lw_scalar_unsqueeze_f32(tensor_input, reshape_output, 2u, unsqueeze_input_dimensions,
                                     2u, unsqueeze_axes, 4u, unsqueeze_output_dimensions);
    if (!expect_status("unsqueeze", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("unsqueeze", reshape_output, 6u);

    status = lw_scalar_reshape_f32(tensor_input, reshape_output, 4u, squeeze_input_dimensions, 2u,
                                   reshape_output_dimensions);
    if (!expect_status("reshape", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("reshape", reshape_output, 6u);

    status = lw_scalar_reduce_mean_f32(tensor_input, reduce_output, 3u, reduce_input_dimensions, 2u,
                                       reduce_axes, 1u, 0u, 3u, reduce_output_dimensions);
    if (!expect_status("reduce_mean", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("reduce_mean", reduce_output, 2u);

    status = lw_scalar_reduce_mean_f32(tensor_input, reduce_output, 3u, reduce_input_dimensions, 0u,
                                       NULL, 1u, 1u, 3u, reduce_input_dimensions);
    if (!expect_status("reduce_noop", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("reduce_noop", reduce_output, 24u);

    status = lw_scalar_reduce_mean_f32(
        tensor_input, reduce_spatial_output, 4u, reduce_spatial_input_dimensions, 2u,
        reduce_spatial_axes, 1u, 0u, 4u, reduce_spatial_output_dimensions);
    if (!expect_status("NCHW spatial reduce mean", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("reduce_mean_nchw_spatial", reduce_spatial_output, 4u);

    status = lw_scalar_average_pool2d_f32(tensor_input, pool_output, pool_input_dimensions,
                                          pool_output_dimensions, pool_kernel, pool_strides,
                                          pool_pads, 0u, 0u);
    if (!expect_status("average_pool", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("average_pool", pool_output, 18u);

    status = lw_scalar_average_pool2d_f32(tensor_input, pool_output, pool_input_dimensions,
                                          pool_output_dimensions, pool_kernel, pool_strides,
                                          pool_pads, 0u, 1u);
    if (!expect_status("average_pool_include_pad", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("average_pool_include_pad", pool_output, 18u);

    status =
        lw_scalar_max_pool2d_f32(tensor_input, pool_output, pool_input_dimensions,
                                 pool_output_dimensions, pool_kernel, pool_strides, pool_pads, 0u);
    if (!expect_status("max_pool", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("max_pool", pool_output, 18u);

    status = lw_scalar_max_pool2d_f32(
        tensor_input, max_pool_same_output, max_pool_same_input_dimensions,
        max_pool_same_output_dimensions, max_pool_same_kernel, max_pool_same_strides,
        max_pool_same_pads, 0u);
    if (!expect_status("2x2 SAME_UPPER max pool", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("max_pool_same_upper_2x2", max_pool_same_output, 24u);

    status = lw_scalar_concat_f32(concat_inputs, 2u, concat_ranks, concat_dimensions, concat_output,
                                  3u, concat_output_dimensions, 1);
    if (!expect_status("concat", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("concat", concat_output, 9u);

    status = lw_scalar_resize_nearest_f32(concat_left, resize_output, 4u, resize_input_dimensions,
                                          resize_output_dimensions, resize_scales);
    if (!expect_status("resize", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("resize_nearest", resize_output, 24u);

    status = lw_scalar_resize_nearest_f32(tensor_input, resize_multi_output, 4u,
                                          resize_multi_input_dimensions,
                                          resize_multi_output_dimensions, resize_multi_scales);
    if (!expect_status("multi-plane resize", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("resize_nearest_nchw", resize_multi_output, 144u);

    /* A fractional height scale must retain the general coordinate path. */
    status = lw_scalar_resize_nearest_f32(
        concat_left, resize_fractional_output, 4u, resize_input_dimensions,
        resize_fractional_output_dimensions, resize_fractional_scales);
    if (!expect_status("fractional resize", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("resize_nearest_fractional", resize_fractional_output, 18u);

    for (index = 0u; index < 24u; ++index) {
        matmul_input[index] = (float)((int32_t)((index * 3u) % 13u) - 6) / 4.0f;
    }
    for (index = 0u; index < 20u; ++index) {
        matmul_weights[index] = (float)((int32_t)((index * 7u) % 11u) - 5) / 5.0f;
    }
    status =
        lw_scalar_matmul_shared_f32(matmul_input, matmul_weights, matmul_output, 2u, 3u, 4u, 5u);
    if (!expect_status("matmul", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("matmul", matmul_output, 30u);
    simd_level = lw_detect_simd_level();
    if (simd_level >= LW_SIMD_LEVEL_SSE2) {
        lw_sse2_matmul_shared_f32(matmul_input, matmul_weights, matmul_simd_output, 2u, 3u, 4u, 5u);
        if (memcmp(matmul_output, matmul_simd_output, sizeof(matmul_output)) != 0) {
            fprintf(stderr, "SSE2 matmul differs from scalar output\n");
            return 1;
        }
    }
    if (simd_level >= LW_SIMD_LEVEL_AVX2) {
        lw_avx2_matmul_shared_f32(matmul_input, matmul_weights, matmul_simd_output, 2u, 3u, 4u, 5u);
        if (memcmp(matmul_output, matmul_simd_output, sizeof(matmul_output)) != 0) {
            fprintf(stderr, "AVX2 matmul differs from scalar output\n");
            return 1;
        }
    }
    status = lw_matmul_shared_f32(matmul_input, matmul_weights, matmul_dispatched_output, 2u, 3u,
                                  4u, 5u);
    if (!expect_status("dispatched matmul", status, LW_STATUS_OK)) {
        return 1;
    }
    if (memcmp(matmul_output, matmul_dispatched_output, sizeof(matmul_output)) != 0) {
        fprintf(stderr, "dispatched matmul differs from scalar output\n");
        return 1;
    }
    print_values("matmul_dispatched", matmul_dispatched_output, 30u);

    status = lw_scalar_transpose_f32(tensor_input, transpose_output, 3u, transpose_input_dimensions,
                                     3u, permutation, invalid_transpose_output);
    if (!expect_status("transpose output shape", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_transpose_f32(tensor_input, transpose_output, 3u, transpose_input_dimensions,
                                     3u, duplicate_permutation, transpose_output_dimensions);
    if (!expect_status("transpose duplicate axis", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_squeeze_f32(tensor_input, reshape_output, 4u, squeeze_input_dimensions, 1u,
                                   invalid_squeeze_axis, 3u, NULL);
    if (!expect_status("squeeze non-unit axis", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_unsqueeze_f32(tensor_input, reshape_output, 2u, unsqueeze_input_dimensions,
                                     2u, duplicate_unsqueeze_axes, 4u, unsqueeze_output_dimensions);
    if (!expect_status("unsqueeze duplicate axis", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_reshape_f32(tensor_input, reshape_output, 4u, squeeze_input_dimensions, 2u,
                                   invalid_reshape_output_dimensions);
    if (!expect_status("reshape element count", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_reduce_mean_f32(tensor_input, reduce_output, 3u, reduce_input_dimensions, 2u,
                                       duplicate_reduce_axes, 1u, 0u, 3u, reduce_output_dimensions);
    if (!expect_status("reduce duplicate axis", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_average_pool2d_f32(tensor_input, pool_output, pool_input_dimensions,
                                          invalid_pool_output, pool_kernel, pool_strides, pool_pads,
                                          0u, 0u);
    if (!expect_status("pool output shape", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_matmul_shared_f32(matmul_input, matmul_weights, matmul_output, 0u, 3u, 4u, 5u);
    if (!expect_status("matmul empty batch", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status =
        lw_scalar_matmul_shared_f32(matmul_input, matmul_weights, matmul_input, 2u, 3u, 4u, 5u);
    if (!expect_status("matmul alias", status, LW_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    if (is_32_bit_process()) {
        status = lw_scalar_matmul_shared_f32(matmul_input, matmul_weights, matmul_output, 1u,
                                             65536u, 65536u, 1u);
        if (!expect_status("matmul 32-bit byte overflow", status, LW_STATUS_OUT_OF_BOUNDS)) {
            return 1;
        }
    }
    return 0;
}
