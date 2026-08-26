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

static void fill_values(float* values, uint32_t count, uint32_t multiplier, uint32_t modulus,
                        int32_t offset, float divisor) {
    uint32_t index;
    for (index = 0u; index < count; ++index) {
        values[index] = (float)((int32_t)((index * multiplier) % modulus) - offset) / divisor;
    }
}

int main(void) {
    const int32_t normal_input_dimensions[4] = {2, 2, 4, 5};
    const int32_t normal_weight_dimensions[4] = {3, 2, 3, 3};
    const int32_t invalid_weight_dimensions[4] = {3, 1, 3, 3};
    const int32_t normal_output_dimensions[4] = {2, 3, 2, 3};
    const int32_t stride2_input_dimensions[4] = {1, 2, 5, 18};
    const int32_t stride2_weight_dimensions[4] = {3, 2, 3, 3};
    const int32_t stride2_output_dimensions[4] = {1, 3, 3, 9};
    const int32_t invalid_output_dimensions[4] = {2, 3, 2, 2};
    const int32_t normal_kernel[2] = {3, 3};
    const int32_t normal_strides[2] = {2, 2};
    const int32_t unit_dilations[2] = {1, 1};
    const int32_t normal_pads[4] = {1, 1, 1, 1};
    const int32_t grouped_input_dimensions[4] = {1, 4, 4, 4};
    const int32_t grouped_weight_dimensions[4] = {6, 2, 3, 3};
    const int32_t grouped_output_dimensions[4] = {1, 6, 4, 4};
    const int32_t unit_strides[2] = {1, 1};
    const int32_t depthwise_input_dimensions[4] = {1, 3, 4, 5};
    const int32_t depthwise_weight_dimensions[4] = {3, 1, 3, 2};
    const int32_t depthwise_output_dimensions[4] = {1, 3, 4, 5};
    const int32_t depthwise_kernel[2] = {3, 2};
    const int32_t depthwise_dilations[2] = {1, 2};
    const int32_t unit_depthwise_dimensions[4] = {1, 2, 4, 10};
    const int32_t unit_depthwise_weight_dimensions[4] = {2, 1, 3, 3};
    const int32_t asymmetric_input_dimensions[4] = {1, 1, 2, 3};
    const int32_t asymmetric_weight_dimensions[4] = {1, 1, 2, 2};
    const int32_t asymmetric_output_dimensions[4] = {1, 1, 3, 3};
    const int32_t asymmetric_kernel[2] = {2, 2};
    const int32_t asymmetric_strides[2] = {1, 2};
    const int32_t asymmetric_dilations[2] = {2, 1};
    const int32_t asymmetric_pads[4] = {2, 1, 1, 2};
    const int32_t pointwise_input_dimensions[4] = {2, 4, 2, 5};
    const int32_t pointwise_weight_dimensions[4] = {6, 2, 1, 1};
    const int32_t pointwise_output_dimensions[4] = {2, 6, 2, 5};
    const int32_t point_kernel[2] = {1, 1};
    const int32_t no_pads[4] = {0, 0, 0, 0};
    const int32_t batch_norm_dimensions[4] = {2, 3, 2, 2};
    const int32_t transpose_conv_input_dimensions[4] = {1, 2, 2, 2};
    const int32_t transpose_conv_weight_dimensions[4] = {2, 1, 2, 2};
    const int32_t transpose_conv_output_dimensions[4] = {1, 1, 4, 4};
    const int32_t transpose_conv_kernel[2] = {2, 2};
    const int32_t transpose_conv_strides[2] = {2, 2};
    const float normal_bias[3] = {0.25f, -0.5f, 1.0f};
    const float stride2_bias[3] = {-0.125f, 0.625f, -0.875f};
    const float unit_depthwise_bias[2] = {0.375f, -0.625f};
    const float pointwise_bias[6] = {0.25f, -0.5f, 1.0f, -1.25f, 0.75f, 0.5f};
    const float batch_norm_scale[3] = {1.5f, -0.75f, 0.25f};
    const float batch_norm_bias[3] = {0.1f, 0.5f, -1.0f};
    const float batch_norm_mean[3] = {-0.25f, 1.0f, 0.5f};
    const float batch_norm_variance[3] = {0.5f, 2.0f, 0.25f};
    const float invalid_variance[3] = {0.5f, -1.0f, 0.25f};
    const float transpose_conv_bias[1] = {0.125f};
    float normal_input[80];
    float normal_weights[54];
    float normal_output[36];
    float stride2_input[180];
    float stride2_weights[54];
    float stride2_output[81];
    float stride2_dispatched_output[81];
    float stride2_simd_output[81];
    float grouped_input[64];
    float grouped_weights[108];
    float grouped_output[96];
    float depthwise_input[60];
    float depthwise_weights[18];
    float depthwise_output[60];
    float unit_depthwise_input[80];
    float unit_depthwise_weights[18];
    float unit_depthwise_output[80];
    float unit_depthwise_dispatched_output[80];
    float unit_depthwise_simd_output[80];
    float asymmetric_input[6];
    float asymmetric_weights[4];
    float asymmetric_output[9];
    float pointwise_input[80];
    float pointwise_weights[12];
    float pointwise_output[120];
    float pointwise_dispatched_output[120];
    float pointwise_simd_output[120];
    float batch_norm_input[24];
    float batch_norm_output[24];
    float batch_norm_in_place[24];
    float transpose_conv_input[8];
    float transpose_conv_weights[8];
    float transpose_conv_output[16];
    lw_simd_level simd_level;
    lw_status status;

    fill_values(normal_input, 80u, 5u, 19u, 9, 4.0f);
    fill_values(normal_weights, 54u, 7u, 17u, 8, 6.0f);
    status = lw_scalar_conv2d_f32(normal_input, normal_weights, normal_bias, 3u, normal_output,
                                  normal_input_dimensions, normal_weight_dimensions,
                                  normal_output_dimensions, normal_kernel, normal_strides,
                                  unit_dilations, normal_pads, 1u);
    if (!expect_status("normal conv", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("conv", normal_output, 36u);

    fill_values(stride2_input, 180u, 11u, 41u, 20, 10.0f);
    fill_values(stride2_weights, 54u, 13u, 31u, 15, 8.0f);
    lw_scalar_conv3x3_stride2_pad1_f32(stride2_input, stride2_weights, stride2_bias, stride2_output,
                                       stride2_input_dimensions, stride2_output_dimensions);
    simd_level = lw_detect_simd_level();
    if (simd_level >= LW_SIMD_LEVEL_SSE2) {
        lw_sse2_conv3x3_stride2_pad1_f32(stride2_input, stride2_weights, stride2_bias,
                                         stride2_simd_output, stride2_input_dimensions,
                                         stride2_output_dimensions);
        if (memcmp(stride2_output, stride2_simd_output, sizeof(stride2_output)) != 0) {
            fprintf(stderr, "SSE2 stride-2 Conv differs from scalar output\n");
            return 1;
        }
    }
    if (simd_level >= LW_SIMD_LEVEL_AVX2) {
        lw_avx2_conv3x3_stride2_pad1_f32(stride2_input, stride2_weights, stride2_bias,
                                         stride2_simd_output, stride2_input_dimensions,
                                         stride2_output_dimensions);
        if (memcmp(stride2_output, stride2_simd_output, sizeof(stride2_output)) != 0) {
            fprintf(stderr, "AVX2 stride-2 Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(stride2_input, stride2_weights, stride2_bias, 3u,
                                  stride2_dispatched_output, stride2_input_dimensions,
                                  stride2_weight_dimensions, stride2_output_dimensions,
                                  normal_kernel, normal_strides, unit_dilations, normal_pads, 1u);
    if (!expect_status("stride-2 conv", status, LW_STATUS_OK) ||
        memcmp(stride2_output, stride2_dispatched_output, sizeof(stride2_output)) != 0) {
        fprintf(stderr, "dispatched stride-2 Conv differs from scalar output\n");
        return 1;
    }
    print_values("stride2_conv", stride2_output, 81u);

    fill_values(grouped_input, 64u, 3u, 23u, 11, 5.0f);
    fill_values(grouped_weights, 108u, 11u, 29u, 14, 7.0f);
    status = lw_scalar_conv2d_f32(grouped_input, grouped_weights, NULL, 0u, grouped_output,
                                  grouped_input_dimensions, grouped_weight_dimensions,
                                  grouped_output_dimensions, normal_kernel, unit_strides,
                                  unit_dilations, normal_pads, 2u);
    if (!expect_status("grouped conv", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("grouped_conv", grouped_output, 96u);

    fill_values(depthwise_input, 60u, 13u, 31u, 15, 8.0f);
    fill_values(depthwise_weights, 18u, 5u, 13u, 6, 5.0f);
    status = lw_scalar_conv2d_f32(depthwise_input, depthwise_weights, NULL, 0u, depthwise_output,
                                  depthwise_input_dimensions, depthwise_weight_dimensions,
                                  depthwise_output_dimensions, depthwise_kernel, unit_strides,
                                  depthwise_dilations, normal_pads, 3u);
    if (!expect_status("depthwise conv", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("depthwise_conv", depthwise_output, 60u);

    fill_values(unit_depthwise_input, 80u, 17u, 37u, 18, 9.0f);
    fill_values(unit_depthwise_weights, 18u, 7u, 19u, 9, 6.0f);
    lw_scalar_depthwise_conv3x3_unit_pad1_f32(unit_depthwise_input, unit_depthwise_weights,
                                              unit_depthwise_bias, unit_depthwise_output,
                                              unit_depthwise_dimensions);
    if (simd_level >= LW_SIMD_LEVEL_SSE2) {
        lw_sse2_depthwise_conv3x3_unit_pad1_f32(unit_depthwise_input, unit_depthwise_weights,
                                                unit_depthwise_bias, unit_depthwise_simd_output,
                                                unit_depthwise_dimensions);
        if (memcmp(unit_depthwise_output, unit_depthwise_simd_output,
                   sizeof(unit_depthwise_output)) != 0) {
            fprintf(stderr, "SSE2 depthwise Conv differs from scalar output\n");
            return 1;
        }
    }
    if (simd_level >= LW_SIMD_LEVEL_AVX2) {
        lw_avx2_depthwise_conv3x3_unit_pad1_f32(unit_depthwise_input, unit_depthwise_weights,
                                                unit_depthwise_bias, unit_depthwise_simd_output,
                                                unit_depthwise_dimensions);
        if (memcmp(unit_depthwise_output, unit_depthwise_simd_output,
                   sizeof(unit_depthwise_output)) != 0) {
            fprintf(stderr, "AVX2 depthwise Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(unit_depthwise_input, unit_depthwise_weights, unit_depthwise_bias,
                                  2u, unit_depthwise_dispatched_output, unit_depthwise_dimensions,
                                  unit_depthwise_weight_dimensions, unit_depthwise_dimensions,
                                  normal_kernel, unit_strides, unit_dilations, normal_pads, 2u);
    if (!expect_status("unit depthwise conv", status, LW_STATUS_OK) ||
        memcmp(unit_depthwise_output, unit_depthwise_dispatched_output,
               sizeof(unit_depthwise_output)) != 0) {
        fprintf(stderr, "dispatched depthwise Conv differs from scalar output\n");
        return 1;
    }
    print_values("unit_depthwise_conv", unit_depthwise_output, 80u);

    fill_values(asymmetric_input, 6u, 3u, 11u, 5, 4.0f);
    fill_values(asymmetric_weights, 4u, 5u, 13u, 6, 3.0f);
    status = lw_scalar_conv2d_f32(asymmetric_input, asymmetric_weights, NULL, 0u, asymmetric_output,
                                  asymmetric_input_dimensions, asymmetric_weight_dimensions,
                                  asymmetric_output_dimensions, asymmetric_kernel,
                                  asymmetric_strides, asymmetric_dilations, asymmetric_pads, 1u);
    if (!expect_status("asymmetric dilated conv", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("asymmetric_conv", asymmetric_output, 9u);

    fill_values(pointwise_input, 80u, 7u, 19u, 9, 5.0f);
    fill_values(pointwise_weights, 12u, 11u, 23u, 11, 6.0f);
    lw_scalar_conv1x1_unit_f32(pointwise_input, pointwise_weights, pointwise_bias, pointwise_output,
                               pointwise_input_dimensions, pointwise_output_dimensions, 2u, 2u, 3u);
    if (simd_level >= LW_SIMD_LEVEL_SSE2) {
        lw_sse2_conv1x1_unit_f32(pointwise_input, pointwise_weights, pointwise_bias,
                                 pointwise_simd_output, pointwise_input_dimensions,
                                 pointwise_output_dimensions, 2u, 2u, 3u);
        if (memcmp(pointwise_output, pointwise_simd_output, sizeof(pointwise_output)) != 0) {
            fprintf(stderr, "SSE2 pointwise Conv differs from scalar output\n");
            return 1;
        }
    }
    if (simd_level >= LW_SIMD_LEVEL_AVX2) {
        lw_avx2_conv1x1_unit_f32(pointwise_input, pointwise_weights, pointwise_bias,
                                 pointwise_simd_output, pointwise_input_dimensions,
                                 pointwise_output_dimensions, 2u, 2u, 3u);
        if (memcmp(pointwise_output, pointwise_simd_output, sizeof(pointwise_output)) != 0) {
            fprintf(stderr, "AVX2 pointwise Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(pointwise_input, pointwise_weights, pointwise_bias, 6u,
                                  pointwise_dispatched_output, pointwise_input_dimensions,
                                  pointwise_weight_dimensions, pointwise_output_dimensions,
                                  point_kernel, unit_strides, unit_dilations, no_pads, 2u);
    if (!expect_status("grouped pointwise conv", status, LW_STATUS_OK)) {
        return 1;
    }
    if (memcmp(pointwise_output, pointwise_dispatched_output, sizeof(pointwise_output)) != 0) {
        fprintf(stderr, "dispatched pointwise Conv differs from scalar output\n");
        return 1;
    }
    print_values("grouped_pointwise_conv", pointwise_output, 120u);

    fill_values(transpose_conv_input, 8u, 3u, 13u, 6, 4.0f);
    fill_values(transpose_conv_weights, 8u, 5u, 17u, 8, 6.0f);
    status = lw_scalar_conv_transpose2d_f32(
        transpose_conv_input, transpose_conv_weights, transpose_conv_bias, 1u,
        transpose_conv_output, transpose_conv_input_dimensions, transpose_conv_weight_dimensions,
        transpose_conv_output_dimensions, transpose_conv_kernel, transpose_conv_strides,
        unit_dilations, no_pads, 1u);
    if (!expect_status("transpose conv", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("conv_transpose", transpose_conv_output, 16u);

    status = lw_scalar_conv_transpose2d_f32(
        transpose_conv_input, transpose_conv_weights, transpose_conv_bias, 1u, transpose_conv_input,
        transpose_conv_input_dimensions, transpose_conv_weight_dimensions,
        transpose_conv_output_dimensions, transpose_conv_kernel, transpose_conv_strides,
        unit_dilations, no_pads, 1u);
    if (!expect_status("transpose conv alias", status, LW_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    status = lw_scalar_conv_transpose2d_f32(
        transpose_conv_input, transpose_conv_weights, transpose_conv_bias, 1u,
        transpose_conv_output, transpose_conv_input_dimensions, transpose_conv_weight_dimensions,
        transpose_conv_output_dimensions, transpose_conv_kernel, transpose_conv_strides,
        unit_dilations, no_pads, 3u);
    if (!expect_status("transpose conv groups", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }

    fill_values(batch_norm_input, 24u, 7u, 21u, 10, 4.0f);
    status = lw_scalar_batch_normalization_f32(batch_norm_input, batch_norm_scale, batch_norm_bias,
                                               batch_norm_mean, batch_norm_variance, 3u, 1.0e-5f,
                                               batch_norm_output, 4u, batch_norm_dimensions);
    if (!expect_status("batch normalization", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("batch_norm", batch_norm_output, 24u);

    memcpy(batch_norm_in_place, batch_norm_input, sizeof(batch_norm_input));
    status = lw_scalar_batch_normalization_f32(
        batch_norm_in_place, batch_norm_scale, batch_norm_bias, batch_norm_mean,
        batch_norm_variance, 3u, 1.0e-5f, batch_norm_in_place, 4u, batch_norm_dimensions);
    if (!expect_status("in-place batch normalization", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("batch_norm_in_place", batch_norm_in_place, 24u);

    status = lw_scalar_conv2d_f32(normal_input, normal_weights, normal_bias, 3u, normal_output,
                                  normal_input_dimensions, invalid_weight_dimensions,
                                  normal_output_dimensions, normal_kernel, normal_strides,
                                  unit_dilations, normal_pads, 1u);
    if (!expect_status("conv input channels", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_conv2d_f32(normal_input, normal_weights, normal_bias, 3u, normal_output,
                                  normal_input_dimensions, normal_weight_dimensions,
                                  invalid_output_dimensions, normal_kernel, normal_strides,
                                  unit_dilations, normal_pads, 1u);
    if (!expect_status("conv output shape", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_conv2d_f32(normal_input, normal_weights, normal_bias, 2u, normal_output,
                                  normal_input_dimensions, normal_weight_dimensions,
                                  normal_output_dimensions, normal_kernel, normal_strides,
                                  unit_dilations, normal_pads, 1u);
    if (!expect_status("conv bias count", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_conv2d_f32(normal_input, normal_weights, normal_bias, 3u, normal_output,
                                  normal_input_dimensions, normal_weight_dimensions,
                                  normal_output_dimensions, normal_kernel, normal_strides,
                                  unit_dilations, normal_pads, 0u);
    if (!expect_status("conv zero groups", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_conv2d_f32(normal_input, normal_weights, normal_bias, 3u, normal_input,
                                  normal_input_dimensions, normal_weight_dimensions,
                                  normal_output_dimensions, normal_kernel, normal_strides,
                                  unit_dilations, normal_pads, 1u);
    if (!expect_status("conv alias", status, LW_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    status = lw_scalar_batch_normalization_f32(batch_norm_input, batch_norm_scale, batch_norm_bias,
                                               batch_norm_mean, batch_norm_variance, 2u, 1.0e-5f,
                                               batch_norm_output, 4u, batch_norm_dimensions);
    if (!expect_status("batch normalization parameter count", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_batch_normalization_f32(batch_norm_input, batch_norm_scale, batch_norm_bias,
                                               batch_norm_mean, invalid_variance, 3u, 1.0e-5f,
                                               batch_norm_output, 4u, batch_norm_dimensions);
    if (!expect_status("batch normalization variance", status, LW_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    status = lw_scalar_batch_normalization_f32(batch_norm_input, batch_norm_scale, batch_norm_bias,
                                               batch_norm_mean, batch_norm_variance, 3u, 0.0f,
                                               batch_norm_output, 4u, batch_norm_dimensions);
    if (!expect_status("batch normalization epsilon", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    if (is_32_bit_process()) {
        const int32_t large_dimensions[4] = {1, 1, 65536, 65536};
        const int32_t point_weight_dimensions[4] = {1, 1, 1, 1};
        status = lw_scalar_conv2d_f32(normal_input, normal_weights, NULL, 0u, normal_output,
                                      large_dimensions, point_weight_dimensions, large_dimensions,
                                      point_kernel, unit_strides, unit_dilations, no_pads, 1u);
        if (!expect_status("conv 32-bit byte overflow", status, LW_STATUS_OUT_OF_BOUNDS)) {
            return 1;
        }
    }
    return 0;
}
