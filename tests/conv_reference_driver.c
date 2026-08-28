#include "scalar_kernels.h"
#include "cpu_features.h"
#include "packed_conv_internal.h"
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
    const int32_t stride2_four_output_dimensions[4] = {1, 4, 3, 9};
    const int32_t unit_conv_input_dimensions[4] = {1, 3, 5, 19};
    const int32_t unit_conv_weight_dimensions[4] = {2, 3, 3, 3};
    const int32_t unit_conv_output_dimensions[4] = {1, 2, 5, 19};
    const int32_t unit_conv_four_output_dimensions[4] = {1, 4, 5, 19};
    const int32_t unit_conv2x2_weight_dimensions[4] = {2, 3, 2, 2};
    const int32_t unit_conv2x2_kernel[2] = {2, 2};
    const int32_t unit_conv2x2_pads[4] = {0, 0, 1, 1};
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
    const int32_t unit_depthwise5x5_dimensions[4] = {1, 2, 6, 19};
    const int32_t unit_depthwise5x5_weight_dimensions[4] = {2, 1, 5, 5};
    const int32_t depthwise5x5_kernel[2] = {5, 5};
    const int32_t pad2[4] = {2, 2, 2, 2};
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
    const int32_t packed_pointwise_input_dimensions[4] = {2, 5, 3, 7};
    const int32_t packed_pointwise_output_dimensions[4] = {2, 7, 3, 7};
    const int32_t point_kernel[2] = {1, 1};
    const int32_t no_pads[4] = {0, 0, 0, 0};
    const int32_t batch_norm_dimensions[4] = {2, 3, 2, 2};
    const int32_t transpose_conv_input_dimensions[4] = {1, 2, 2, 2};
    const int32_t transpose_conv_weight_dimensions[4] = {2, 1, 2, 2};
    const int32_t transpose_conv_output_dimensions[4] = {1, 1, 4, 4};
    const int32_t transpose_conv_kernel[2] = {2, 2};
    const int32_t transpose_conv_strides[2] = {2, 2};
    const int32_t transpose_conv_simd_input_dimensions[4] = {1, 2, 3, 8};
    const int32_t transpose_conv_simd_weight_dimensions[4] = {2, 3, 2, 2};
    const int32_t transpose_conv_simd_output_dimensions[4] = {1, 3, 6, 16};
    const float normal_bias[3] = {0.25f, -0.5f, 1.0f};
    const float stride2_bias[3] = {-0.125f, 0.625f, -0.875f};
    const float stride2_four_bias[4] = {-0.125f, 0.625f, -0.875f, 0.375f};
    const float unit_conv_bias[2] = {0.375f, -0.625f};
    const float unit_conv_four_bias[4] = {0.375f, -0.625f, 0.125f, -0.875f};
    const float unit_depthwise_bias[2] = {0.375f, -0.625f};
    const float pointwise_bias[6] = {0.25f, -0.5f, 1.0f, -1.25f, 0.75f, 0.5f};
    const float packed_pointwise_bias[7] = {0.25f, -0.5f, 1.0f, -1.25f, 0.75f, 0.5f, -0.125f};
    const float batch_norm_scale[3] = {1.5f, -0.75f, 0.25f};
    const float batch_norm_bias[3] = {0.1f, 0.5f, -1.0f};
    const float batch_norm_mean[3] = {-0.25f, 1.0f, 0.5f};
    const float batch_norm_variance[3] = {0.5f, 2.0f, 0.25f};
    const float invalid_variance[3] = {0.5f, -1.0f, 0.25f};
    const float transpose_conv_bias[1] = {0.125f};
    const float transpose_conv_simd_bias[3] = {0.125f, -0.25f, 0.5f};
    float normal_input[80];
    float normal_weights[54];
    float normal_output[36];
    float stride2_input[180];
    float stride2_weights[54];
    float stride2_output[81];
    float stride2_dispatched_output[81];
    float stride2_simd_output[81];
    float stride2_four_weights[72];
    float stride2_four_output[108];
    float stride2_four_simd_output[108];
    float unit_conv_input[285];
    float unit_conv_weights[54];
    float unit_conv_output[190];
    float unit_conv_dispatched_output[190];
    float unit_conv_simd_output[190];
    float unit_conv_four_weights[108];
    float unit_conv_four_output[380];
    float unit_conv_four_simd_output[380];
    float unit_conv2x2_weights[24];
    float unit_conv2x2_output[190];
    float unit_conv2x2_dispatched_output[190];
    float unit_conv2x2_simd_output[190];
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
    float unit_depthwise5x5_input[228];
    float unit_depthwise5x5_weights[50];
    float unit_depthwise5x5_output[228];
    float unit_depthwise5x5_dispatched_output[228];
    float unit_depthwise5x5_simd_output[228];
    float asymmetric_input[6];
    float asymmetric_weights[4];
    float asymmetric_output[9];
    float pointwise_input[80];
    float pointwise_weights[12];
    float pointwise_output[120];
    float pointwise_dispatched_output[120];
    float pointwise_simd_output[120];
    float packed_pointwise_input[210];
    float packed_pointwise_weights[35];
    float packed_pointwise_packed_weights[40];
    float packed_pointwise_output[294];
    float packed_pointwise_simd_output[294];
    float batch_norm_input[24];
    float batch_norm_output[24];
    float batch_norm_in_place[24];
    float transpose_conv_input[8];
    float transpose_conv_weights[8];
    float transpose_conv_output[16];
    float transpose_conv_simd_input[48];
    float transpose_conv_simd_weights[24];
    float transpose_conv_simd_output[288];
    float transpose_conv_sse2_output[288];
    float transpose_conv_avx2_output[288];
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

    /* Four output channels exercise the AVX2 kernel that reuses each gathered
     * input vector across four independent NCHW output planes. */
    fill_values(stride2_four_weights, 72u, 19u, 47u, 23, 12.0f);
    lw_scalar_conv3x3_stride2_pad1_f32(stride2_input, stride2_four_weights, stride2_four_bias,
                                       stride2_four_output, stride2_input_dimensions,
                                       stride2_four_output_dimensions);
    if (simd_level >= LW_SIMD_LEVEL_AVX2) {
        lw_avx2_conv3x3_stride2_pad1_f32(stride2_input, stride2_four_weights, stride2_four_bias,
                                         stride2_four_simd_output, stride2_input_dimensions,
                                         stride2_four_output_dimensions);
        if (memcmp(stride2_four_output, stride2_four_simd_output, sizeof(stride2_four_output)) !=
            0) {
            fprintf(stderr, "AVX2 four-output stride-2 Conv differs from scalar output\n");
            return 1;
        }
    }

    fill_values(unit_conv_input, 285u, 17u, 43u, 21, 11.0f);
    fill_values(unit_conv_weights, 54u, 19u, 37u, 18, 9.0f);
    lw_scalar_conv3x3_unit_pad1_f32(unit_conv_input, unit_conv_weights, unit_conv_bias,
                                    unit_conv_output, unit_conv_input_dimensions,
                                    unit_conv_output_dimensions);
    if (simd_level >= LW_SIMD_LEVEL_SSE2) {
        lw_sse2_conv3x3_unit_pad1_f32(unit_conv_input, unit_conv_weights, unit_conv_bias,
                                      unit_conv_simd_output, unit_conv_input_dimensions,
                                      unit_conv_output_dimensions);
        if (memcmp(unit_conv_output, unit_conv_simd_output, sizeof(unit_conv_output)) != 0) {
            fprintf(stderr, "SSE2 unit-stride Conv differs from scalar output\n");
            return 1;
        }
    }
    if (simd_level >= LW_SIMD_LEVEL_AVX2) {
        lw_avx2_conv3x3_unit_pad1_f32(unit_conv_input, unit_conv_weights, unit_conv_bias,
                                      unit_conv_simd_output, unit_conv_input_dimensions,
                                      unit_conv_output_dimensions);
        if (memcmp(unit_conv_output, unit_conv_simd_output, sizeof(unit_conv_output)) != 0) {
            fprintf(stderr, "AVX2 unit-stride Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(unit_conv_input, unit_conv_weights, unit_conv_bias, 2u,
                                  unit_conv_dispatched_output, unit_conv_input_dimensions,
                                  unit_conv_weight_dimensions, unit_conv_output_dimensions,
                                  normal_kernel, unit_strides, unit_dilations, normal_pads, 1u);
    if (!expect_status("unit-stride conv", status, LW_STATUS_OK) ||
        memcmp(unit_conv_output, unit_conv_dispatched_output, sizeof(unit_conv_output)) != 0) {
        fprintf(stderr, "dispatched unit-stride Conv differs from scalar output\n");
        return 1;
    }
    print_values("unit_stride_conv", unit_conv_output, 190u);

    /* Four output channels exercise the AVX2 unit-stride kernel that reuses
     * every input vector across four independent NCHW output planes. */
    fill_values(unit_conv_four_weights, 108u, 23u, 47u, 23, 12.0f);
    lw_scalar_conv3x3_unit_pad1_f32(unit_conv_input, unit_conv_four_weights, unit_conv_four_bias,
                                    unit_conv_four_output, unit_conv_input_dimensions,
                                    unit_conv_four_output_dimensions);
    if (simd_level >= LW_SIMD_LEVEL_AVX2) {
        lw_avx2_conv3x3_unit_pad1_f32(unit_conv_input, unit_conv_four_weights, unit_conv_four_bias,
                                      unit_conv_four_simd_output, unit_conv_input_dimensions,
                                      unit_conv_four_output_dimensions);
        if (memcmp(unit_conv_four_output, unit_conv_four_simd_output,
                   sizeof(unit_conv_four_output)) != 0) {
            fprintf(stderr, "AVX2 four-output unit-stride Conv differs from scalar output\n");
            return 1;
        }
    }

    fill_values(unit_conv2x2_weights, 24u, 23u, 41u, 20, 10.0f);
    lw_scalar_conv2x2_unit_pad_end1_f32(unit_conv_input, unit_conv2x2_weights, unit_conv_bias,
                                        unit_conv2x2_output, unit_conv_input_dimensions,
                                        unit_conv_output_dimensions);
    if (simd_level >= LW_SIMD_LEVEL_SSE2) {
        lw_sse2_conv2x2_unit_pad_end1_f32(unit_conv_input, unit_conv2x2_weights, unit_conv_bias,
                                          unit_conv2x2_simd_output, unit_conv_input_dimensions,
                                          unit_conv_output_dimensions);
        if (memcmp(unit_conv2x2_output, unit_conv2x2_simd_output, sizeof(unit_conv2x2_output)) !=
            0) {
            fprintf(stderr, "SSE2 2x2 Conv differs from scalar output\n");
            return 1;
        }
    }
    if (simd_level >= LW_SIMD_LEVEL_AVX2) {
        lw_avx2_conv2x2_unit_pad_end1_f32(unit_conv_input, unit_conv2x2_weights, unit_conv_bias,
                                          unit_conv2x2_simd_output, unit_conv_input_dimensions,
                                          unit_conv_output_dimensions);
        if (memcmp(unit_conv2x2_output, unit_conv2x2_simd_output, sizeof(unit_conv2x2_output)) !=
            0) {
            fprintf(stderr, "AVX2 2x2 Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(
        unit_conv_input, unit_conv2x2_weights, unit_conv_bias, 2u, unit_conv2x2_dispatched_output,
        unit_conv_input_dimensions, unit_conv2x2_weight_dimensions, unit_conv_output_dimensions,
        unit_conv2x2_kernel, unit_strides, unit_dilations, unit_conv2x2_pads, 1u);
    if (!expect_status("2x2 unit-stride conv", status, LW_STATUS_OK) ||
        memcmp(unit_conv2x2_output, unit_conv2x2_dispatched_output, sizeof(unit_conv2x2_output)) !=
            0) {
        fprintf(stderr, "dispatched 2x2 Conv differs from scalar output\n");
        return 1;
    }
    print_values("unit_stride_conv2x2", unit_conv2x2_output, 190u);

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

    fill_values(unit_depthwise5x5_input, 228u, 29u, 53u, 26, 13.0f);
    fill_values(unit_depthwise5x5_weights, 50u, 31u, 47u, 23, 12.0f);
    lw_scalar_depthwise_conv5x5_unit_pad2_f32(unit_depthwise5x5_input, unit_depthwise5x5_weights,
                                              unit_depthwise_bias, unit_depthwise5x5_output,
                                              unit_depthwise5x5_dimensions);
    if (simd_level >= LW_SIMD_LEVEL_SSE2) {
        lw_sse2_depthwise_conv5x5_unit_pad2_f32(unit_depthwise5x5_input, unit_depthwise5x5_weights,
                                                unit_depthwise_bias, unit_depthwise5x5_simd_output,
                                                unit_depthwise5x5_dimensions);
        if (memcmp(unit_depthwise5x5_output, unit_depthwise5x5_simd_output,
                   sizeof(unit_depthwise5x5_output)) != 0) {
            fprintf(stderr, "SSE2 depthwise 5x5 Conv differs from scalar output\n");
            return 1;
        }
    }
    if (simd_level >= LW_SIMD_LEVEL_AVX2) {
        lw_avx2_depthwise_conv5x5_unit_pad2_f32(unit_depthwise5x5_input, unit_depthwise5x5_weights,
                                                unit_depthwise_bias, unit_depthwise5x5_simd_output,
                                                unit_depthwise5x5_dimensions);
        if (memcmp(unit_depthwise5x5_output, unit_depthwise5x5_simd_output,
                   sizeof(unit_depthwise5x5_output)) != 0) {
            fprintf(stderr, "AVX2 depthwise 5x5 Conv differs from scalar output\n");
            return 1;
        }
    }
    status = lw_scalar_conv2d_f32(unit_depthwise5x5_input, unit_depthwise5x5_weights,
                                  unit_depthwise_bias, 2u, unit_depthwise5x5_dispatched_output,
                                  unit_depthwise5x5_dimensions, unit_depthwise5x5_weight_dimensions,
                                  unit_depthwise5x5_dimensions, depthwise5x5_kernel, unit_strides,
                                  unit_dilations, pad2, 2u);
    if (!expect_status("unit depthwise 5x5 conv", status, LW_STATUS_OK) ||
        memcmp(unit_depthwise5x5_output, unit_depthwise5x5_dispatched_output,
               sizeof(unit_depthwise5x5_output)) != 0) {
        fprintf(stderr, "dispatched depthwise 5x5 Conv differs from scalar output\n");
        return 1;
    }
    print_values("unit_depthwise_conv5x5", unit_depthwise5x5_output, 228u);

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

    /* Exercise four-output-channel packing, including the x64 16-value spatial
     * main loop, its five-value tail, and a three-output-channel block. */
    fill_values(packed_pointwise_input, 210u, 13u, 43u, 21, 11.0f);
    fill_values(packed_pointwise_weights, 35u, 17u, 37u, 18, 9.0f);
    lw_scalar_conv1x1_unit_f32(packed_pointwise_input, packed_pointwise_weights,
                               packed_pointwise_bias, packed_pointwise_output,
                               packed_pointwise_input_dimensions,
                               packed_pointwise_output_dimensions, 1u, 5u, 7u);
    lw_pack_conv1x1_weights_f32(packed_pointwise_weights, 5u, 7u, packed_pointwise_packed_weights);
    lw_scalar_packed_conv1x1_f32(packed_pointwise_input, packed_pointwise_packed_weights,
                                 packed_pointwise_bias, packed_pointwise_simd_output,
                                 packed_pointwise_input_dimensions,
                                 packed_pointwise_output_dimensions);
    if (memcmp(packed_pointwise_output, packed_pointwise_simd_output,
               sizeof(packed_pointwise_output)) != 0) {
        fprintf(stderr, "scalar packed pointwise Conv differs from canonical output\n");
        return 1;
    }
    if (simd_level >= LW_SIMD_LEVEL_SSE2) {
        lw_sse2_packed_conv1x1_f32(packed_pointwise_input, packed_pointwise_packed_weights,
                                   packed_pointwise_bias, packed_pointwise_simd_output,
                                   packed_pointwise_input_dimensions,
                                   packed_pointwise_output_dimensions);
        if (memcmp(packed_pointwise_output, packed_pointwise_simd_output,
                   sizeof(packed_pointwise_output)) != 0) {
            fprintf(stderr, "SSE2 packed pointwise Conv differs from canonical output\n");
            return 1;
        }
    }
    if (simd_level >= LW_SIMD_LEVEL_AVX2) {
        lw_avx2_packed_conv1x1_f32(packed_pointwise_input, packed_pointwise_packed_weights,
                                   packed_pointwise_bias, packed_pointwise_simd_output,
                                   packed_pointwise_input_dimensions,
                                   packed_pointwise_output_dimensions);
        if (memcmp(packed_pointwise_output, packed_pointwise_simd_output,
                   sizeof(packed_pointwise_output)) != 0) {
            fprintf(stderr, "AVX2 packed pointwise Conv differs from canonical output\n");
            return 1;
        }
    }
    lw_packed_conv1x1_f32(packed_pointwise_input, packed_pointwise_packed_weights,
                          packed_pointwise_bias, packed_pointwise_simd_output,
                          packed_pointwise_input_dimensions, packed_pointwise_output_dimensions);
    if (memcmp(packed_pointwise_output, packed_pointwise_simd_output,
               sizeof(packed_pointwise_output)) != 0) {
        fprintf(stderr, "dispatched packed pointwise Conv differs from canonical output\n");
        return 1;
    }
    print_values("packed_pointwise_conv", packed_pointwise_output, 294u);

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

    /* Width eight exercises the vector body of both dedicated transpose-Conv
     * kernels; compare each implementation with the dispatched result. */
    fill_values(transpose_conv_simd_input, 48u, 17u, 37u, 18, 9.0f);
    fill_values(transpose_conv_simd_weights, 24u, 11u, 29u, 14, 7.0f);
    status = lw_scalar_conv_transpose2d_f32(
        transpose_conv_simd_input, transpose_conv_simd_weights, transpose_conv_simd_bias, 3u,
        transpose_conv_simd_output, transpose_conv_simd_input_dimensions,
        transpose_conv_simd_weight_dimensions, transpose_conv_simd_output_dimensions,
        transpose_conv_kernel, transpose_conv_strides, unit_dilations, no_pads, 1u);
    if (!expect_status("transpose conv SIMD shape", status, LW_STATUS_OK)) {
        return 1;
    }
    if (simd_level >= LW_SIMD_LEVEL_SSE2) {
        lw_sse2_conv_transpose2x2_stride2_f32(
            transpose_conv_simd_input, transpose_conv_simd_weights, transpose_conv_simd_bias,
            transpose_conv_sse2_output, transpose_conv_simd_input_dimensions,
            transpose_conv_simd_output_dimensions);
        if (memcmp(transpose_conv_simd_output, transpose_conv_sse2_output,
                   sizeof(transpose_conv_simd_output)) != 0) {
            fprintf(stderr, "SSE2 transpose Conv differs from dispatched output\n");
            return 1;
        }
    }
    if (simd_level >= LW_SIMD_LEVEL_AVX2) {
        lw_avx2_conv_transpose2x2_stride2_f32(
            transpose_conv_simd_input, transpose_conv_simd_weights, transpose_conv_simd_bias,
            transpose_conv_avx2_output, transpose_conv_simd_input_dimensions,
            transpose_conv_simd_output_dimensions);
        if (memcmp(transpose_conv_simd_output, transpose_conv_avx2_output,
                   sizeof(transpose_conv_simd_output)) != 0) {
            fprintf(stderr, "AVX2 transpose Conv differs from dispatched output\n");
            return 1;
        }
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
