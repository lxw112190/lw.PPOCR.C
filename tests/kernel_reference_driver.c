#include "scalar_kernels.h"
#include "cpu_features.h"
#include "simd_kernels.h"

#include <inttypes.h>
#include <math.h>
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

int main(void) {
    enum { ERF_DENSE_COUNT = 4099 };
    const int32_t left_dimensions[3] = {2, 3, 4};
    const int32_t right_dimensions[2] = {3, 1};
    const int32_t output_dimensions[3] = {2, 3, 4};
    const int32_t invalid_output_dimensions[3] = {2, 3, 5};
    const int32_t softmax_dimensions[3] = {2, 3, 4};
    const int32_t softmax_last_dimensions[2] = {2, 12};
    const int32_t flat_dimensions[1] = {10};
    const int32_t trailing_left_dimensions[2] = {2, 10};
    const int32_t trailing_right_dimensions[1] = {10};
    const int32_t general_right_dimensions[3] = {2, 1, 4};
    const float add_right[3] = {0.25f, -1.0f, 2.0f};
    const float mul_right[3] = {-2.0f, 0.5f, 3.0f};
    const float div_right[3] = {0.5f, -2.0f, 4.0f};
    const float activation_input[9] = {-4.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f};
    const float softmax_input[24] = {1000.0f, -1000.0f, 0.0f,    50.0f,    1001.0f, -999.0f,
                                     2.0f,    48.0f,    999.0f,  -1002.0f, -1.0f,   51.0f,
                                     -500.0f, 500.0f,   10.0f,   -10.0f,   -501.0f, 501.0f,
                                     12.0f,   -9.0f,    -499.0f, 499.0f,   8.0f,    -11.0f};
    const float flat_left[10] = {-4.0f, -2.5f, -1.0f, -0.25f, 0.0f, 0.5f, 1.25f, 2.0f, 3.5f, 5.0f};
    const float flat_right[10] = {0.5f, -2.0f,  4.0f, 0.25f, -0.75f,
                                  2.5f, -1.25f, 8.0f, 1.75f, -4.0f};
    const float flat_scalar[1] = {1.25f};
    const float general_right[8] = {0.5f, -1.0f, 1.5f, -2.0f, 2.5f, -3.0f, 3.5f, -4.0f};
    const char* flat_names[3] = {"flat_add", "flat_mul", "flat_div"};
    const char* scalar_names[3] = {"right_scalar_add", "right_scalar_mul", "right_scalar_div"};
    float left[24];
    float binary_output[24];
    float flat_output[10];
    float flat_dispatched_output[10];
    float flat_simd_output[10];
    float trailing_left[20];
    float trailing_output[20];
    float activation_output[9];
    float erf_dense_input[ERF_DENSE_COUNT];
    float erf_dense_output[ERF_DENSE_COUNT];
    float gelu_reference[ERF_DENSE_COUNT];
    float gelu_output[ERF_DENSE_COUNT];
    float gelu_temporary[ERF_DENSE_COUNT];
    float softmax_output[24];
    float softmax_in_place[24];
    lw_simd_level simd_level;
    uint32_t index;
    lw_status status;

    for (index = 0u; index < 24u; ++index) {
        left[index] = (float)((int32_t)((index * 7u) % 19u) - 9) / 5.0f;
    }
    for (index = 0u; index < 20u; ++index) {
        trailing_left[index] = (float)((int32_t)((index * 11u) % 23u) - 11) / 4.0f;
    }

    status = lw_scalar_binary_f32(LW_SCALAR_BINARY_ADD, left, 3u, left_dimensions, add_right, 2u,
                                  right_dimensions, binary_output, 3u, output_dimensions);
    if (!expect_status("add", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("add", binary_output, 24u);

    status = lw_scalar_binary_f32(LW_SCALAR_BINARY_MUL, left, 3u, left_dimensions, mul_right, 2u,
                                  right_dimensions, binary_output, 3u, output_dimensions);
    if (!expect_status("mul", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("mul", binary_output, 24u);

    status = lw_scalar_binary_f32(LW_SCALAR_BINARY_DIV, left, 3u, left_dimensions, div_right, 2u,
                                  right_dimensions, binary_output, 3u, output_dimensions);
    if (!expect_status("div", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("div", binary_output, 24u);

    status = lw_scalar_binary_f32(LW_SCALAR_BINARY_ADD, trailing_left, 2u, trailing_left_dimensions,
                                  flat_right, 1u, trailing_right_dimensions, trailing_output, 2u,
                                  trailing_left_dimensions);
    if (!expect_status("trailing add", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("trailing_add", trailing_output, 20u);

    status =
        lw_scalar_binary_f32(LW_SCALAR_BINARY_ADD, left, 3u, left_dimensions, general_right, 3u,
                             general_right_dimensions, binary_output, 3u, output_dimensions);
    if (!expect_status("general add", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("general_add", binary_output, 24u);

    simd_level = lw_detect_simd_level();
    for (index = 0u; index < 3u; ++index) {
        lw_scalar_binary_op operation = (lw_scalar_binary_op)(index + 1u);
        lw_scalar_binary_contiguous_f32(operation, flat_left, flat_right, flat_output, 10u);
        if (simd_level >= LW_SIMD_LEVEL_SSE2) {
            lw_sse2_binary_contiguous_f32(operation, flat_left, flat_right, flat_simd_output, 10u);
            if (memcmp(flat_output, flat_simd_output, sizeof(flat_output)) != 0) {
                fprintf(stderr, "SSE2 flat binary differs from scalar output\n");
                return 1;
            }
        }
        if (simd_level >= LW_SIMD_LEVEL_AVX2) {
            lw_avx2_binary_contiguous_f32(operation, flat_left, flat_right, flat_simd_output, 10u);
            if (memcmp(flat_output, flat_simd_output, sizeof(flat_output)) != 0) {
                fprintf(stderr, "AVX2 flat binary differs from scalar output\n");
                return 1;
            }
        }
        status = lw_scalar_binary_f32(operation, flat_left, 1u, flat_dimensions, flat_right, 1u,
                                      flat_dimensions, flat_dispatched_output, 1u, flat_dimensions);
        if (!expect_status("flat binary", status, LW_STATUS_OK) ||
            memcmp(flat_output, flat_dispatched_output, sizeof(flat_output)) != 0) {
            fprintf(stderr, "dispatched flat binary differs from scalar output\n");
            return 1;
        }
        print_values(flat_names[index], flat_output, 10u);

        lw_scalar_binary_right_scalar_f32(operation, flat_left, flat_scalar[0], flat_output, 10u);
        if (simd_level >= LW_SIMD_LEVEL_SSE2) {
            lw_sse2_binary_right_scalar_f32(operation, flat_left, flat_scalar[0], flat_simd_output,
                                            10u);
            if (memcmp(flat_output, flat_simd_output, sizeof(flat_output)) != 0) {
                fprintf(stderr, "SSE2 scalar binary differs from scalar output\n");
                return 1;
            }
        }
        if (simd_level >= LW_SIMD_LEVEL_AVX2) {
            lw_avx2_binary_right_scalar_f32(operation, flat_left, flat_scalar[0], flat_simd_output,
                                            10u);
            if (memcmp(flat_output, flat_simd_output, sizeof(flat_output)) != 0) {
                fprintf(stderr, "AVX2 scalar binary differs from scalar output\n");
                return 1;
            }
        }
        status = lw_scalar_binary_f32(operation, flat_left, 1u, flat_dimensions, flat_scalar, 0u,
                                      NULL, flat_dispatched_output, 1u, flat_dimensions);
        if (!expect_status("right-scalar binary", status, LW_STATUS_OK) ||
            memcmp(flat_output, flat_dispatched_output, sizeof(flat_output)) != 0) {
            fprintf(stderr, "dispatched scalar binary differs from scalar output\n");
            return 1;
        }
        print_values(scalar_names[index], flat_output, 10u);
    }

    status = lw_scalar_relu_f32(activation_input, activation_output, 9u);
    if (!expect_status("relu", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("relu", activation_output, 9u);

    status = lw_scalar_erf_f32(activation_input, activation_output, 9u);
    if (!expect_status("erf", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("erf", activation_output, 9u);

    if (simd_level >= LW_SIMD_LEVEL_AVX2) {
        const float special_input[8] = {0.0f, -0.0f, INFINITY, -INFINITY, NAN, 0.5f, -0.5f, 4.0f};
        float special_output[8];
        float maximum_error = 0.0f;
        for (index = 0u; index < ERF_DENSE_COUNT; ++index) {
            erf_dense_input[index] = -6.0f + 12.0f * (float)index / (float)(ERF_DENSE_COUNT - 1u);
        }
        lw_avx2_erf_f32(erf_dense_input, erf_dense_output, ERF_DENSE_COUNT);
        for (index = 0u; index < ERF_DENSE_COUNT; ++index) {
            float error = fabsf(erf_dense_output[index] - erff(erf_dense_input[index]));
            if (error > maximum_error) {
                maximum_error = error;
            }
            if (index != 0u && erf_dense_output[index] < erf_dense_output[index - 1u]) {
                fprintf(stderr, "AVX2 Erf approximation is not monotonic\n");
                return 1;
            }
        }
        if (maximum_error > 5.0e-7f) {
            fprintf(stderr, "AVX2 Erf maximum absolute error %.9g exceeds limit\n",
                    (double)maximum_error);
            return 1;
        }
        lw_avx2_binary_right_scalar_f32(LW_SCALAR_BINARY_DIV, erf_dense_input, 1.4142135381698608f,
                                        gelu_temporary, ERF_DENSE_COUNT);
        lw_avx2_erf_f32(gelu_temporary, gelu_reference, ERF_DENSE_COUNT);
        lw_avx2_binary_right_scalar_f32(LW_SCALAR_BINARY_ADD, gelu_reference, 1.0f, gelu_temporary,
                                        ERF_DENSE_COUNT);
        lw_avx2_binary_contiguous_f32(LW_SCALAR_BINARY_MUL, erf_dense_input, gelu_temporary,
                                      gelu_reference, ERF_DENSE_COUNT);
        lw_avx2_binary_right_scalar_f32(LW_SCALAR_BINARY_MUL, gelu_reference, 0.5f, gelu_temporary,
                                        ERF_DENSE_COUNT);
        lw_avx2_gelu_f32(erf_dense_input, gelu_output, ERF_DENSE_COUNT);
        if (memcmp(gelu_temporary, gelu_output, sizeof(gelu_output)) != 0) {
            fprintf(stderr, "AVX2 fused GELU differs from its five-node graph\n");
            return 1;
        }
        memcpy(gelu_reference, erf_dense_input, sizeof(gelu_reference));
        lw_avx2_gelu_f32(gelu_reference, gelu_reference, ERF_DENSE_COUNT);
        if (memcmp(gelu_reference, gelu_output, sizeof(gelu_output)) != 0) {
            fprintf(stderr, "AVX2 fused GELU in-place output differs\n");
            return 1;
        }
        lw_avx2_erf_f32(special_input, special_output, 8u);
        if (special_output[0] != 0.0f || signbit(special_output[0]) || special_output[1] != 0.0f ||
            !signbit(special_output[1]) || special_output[2] != 1.0f ||
            special_output[3] != -1.0f || !isnan(special_output[4])) {
            fprintf(stderr, "AVX2 Erf special-value contract failed\n");
            return 1;
        }
    }

    status = lw_scalar_hard_sigmoid_f32(activation_input, activation_output, 9u, 0.2f, 0.5f);
    if (!expect_status("hard_sigmoid", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("hard_sigmoid", activation_output, 9u);

    status = lw_scalar_sigmoid_f32(activation_input, activation_output, 9u);
    if (!expect_status("sigmoid", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("sigmoid", activation_output, 9u);

    status = lw_scalar_softmax_f32(softmax_input, softmax_output, 3u, softmax_dimensions, 1);
    if (!expect_status("softmax", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("softmax", softmax_output, 24u);

    memcpy(softmax_in_place, softmax_input, sizeof(softmax_input));
    status = lw_scalar_softmax_f32(softmax_in_place, softmax_in_place, 3u, softmax_dimensions, -2);
    if (!expect_status("softmax_in_place", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("softmax_in_place", softmax_in_place, 24u);

    status = lw_scalar_softmax_f32(softmax_input, softmax_output, 2u, softmax_last_dimensions, -1);
    if (!expect_status("contiguous-axis softmax", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("softmax_contiguous_axis", softmax_output, 24u);

    memcpy(softmax_in_place, softmax_input, sizeof(softmax_input));
    status =
        lw_scalar_softmax_f32(softmax_in_place, softmax_in_place, 2u, softmax_last_dimensions, 1);
    if (!expect_status("in-place contiguous-axis softmax", status, LW_STATUS_OK) ||
        memcmp(softmax_output, softmax_in_place, sizeof(softmax_output)) != 0) {
        fprintf(stderr, "in-place contiguous-axis softmax differs from separate output\n");
        return 1;
    }
    print_values("softmax_contiguous_axis_in_place", softmax_in_place, 24u);

    status = lw_scalar_binary_f32(LW_SCALAR_BINARY_ADD, left, 3u, left_dimensions, add_right, 2u,
                                  right_dimensions, binary_output, 3u, invalid_output_dimensions);
    if (!expect_status("invalid broadcast output", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_binary_f32(LW_SCALAR_BINARY_ADD, left, 3u, left_dimensions, add_right, 2u,
                                  right_dimensions, left, 3u, output_dimensions);
    if (!expect_status("binary alias", status, LW_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    status = lw_scalar_softmax_f32(softmax_input, softmax_output, 3u, softmax_dimensions, 3);
    if (!expect_status("invalid softmax axis", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_relu_f32(NULL, activation_output, 1u);
    if (!expect_status("null activation input", status, LW_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    status = lw_scalar_hard_sigmoid_f32(activation_input, activation_output, 9u, NAN, 0.5f);
    if (!expect_status("non-finite hard sigmoid parameter", status, LW_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    return 0;
}
