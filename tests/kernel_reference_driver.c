#include "scalar_kernels.h"

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
        fprintf(stderr, "%s: expected %s, got %s\n",
                name, lw_status_string(expected), lw_status_string(actual));
        return 0;
    }
    return 1;
}

int main(void) {
    const int32_t left_dimensions[3] = {2, 3, 4};
    const int32_t right_dimensions[2] = {3, 1};
    const int32_t output_dimensions[3] = {2, 3, 4};
    const int32_t invalid_output_dimensions[3] = {2, 3, 5};
    const int32_t softmax_dimensions[3] = {2, 3, 4};
    const float add_right[3] = {0.25f, -1.0f, 2.0f};
    const float mul_right[3] = {-2.0f, 0.5f, 3.0f};
    const float div_right[3] = {0.5f, -2.0f, 4.0f};
    const float activation_input[9] = {
        -4.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 4.0f};
    const float softmax_input[24] = {
        1000.0f, -1000.0f, 0.0f, 50.0f,
        1001.0f, -999.0f, 2.0f, 48.0f,
        999.0f, -1002.0f, -1.0f, 51.0f,
        -500.0f, 500.0f, 10.0f, -10.0f,
        -501.0f, 501.0f, 12.0f, -9.0f,
        -499.0f, 499.0f, 8.0f, -11.0f};
    float left[24];
    float binary_output[24];
    float activation_output[9];
    float softmax_output[24];
    float softmax_in_place[24];
    uint32_t index;
    lw_status status;

    for (index = 0u; index < 24u; ++index) {
        left[index] = (float)((int32_t)((index * 7u) % 19u) - 9) / 5.0f;
    }

    status = lw_scalar_binary_f32(
        LW_SCALAR_BINARY_ADD, left, 3u, left_dimensions,
        add_right, 2u, right_dimensions, binary_output, 3u, output_dimensions);
    if (!expect_status("add", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("add", binary_output, 24u);

    status = lw_scalar_binary_f32(
        LW_SCALAR_BINARY_MUL, left, 3u, left_dimensions,
        mul_right, 2u, right_dimensions, binary_output, 3u, output_dimensions);
    if (!expect_status("mul", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("mul", binary_output, 24u);

    status = lw_scalar_binary_f32(
        LW_SCALAR_BINARY_DIV, left, 3u, left_dimensions,
        div_right, 2u, right_dimensions, binary_output, 3u, output_dimensions);
    if (!expect_status("div", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("div", binary_output, 24u);

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

    status = lw_scalar_hard_sigmoid_f32(
        activation_input, activation_output, 9u, 0.2f, 0.5f);
    if (!expect_status("hard_sigmoid", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("hard_sigmoid", activation_output, 9u);

    status = lw_scalar_softmax_f32(
        softmax_input, softmax_output, 3u, softmax_dimensions, 1);
    if (!expect_status("softmax", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("softmax", softmax_output, 24u);

    memcpy(softmax_in_place, softmax_input, sizeof(softmax_input));
    status = lw_scalar_softmax_f32(
        softmax_in_place, softmax_in_place, 3u, softmax_dimensions, -2);
    if (!expect_status("softmax_in_place", status, LW_STATUS_OK)) {
        return 1;
    }
    print_values("softmax_in_place", softmax_in_place, 24u);

    status = lw_scalar_binary_f32(
        LW_SCALAR_BINARY_ADD, left, 3u, left_dimensions,
        add_right, 2u, right_dimensions, binary_output, 3u, invalid_output_dimensions);
    if (!expect_status("invalid broadcast output", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_binary_f32(
        LW_SCALAR_BINARY_ADD, left, 3u, left_dimensions,
        add_right, 2u, right_dimensions, left, 3u, output_dimensions);
    if (!expect_status("binary alias", status, LW_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    status = lw_scalar_softmax_f32(
        softmax_input, softmax_output, 3u, softmax_dimensions, 3);
    if (!expect_status("invalid softmax axis", status, LW_STATUS_INVALID_SHAPE)) {
        return 1;
    }
    status = lw_scalar_relu_f32(NULL, activation_output, 1u);
    if (!expect_status("null activation input", status, LW_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    status = lw_scalar_hard_sigmoid_f32(
        activation_input, activation_output, 9u, NAN, 0.5f);
    if (!expect_status("non-finite hard sigmoid parameter", status, LW_STATUS_INVALID_ARGUMENT)) {
        return 1;
    }
    return 0;
}
