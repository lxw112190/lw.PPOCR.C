#include "simd_kernels.h"

#include "scalar_kernels.h"

#include <stddef.h>

#if defined(_M_IX86) || defined(_M_X64) || \
    defined(__i386__) || defined(__x86_64__)
#  include <emmintrin.h>
#  define LW_COMPILES_SSE2_CONV1X1 1
#else
#  define LW_COMPILES_SSE2_CONV1X1 0
#endif

#if LW_COMPILES_SSE2_CONV1X1 && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("sse2")))
#endif
void lw_sse2_conv1x1_unit_f32(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const int32_t input_dimensions[4],
    const int32_t output_dimensions[4],
    uint32_t groups,
    uint32_t input_channels_per_group,
    uint32_t output_channels_per_group) {
#if LW_COMPILES_SSE2_CONV1X1
    uint64_t channel_plane =
        (uint64_t)(uint32_t)input_dimensions[2] *
        (uint32_t)input_dimensions[3];
    uint32_t batch;
    for (batch = 0u; batch < (uint32_t)input_dimensions[0]; ++batch) {
        uint32_t group;
        for (group = 0u; group < groups; ++group) {
            uint32_t input_channel_start = group * input_channels_per_group;
            uint32_t output_channel_start = group * output_channels_per_group;
            const float* group_input = input + (size_t)(
                ((uint64_t)batch * (uint32_t)input_dimensions[1] +
                 input_channel_start) * channel_plane);
            float* group_output = output + (size_t)(
                ((uint64_t)batch * (uint32_t)output_dimensions[1] +
                 output_channel_start) * channel_plane);
            uint32_t output_channel_in_group;
            for (output_channel_in_group = 0u;
                 output_channel_in_group < output_channels_per_group;
                 ++output_channel_in_group) {
                uint32_t output_channel =
                    output_channel_start + output_channel_in_group;
                const float* output_channel_weights = weights + (size_t)(
                    (uint64_t)output_channel * input_channels_per_group);
                float* output_channel_data = group_output + (size_t)(
                    (uint64_t)output_channel_in_group * channel_plane);
                float initial = bias == NULL ? 0.0f : bias[output_channel];
                __m128 initial_values = _mm_set1_ps(initial);
                uint64_t spatial = 0u;
                uint32_t input_channel_in_group;
                for (; spatial + 4u <= channel_plane; spatial += 4u) {
                    _mm_storeu_ps(output_channel_data + (size_t)spatial,
                                  initial_values);
                }
                for (; spatial < channel_plane; ++spatial) {
                    output_channel_data[(size_t)spatial] = initial;
                }
                for (input_channel_in_group = 0u;
                     input_channel_in_group < input_channels_per_group;
                     ++input_channel_in_group) {
                    const float* input_channel_data = group_input + (size_t)(
                        (uint64_t)input_channel_in_group * channel_plane);
                    __m128 weight_values = _mm_set1_ps(
                        output_channel_weights[input_channel_in_group]);
                    spatial = 0u;
                    for (; spatial + 4u <= channel_plane; spatial += 4u) {
                        __m128 output_values = _mm_loadu_ps(
                            output_channel_data + (size_t)spatial);
                        __m128 input_values = _mm_loadu_ps(
                            input_channel_data + (size_t)spatial);
                        output_values = _mm_add_ps(
                            output_values,
                            _mm_mul_ps(input_values, weight_values));
                        _mm_storeu_ps(
                            output_channel_data + (size_t)spatial, output_values);
                    }
                    for (; spatial < channel_plane; ++spatial) {
                        output_channel_data[(size_t)spatial] +=
                            input_channel_data[(size_t)spatial] *
                            output_channel_weights[input_channel_in_group];
                    }
                }
            }
        }
    }
#else
    lw_scalar_conv1x1_unit_f32(
        input, weights, bias, output, input_dimensions, output_dimensions,
        groups, input_channels_per_group, output_channels_per_group);
#endif
}
