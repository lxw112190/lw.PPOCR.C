#ifndef LW_PACKED_CONV3X3_INTERNAL_H
#define LW_PACKED_CONV3X3_INTERNAL_H

/*
 * Private packed-weight contract for NCHW group-1 3x3 Conv with stride 2 and
 * pad 1. LWM keeps canonical OIHW weights; eligible sessions may convert them
 * once to [OC/8][IC][3][3][8].
 */

#include <stdint.h>

#define LW_PACKED_CONV3X3_STRIDE2_OUTPUT_TILE 8u

typedef void (*lw_packed_conv3x3_kernel_fn)(
    const float* input, const float* packed_weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]);

int lw_packed_conv3x3_stride2_weight_count(uint32_t input_channels,
                                           uint32_t output_channels,
                                           uint64_t* weight_count);
void lw_pack_conv3x3_stride2_weights_f32(const float* weights, uint32_t input_channels,
                                         uint32_t output_channels, float* packed_weights);
void lw_scalar_packed_conv3x3_stride2_pad1_f32(
    const float* input, const float* packed_weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]);
void lw_packed_conv3x3_stride2_pad1_f32(
    const float* input, const float* packed_weights, const float* bias, float* output,
    const int32_t input_dimensions[4], const int32_t output_dimensions[4]);

#endif
