#ifndef LW_PACKED_CONV_INTERNAL_H
#define LW_PACKED_CONV_INTERNAL_H

/*
 * Private packed-weight contract for NCHW 1x1 Conv. LWM keeps canonical OIHW
 * weights; a session converts eligible constant weights to [OC/4][IC][4]
 * once and reuses them for every execution.
 */

#include "lw_infer.h"

#include <stdint.h>

#define LW_PACKED_CONV1X1_OUTPUT_TILE 4u

int lw_packed_conv1x1_weight_count(uint32_t input_channels, uint32_t output_channels,
                                   uint64_t* weight_count);
void lw_pack_conv1x1_weights_f32(const float* weights, uint32_t input_channels,
                                 uint32_t output_channels, float* packed_weights);

void lw_scalar_packed_conv1x1_f32(const float* input, const float* packed_weights,
                                  const float* bias, float* output,
                                  const int32_t input_dimensions[4],
                                  const int32_t output_dimensions[4]);
void lw_packed_conv1x1_f32(const float* input, const float* packed_weights, const float* bias,
                           float* output, const int32_t input_dimensions[4],
                           const int32_t output_dimensions[4]);

#endif
