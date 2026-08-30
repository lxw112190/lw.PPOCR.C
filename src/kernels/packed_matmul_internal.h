#ifndef LW_PACKED_MATMUL_INTERNAL_H
#define LW_PACKED_MATMUL_INTERNAL_H

/*
 * Private packed-weight contract for wide shared-weight MatMul. LWM keeps B
 * in canonical [K][N] order; eligible x64 AVX2 sessions convert it once to
 * [ceil(N/16)][K][16] and reuse the packed copy for every graph execution.
 */

#include <stdint.h>

#define LW_PACKED_MATMUL_COLUMN_TILE 16u

int lw_packed_matmul_weight_count(uint32_t inner_dimension, uint32_t columns,
                                  uint64_t* weight_count);
void lw_pack_matmul_weights_f32(const float* weights, uint32_t inner_dimension,
                                uint32_t columns, float* packed_weights);
void lw_scalar_packed_matmul_shared_f32(const float* input, const float* packed_weights,
                                        float* output, uint32_t batch_count, uint32_t rows,
                                        uint32_t inner_dimension, uint32_t columns);

#endif
