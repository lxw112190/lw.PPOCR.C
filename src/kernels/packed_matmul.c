#include "packed_matmul_internal.h"

#include <stddef.h>
#include <stdint.h>

int lw_packed_matmul_weight_count(uint32_t inner_dimension, uint32_t columns,
                                  uint64_t* weight_count) {
    uint64_t column_panels;
    if (inner_dimension == 0u || columns == 0u || weight_count == NULL) {
        return 0;
    }
    column_panels = ((uint64_t)columns + LW_PACKED_MATMUL_COLUMN_TILE - 1u) /
                    LW_PACKED_MATMUL_COLUMN_TILE;
    if (column_panels > UINT64_MAX / inner_dimension ||
        column_panels * inner_dimension > UINT64_MAX / LW_PACKED_MATMUL_COLUMN_TILE) {
        return 0;
    }
    *weight_count = column_panels * inner_dimension * LW_PACKED_MATMUL_COLUMN_TILE;
    return *weight_count <= (uint64_t)(SIZE_MAX / sizeof(float));
}

void lw_pack_matmul_weights_f32(const float* weights, uint32_t inner_dimension,
                                uint32_t columns, float* packed_weights) {
    uint32_t column_panel;
    const uint32_t column_panels =
        (columns + LW_PACKED_MATMUL_COLUMN_TILE - 1u) / LW_PACKED_MATMUL_COLUMN_TILE;
    for (column_panel = 0u; column_panel < column_panels; ++column_panel) {
        uint32_t inner;
        for (inner = 0u; inner < inner_dimension; ++inner) {
            uint32_t lane;
            for (lane = 0u; lane < LW_PACKED_MATMUL_COLUMN_TILE; ++lane) {
                const uint32_t column =
                    column_panel * LW_PACKED_MATMUL_COLUMN_TILE + lane;
                const uint64_t packed_index =
                    ((uint64_t)column_panel * inner_dimension + inner) *
                        LW_PACKED_MATMUL_COLUMN_TILE +
                    lane;
                packed_weights[(size_t)packed_index] =
                    column < columns
                        ? weights[(size_t)((uint64_t)inner * columns + column)]
                        : 0.0f;
            }
        }
    }
}

void lw_scalar_packed_matmul_shared_f32(const float* input, const float* packed_weights,
                                        float* output, uint32_t batch_count, uint32_t rows,
                                        uint32_t inner_dimension, uint32_t columns) {
    uint32_t batch;
    const uint32_t column_panels =
        (columns + LW_PACKED_MATMUL_COLUMN_TILE - 1u) / LW_PACKED_MATMUL_COLUMN_TILE;
    for (batch = 0u; batch < batch_count; ++batch) {
        uint32_t row;
        for (row = 0u; row < rows; ++row) {
            uint32_t column_panel;
            for (column_panel = 0u; column_panel < column_panels; ++column_panel) {
                float accumulators[LW_PACKED_MATMUL_COLUMN_TILE] = {0.0f};
                const uint32_t column_base = column_panel * LW_PACKED_MATMUL_COLUMN_TILE;
                const uint32_t valid_columns =
                    columns - column_base < LW_PACKED_MATMUL_COLUMN_TILE
                        ? columns - column_base
                        : LW_PACKED_MATMUL_COLUMN_TILE;
                uint32_t inner;
                for (inner = 0u; inner < inner_dimension; ++inner) {
                    const float input_value =
                        input[(size_t)(((uint64_t)batch * rows + row) * inner_dimension + inner)];
                    const float* packed =
                        packed_weights +
                        (size_t)(((uint64_t)column_panel * inner_dimension + inner) *
                                 LW_PACKED_MATMUL_COLUMN_TILE);
                    uint32_t lane;
                    for (lane = 0u; lane < valid_columns; ++lane) {
                        accumulators[lane] += input_value * packed[lane];
                    }
                }
                for (uint32_t lane = 0u; lane < valid_columns; ++lane) {
                    output[(size_t)(((uint64_t)batch * rows + row) * columns + column_base +
                                    lane)] = accumulators[lane];
                }
            }
        }
    }
}
