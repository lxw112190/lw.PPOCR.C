#ifndef LW_X64_DET_BACKEND_INTERNAL_H
#define LW_X64_DET_BACKEND_INTERNAL_H

/*
 * Standalone x64 DET graph backend (experimental, AVX2+FMA only). Cloned from
 * the proven x64_rec_backend structure: two-pass lowering into a physical op
 * table, a greedy-by-size arena interval planner, and a per-instance arena.
 * The one new design piece over the REC clone is the effective-layout walk:
 * the planner's NHWC classification is crossed with a v1 lowerability set
 * (Resize/ConvTranspose/Concat/Sigmoid stay NCHW for now), and tensors that
 * cross an effective-layout boundary get a lazily materialized LAYOUT_CONVERT
 * physical op writing into a dedicated alternate arena slot.
 */

#include "cpu_features.h"
#include "model_internal.h"
#include "session_internal.h"
#include "executor_internal.h"
#include "../kernels/nhwc_internal.h"
#include "../kernels/scalar_kernels.h"

#include <stddef.h>
#include <stdint.h>

/* Provided by x64_rec_backend_arena.c (compiled under the same gate). */
uint64_t lw_x64_rec_arena_align(uint64_t value, uint64_t alignment);
lw_status lw_x64_rec_arena_alloc(uint64_t* cursor, uint64_t bytes, uint64_t alignment,
                                 uint64_t* out_offset, lw_error* error);

/* Provided by layout_convert_x64.c (same gate; declared again here so the DET
 * backend does not depend on the REC fast-plan header). */
void lw_x64_fast_nchw_to_nhwc(const float* source, float* destination,
                              uint32_t batch, uint32_t channels,
                              uint32_t height, uint32_t width);
void lw_x64_fast_nhwc_to_nchw(const float* source, float* destination,
                              uint32_t batch, uint32_t channels,
                              uint32_t height, uint32_t width);

typedef enum lw_x64_det_compile_result {
    LW_X64_DET_COMPILE_OK = 0,
    LW_X64_DET_COMPILE_UNSUPPORTED = 1,
    LW_X64_DET_COMPILE_INVALID_GRAPH = 2,
    LW_X64_DET_COMPILE_OUT_OF_MEMORY = 3
} lw_x64_det_compile_result;

typedef enum lw_x64_det_compile_strategy {
    LW_X64_DET_COMPILE_NCHW = 0,
    LW_X64_DET_COMPILE_NHWC = 1,
    LW_X64_DET_COMPILE_NHWC_DIRECT = 2
} lw_x64_det_compile_strategy;

typedef enum lw_x64_det_storage_layout {
    LW_X64_DET_LAYOUT_NCHW = 0,
    LW_X64_DET_LAYOUT_NHWC = 1
} lw_x64_det_storage_layout;

typedef enum lw_x64_det_op_kind {
    LW_X64_DET_OP_DENSE = 1,
    LW_X64_DET_OP_POINTWISE = 2,
    LW_X64_DET_OP_DEPTHWISE = 3,
    LW_X64_DET_OP_AFFINE = 4,
    LW_X64_DET_OP_ADD = 5,
    LW_X64_DET_OP_MUL = 6,
    LW_X64_DET_OP_DIV = 7,
    LW_X64_DET_OP_RELU = 8,
    LW_X64_DET_OP_ERF = 9,
    LW_X64_DET_OP_GELU = 10,
    LW_X64_DET_OP_HARD_SIGMOID = 11,
    LW_X64_DET_OP_REDUCE_MEAN = 12,
    LW_X64_DET_OP_AVG_POOL = 13,
    LW_X64_DET_OP_MAX_POOL = 14,
    LW_X64_DET_OP_CONV_NCHW = 15,
    LW_X64_DET_OP_LAYOUT_CONVERT = 16,
    LW_X64_DET_OP_CONV_TRANSPOSE_NCHW = 17,
    LW_X64_DET_OP_RESIZE_NCHW = 18,
    LW_X64_DET_OP_SIGMOID_NCHW = 19,
    LW_X64_DET_OP_CONCAT_NCHW = 20,
    LW_X64_DET_OP_CONV_TRANSPOSE_NHWC = 21,
    LW_X64_DET_OP_CONV_TRANSPOSE_NHWC_C1 = 22,
    LW_X64_DET_OP_RESIZE_NHWC = 23,
    LW_X64_DET_OP_CONCAT_NHWC = 24,
    LW_X64_DET_OP_GENERIC_UNSUPPORTED = 0xffff
} lw_x64_det_op_kind;

typedef enum lw_x64_det_conv_kind {
    LW_X64_DET_CONV_POINTWISE = 1,
    LW_X64_DET_CONV_DEPTHWISE = 2,
    LW_X64_DET_CONV_DENSE = 3
} lw_x64_det_conv_kind;

typedef enum lw_x64_det_pointwise_kernel {
    LW_X64_DET_PW_6X16 = 0,
    LW_X64_DET_PW_4X16 = 1,
    LW_X64_DET_PW_3X32 = 2,
    LW_X64_DET_PW_2X32 = 3
} lw_x64_det_pointwise_kernel;

typedef enum lw_x64_det_nchw_pointwise_kernel {
    LW_X64_DET_NCHW_PW_FMA4 = 0,
    LW_X64_DET_NCHW_PW_FMA8 = 1
} lw_x64_det_nchw_pointwise_kernel;

typedef enum lw_x64_det_nchw_conv_kernel {
    LW_X64_DET_NCHW_CONV_POINTWISE_PACKED = 0,
    LW_X64_DET_NCHW_CONV_3X3S2 = 1,
    LW_X64_DET_NCHW_CONV_3X3_UNIT = 2,
    LW_X64_DET_NCHW_CONV_2X2_PADEND1 = 3,
    LW_X64_DET_NCHW_CONV_DEPTHWISE3X3 = 4,
    LW_X64_DET_NCHW_CONV_DEPTHWISE3X3_S2X1 = 5,
    LW_X64_DET_NCHW_CONV_DEPTHWISE5X5 = 6,
    LW_X64_DET_NCHW_CONV_SCALAR = 7
} lw_x64_det_nchw_conv_kernel;

typedef enum lw_x64_det_broadcast_kind {
    LW_X64_DET_BROADCAST_SAME = 0,
    LW_X64_DET_BROADCAST_RIGHT_SCALAR = 1,
    LW_X64_DET_BROADCAST_LEFT_SCALAR = 2,
    LW_X64_DET_BROADCAST_RIGHT_CHANNEL = 3,
    LW_X64_DET_BROADCAST_LEFT_CHANNEL = 4,
    LW_X64_DET_BROADCAST_GENERAL = 5
} lw_x64_det_broadcast_kind;

typedef struct lw_x64_det_value {
    uint64_t offset;
    /* Alternate-layout copy written by a LAYOUT_CONVERT op. Zero until the
     * arena plan assigns it; alt_producer == -1 means "not needed". */
    uint64_t alt_offset;
    uint64_t bytes;
    const float* constant_data;
    int32_t producer;
    int32_t last_use;
    int32_t alt_producer;
    int32_t alt_last_use;
    int32_t dimensions[4];
    uint32_t rank;
    uint8_t layout;
    uint8_t alt_layout;
    uint8_t alias;
    uint8_t reserved;
    uint32_t alias_of;
} lw_x64_det_value;

typedef struct lw_x64_det_conv_op {
    uint64_t input_offset;
    uint64_t output_offset;
    const float* packed_weights;
    const float* original_weights;
    const float* bias;
    /* Folded epilogue extras: a per-channel constant Add applied to the
     * finished accumulator (post_bias), and an elementwise residual Add read
     * from the arena at residual_offset when has_residual != 0. */
    const float* post_bias;
    uint64_t residual_offset;
    uint8_t has_residual;
    uint8_t scalar_fallback;
    uint8_t pointwise_kernel;
    uint8_t nchw_pointwise_kernel;
    uint8_t nchw_conv_kernel;
    uint32_t input_channels;
    uint32_t output_channels;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t output_height;
    uint32_t output_width;
    uint32_t kernel_h;
    uint32_t kernel_w;
    uint32_t weight_h;
    uint32_t weight_w;
    uint32_t stride_h;
    uint32_t stride_w;
    uint32_t pad_top;
    uint32_t pad_left;
    uint32_t pad_bottom;
    uint32_t pad_right;
    uint32_t groups;
    uint32_t dense_kc;
    uint64_t scratch_bytes;
    uint16_t kernel_kind;
    uint16_t activation;
} lw_x64_det_conv_op;

typedef struct lw_x64_det_binary_op {
    uint64_t left_offset;
    uint64_t right_offset;
    uint64_t output_offset;
    const float* left_constant;
    const float* right_constant;
    uint64_t element_count;
    uint32_t pixels;
    uint32_t channels;
    uint16_t operation;
    uint8_t broadcast_kind;
    uint8_t layout;
    int32_t dimensions[4];
} lw_x64_det_binary_op;

typedef struct lw_x64_det_unary_op {
    uint64_t input_offset;
    uint64_t output_offset;
    uint64_t element_count;
    float alpha;
    float beta;
    int32_t dimensions[4];
    uint32_t rank;
    uint8_t layout;
    uint8_t reserved[3];
} lw_x64_det_unary_op;

typedef struct lw_x64_det_affine_op {
    uint64_t input_offset;
    uint64_t output_offset;
    const float* mul;
    const float* add;
    const float* scale;
    const float* bias;
    const float* mean;
    const float* variance;
    float epsilon;
    uint8_t channel_major;
    uint32_t pixels;
    uint32_t channels;
} lw_x64_det_affine_op;

typedef struct lw_x64_det_reduce_op {
    uint64_t input_offset;
    uint64_t output_offset;
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    uint8_t layout;
    uint8_t reserved[3];
} lw_x64_det_reduce_op;

typedef struct lw_x64_det_pool_op {
    uint64_t input_offset;
    uint64_t output_offset;
    int32_t input_dimensions[4];
    int32_t output_dimensions[4];
    int32_t kernel[2];
    int32_t strides[2];
    int32_t pads[4];
    uint8_t count_include_pad;
    uint8_t is_max;
    uint8_t layout;
    uint8_t reserved;
} lw_x64_det_pool_op;

typedef struct lw_x64_det_convert_op {
    uint64_t input_offset;
    uint64_t output_offset;
    uint32_t batch;
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    uint8_t to_nhwc;
    uint8_t reserved[3];
} lw_x64_det_convert_op;

typedef struct lw_x64_det_conv_transpose_op {
    uint64_t input_offset;
    uint64_t output_offset;
    const float* weights;
    const float* packed_weights;
    const float* bias;
    int32_t input_dimensions[4];
    int32_t output_dimensions[4];
    uint8_t layout;
    uint8_t reserved[3];
} lw_x64_det_conv_transpose_op;

typedef struct lw_x64_det_resize_op {
    uint64_t input_offset;
    uint64_t output_offset;
    uint32_t rank;
    int32_t input_dimensions[4];
    int32_t output_dimensions[4];
    float scales[4];
    uint8_t layout;
    uint8_t reserved[3];
} lw_x64_det_resize_op;

typedef struct lw_x64_det_concat_op {
    uint64_t input_offsets[LWM_V0_MAX_NODE_INPUTS];
    uint64_t output_offset;
    uint32_t input_count;
    uint32_t input_ranks[LWM_V0_MAX_NODE_INPUTS];
    int32_t input_dimensions[LWM_V0_MAX_NODE_INPUTS][4];
    uint32_t output_rank;
    int32_t output_dimensions[4];
    int32_t axis;
    uint8_t layout;
    uint8_t reserved[3];
} lw_x64_det_concat_op;

typedef struct lw_x64_det_op {
    uint16_t kind;
    uint16_t reserved;
    uint32_t semantic_begin;
    uint16_t semantic_count;
    uint16_t flags;
    union {
        lw_x64_det_conv_op conv;
        lw_x64_det_binary_op binary;
        lw_x64_det_unary_op unary;
        lw_x64_det_affine_op affine;
        lw_x64_det_reduce_op reduce;
        lw_x64_det_pool_op pool;
        lw_x64_det_convert_op convert;
        lw_x64_det_conv_transpose_op conv_transpose;
        lw_x64_det_resize_op resize;
        lw_x64_det_concat_op concat;
    } data;
} lw_x64_det_op;

typedef struct lw_x64_det_constant {
    void* data;
    uint64_t bytes;
} lw_x64_det_constant;

typedef struct lw_x64_det_program {
    const lw_model* model;
    lw_cpu_capabilities cpu;
    uint64_t model_signature;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t input_value;
    uint32_t output_value;
    lw_x64_det_op* ops;
    uint32_t op_count;
    lw_x64_det_value* values;
    uint32_t value_count;
    uint32_t semantic_consumed;
    uint32_t semantic_elided;
    uint32_t semantic_fused;
    uint32_t unsupported_nodes;
    uint32_t layout_conversions;
    uint8_t direct_nhwc;
    uint8_t backend_layout;
    uint16_t reserved;
    uint64_t arena_bytes;
    uint64_t scratch_bytes;
    uint32_t packed_constant_count;
    lw_x64_det_constant* constants;
    /* Effective-layout walk output, kept for the contract driver. */
    uint8_t* effective_node_layout;
    uint32_t nhwc_effective_nodes;
    uint32_t nchw_effective_nodes;
} lw_x64_det_program;

typedef struct lw_x64_det_profile {
    uint64_t total_ns;
    uint64_t pointwise_ns;
    uint64_t dense_ns;
    uint64_t depthwise_ns;
    uint64_t conv_nchw_ns;
    uint64_t binary_ns;
    uint64_t unary_ns;
    uint64_t reduce_ns;
    uint64_t pool_ns;
    uint64_t convert_ns;
    uint64_t conv_transpose_ns;
    uint64_t resize_ns;
    uint64_t sigmoid_ns;
    uint64_t concat_ns;
} lw_x64_det_profile;

typedef struct lw_x64_det_instance {
    const lw_x64_det_program* program;
    uint8_t* arena;
    uint8_t* scratch;
    /* Per-worker dense-kernel scratch slices (64B-aligned, one per pool
     * worker) so sharded dense ops never share a patch buffer. */
    uint8_t* shard_scratch;
    uint64_t shard_scratch_slice_bytes;
    lw_x64_det_profile profile;
} lw_x64_det_instance;

lw_x64_det_compile_result lw_x64_det_backend_compile_ex(
    const lw_model* model, uint32_t height, uint32_t width,
    lw_x64_det_compile_strategy strategy, lw_x64_det_program** out_program,
    lw_error* error);
void lw_x64_det_program_free(lw_x64_det_program* program);
lw_status lw_x64_det_instance_create(const lw_x64_det_program* program,
                                     lw_x64_det_instance** out, lw_error* error);
void lw_x64_det_instance_free(lw_x64_det_instance* instance);
float* lw_x64_det_instance_input(lw_x64_det_instance* instance, uint64_t* element_count);
lw_status lw_x64_det_instance_run(lw_x64_det_instance* instance, lw_error* error);
/* Timed run that maps physical-op elapsed time onto the semantic profile and
 * copies the graph-output value (always NCHW [1,1,H,W]) into `output`. */
lw_status lw_x64_det_instance_run_profiled(lw_x64_det_instance* instance, float* output,
                                           uint64_t output_element_count,
                                           lw_execution_profile* profile, lw_error* error);
/* Threaded variant: shardable physical ops fan out over `pool` (worker 0 runs
 * on the caller thread) up to `worker_count` workers. Whole-tile/whole-row
 * partitioning keeps sharded output bit-identical to the serial run. */
lw_status lw_x64_det_instance_run_profiled_ex(lw_x64_det_instance* instance, float* output,
                                              uint64_t output_element_count,
                                              lw_thread_pool* pool, uint32_t worker_count,
                                              lw_execution_profile* profile, lw_error* error);
/* Test-only internal hook: execute one compiled physical op. */
lw_status lw_x64_det_instance_run_op(lw_x64_det_instance* instance, uint32_t op_index,
                                     lw_error* error);
/* Test-only internal hook: execute one physical op through the threaded
 * sharding dispatch (pool NULL keeps the serial path). */
lw_status lw_x64_det_instance_run_op_ex(lw_x64_det_instance* instance, uint32_t op_index,
                                        lw_thread_pool* pool, uint32_t worker_count,
                                        lw_error* error);

#endif
