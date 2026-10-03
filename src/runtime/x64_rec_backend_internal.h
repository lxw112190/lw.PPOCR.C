#ifndef LW_X64_REC_BACKEND_INTERNAL_H
#define LW_X64_REC_BACKEND_INTERNAL_H

#include "cpu_features.h"
#include "atomic_internal.h"
#include "model_internal.h"
#include "session_internal.h"
#include "executor_internal.h"
#include "../kernels/nhwc_internal.h"
#include "../kernels/packed_conv_internal.h"
#include "../kernels/packed_conv3x3_internal.h"
#include "../kernels/scalar_kernels.h"

#include <stddef.h>
#include <stdint.h>

/* Keep the individual physical ops for diagnostic/reference execution. */
#define LW_X64_REC_OP_FFN_BEGIN 1u

typedef enum lw_x64_rec_compile_result {
    LW_X64_REC_COMPILE_OK = 0,
    LW_X64_REC_COMPILE_UNSUPPORTED = 1,
    LW_X64_REC_COMPILE_INVALID_GRAPH = 2,
    LW_X64_REC_COMPILE_OUT_OF_MEMORY = 3
} lw_x64_rec_compile_result;

typedef enum lw_x64_rec_storage_layout {
    LW_X64_REC_LAYOUT_NHWC = 0,
    LW_X64_REC_LAYOUT_TC = 1,
    LW_X64_REC_LAYOUT_VECTOR = 2,
    LW_X64_REC_LAYOUT_SCALAR = 3,
    LW_X64_REC_LAYOUT_NCHW = 4
} lw_x64_rec_storage_layout;

typedef enum lw_x64_rec_backend_layout {
    LW_X64_REC_BACKEND_NHWC = 0,
    LW_X64_REC_BACKEND_NCHW = 1
} lw_x64_rec_backend_layout;

typedef enum lw_x64_rec_compile_strategy {
    LW_X64_REC_COMPILE_NHWC = 0,
    LW_X64_REC_COMPILE_NCHW = 1
} lw_x64_rec_compile_strategy;

typedef enum lw_x64_rec_op_kind {
    LW_X64_REC_OP_DENSE = 1,
    LW_X64_REC_OP_POINTWISE = 2,
    LW_X64_REC_OP_DEPTHWISE = 3,
    LW_X64_REC_OP_POINTWISE_NCHW = 19,
    LW_X64_REC_OP_STEM_NCHW = 20,
    LW_X64_REC_OP_DEPTHWISE_NCHW = 21,
    LW_X64_REC_OP_AFFINE = 4,
    LW_X64_REC_OP_ADD = 5,
    LW_X64_REC_OP_MUL = 6,
    LW_X64_REC_OP_DIV = 7,
    LW_X64_REC_OP_RELU = 8,
    LW_X64_REC_OP_ERF = 9,
    LW_X64_REC_OP_GELU = 10,
    LW_X64_REC_OP_HARD_SIGMOID = 11,
    LW_X64_REC_OP_REDUCE_MEAN = 12,
    LW_X64_REC_OP_AVG_POOL = 13,
    LW_X64_REC_OP_MAX_POOL = 14,
    LW_X64_REC_OP_TRANSPOSE = 15,
    LW_X64_REC_OP_MATMUL = 16,
    LW_X64_REC_OP_CONCAT = 17,
    LW_X64_REC_OP_RESIZE = 18,
    LW_X64_REC_OP_SOFTMAX = 22,
    LW_X64_REC_OP_SUB = 23,
    LW_X64_REC_OP_POW = 24,
    LW_X64_REC_OP_SIGMOID = 25,
    LW_X64_REC_OP_SQRT = 26,
    LW_X64_REC_OP_SLICE = 27,
    LW_X64_REC_OP_GENERIC_UNSUPPORTED = 0xffff
} lw_x64_rec_op_kind;

typedef enum lw_x64_rec_conv_kind {
    LW_X64_REC_CONV_POINTWISE = 1,
    LW_X64_REC_CONV_DEPTHWISE = 2,
    LW_X64_REC_CONV_DENSE = 3
} lw_x64_rec_conv_kind;

typedef enum lw_x64_rec_pointwise_kernel {
    LW_X64_REC_PW_6X16 = 0,
    LW_X64_REC_PW_4X16 = 1,
    LW_X64_REC_PW_3X32 = 2,
    LW_X64_REC_PW_2X32 = 3
} lw_x64_rec_pointwise_kernel;

typedef enum lw_x64_rec_nchw_pointwise_kernel {
    LW_X64_REC_NCHW_PW_FMA4 = 0,
    LW_X64_REC_NCHW_PW_FMA8 = 1
} lw_x64_rec_nchw_pointwise_kernel;

typedef enum lw_x64_rec_broadcast_kind {
    LW_X64_REC_BROADCAST_SAME = 0,
    LW_X64_REC_BROADCAST_RIGHT_SCALAR = 1,
    LW_X64_REC_BROADCAST_LEFT_SCALAR = 2,
    LW_X64_REC_BROADCAST_RIGHT_CHANNEL = 3,
    LW_X64_REC_BROADCAST_LEFT_CHANNEL = 4,
    LW_X64_REC_BROADCAST_GENERAL = 5,
    /* Right operand holds one scalar per output pixel (element_count /
     * channels floats), broadcast over the channel axis. Bit-identical
     * to the canonical scalar broadcast for every operation. */
    LW_X64_REC_BROADCAST_RIGHT_PIXEL_SCALAR = 6
} lw_x64_rec_broadcast_kind;

typedef struct lw_x64_rec_value {
    uint64_t offset;
    uint64_t bytes;
    const float* constant_data;
    int32_t producer;
    int32_t last_use;
    int32_t dimensions[4];
    uint32_t rank;
    uint8_t layout;
    uint8_t alias;
    uint16_t reserved;
    uint32_t alias_of;
} lw_x64_rec_value;

typedef struct lw_x64_rec_conv_op {
    uint64_t input_offset;
    uint64_t output_offset;
    const float* packed_weights;
    const float* original_weights;
    const float* bias;
    uint8_t scalar_fallback;
    uint8_t pointwise_kernel;
    uint8_t nchw_pointwise_kernel;
    uint8_t conv_reserved;
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
    /* Fused epilogue state filled by the compiler's conv -> add -> gelu
     * matcher. post_bias is a folded channel bias applied after accumulation;
     * residual_offset/has_residual describe a fused elementwise residual add
     * that reads from the arena. All zero when nothing was folded in. */
    const float* post_bias;
    uint64_t residual_offset;
    uint8_t has_residual;
    uint8_t fusion_reserved[7];
} lw_x64_rec_conv_op;

typedef struct lw_x64_rec_binary_op {
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
    uint8_t mixed_nhwc; /* 0=uniform layout, 1=right operand physical NHWC, 2=left */
    /* 1 = channel broadcast over an NCHW-physical rank-4 tensor (channel axis
     * is dims[1]); 0 = channel-last or non-channel broadcast.  The NCHW
     * executor branch keys on this flag so canonical rank-3 [.., .., C]
     * tensors keep the channel-last path. */
    uint8_t channel_major_nchw;
    int32_t dimensions[4];
} lw_x64_rec_binary_op;

typedef struct lw_x64_rec_unary_op {
    uint64_t input_offset;
    uint64_t output_offset;
    uint64_t element_count;
    float alpha;
    float beta;
    int32_t dimensions[4];
    uint32_t rank;
} lw_x64_rec_unary_op;

typedef struct lw_x64_rec_affine_op {
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
} lw_x64_rec_affine_op;

typedef struct lw_x64_rec_reduce_op {
    uint64_t input_offset;
    uint64_t output_offset;
    uint32_t batch;
    uint32_t height;
    uint32_t width;
    uint32_t channels;
    /* general != 0: not the NCHW/NHWC global-pool pattern; run the
     * canonical scalar reduce with the original axes for bit-identity. */
    uint32_t general;
    uint32_t input_rank;
    uint32_t output_rank;
    uint32_t axes_count;
    uint32_t keep_dimensions;
    uint32_t no_op_with_empty_axes;
    int32_t axes[LW_MAX_DIMS];
    int32_t input_dimensions[LW_MAX_DIMS];
    int32_t output_dimensions[LW_MAX_DIMS];
} lw_x64_rec_reduce_op;

typedef struct lw_x64_rec_pool_op {
    uint64_t input_offset;
    uint64_t output_offset;
    int32_t input_dimensions[4];
    int32_t output_dimensions[4];
    int32_t kernel[2];
    int32_t strides[2];
    int32_t pads[4];
    uint8_t count_include_pad;
    uint8_t is_max;
} lw_x64_rec_pool_op;

typedef struct lw_x64_rec_transpose_op {
    uint64_t input_offset;
    uint64_t output_offset;
    uint32_t rank;
    int32_t input_dimensions[LW_MAX_DIMS];
    int32_t output_dimensions[LW_MAX_DIMS];
    int32_t permutation[LW_MAX_DIMS];
} lw_x64_rec_transpose_op;

typedef struct lw_x64_rec_matmul_op {
    uint64_t input_offset;
    uint64_t output_offset;
    uint64_t weights_offset;
    const float* weights;
    float* packed_weights;
    uint32_t batch;
    uint32_t rows;
    uint32_t inner;
    uint32_t columns;
    uint32_t general;
    uint32_t input_rank;
    uint32_t weights_rank;
    uint32_t output_rank;
    int32_t input_dimensions[LW_MAX_DIMS];
    int32_t weights_dimensions[LW_MAX_DIMS];
    int32_t output_dimensions[LW_MAX_DIMS];
} lw_x64_rec_matmul_op;

typedef struct lw_x64_rec_softmax_op {
    uint64_t input_offset;
    uint64_t output_offset;
    int32_t axis;
    uint32_t rank;
    int32_t dimensions[4];
} lw_x64_rec_softmax_op;

/* Channel-axis (NCHW axis 1) concat over rank-4 feature maps stored NHWC:
 * per-pixel concatenated channel copies, pure data movement. */
typedef struct lw_x64_rec_concat_op {
    uint64_t input_offsets[LWM_V0_MAX_NODE_INPUTS];
    uint64_t output_offset;
    uint32_t input_channels[LWM_V0_MAX_NODE_INPUTS];
    uint32_t input_count;
    uint32_t output_channels;
    uint32_t pixels;
    int32_t axis;
} lw_x64_rec_concat_op;

/* Generic slice over a canonical-layout (non-NHWC) tensor; executed by
 * the shared scalar slice kernel so results match the interpreter. */
typedef struct lw_x64_rec_slice_op {
    uint64_t input_offset;
    uint64_t output_offset;
    uint32_t rank;
    uint32_t output_rank;
    uint32_t slice_count;
    int32_t input_dimensions[LW_MAX_DIMS];
    int32_t output_dimensions[LW_MAX_DIMS];
    int32_t starts[LW_MAX_DIMS];
    int32_t ends[LW_MAX_DIMS];
    int32_t axes[LW_MAX_DIMS];
    int32_t steps[LW_MAX_DIMS];
} lw_x64_rec_slice_op;

typedef struct lw_x64_rec_op {
    uint16_t kind;
    uint16_t reserved;
    uint32_t semantic_begin;
    uint16_t semantic_count;
    uint16_t flags;
    union {
        lw_x64_rec_conv_op conv;
        lw_x64_rec_binary_op binary;
        lw_x64_rec_unary_op unary;
        lw_x64_rec_affine_op affine;
        lw_x64_rec_reduce_op reduce;
        lw_x64_rec_pool_op pool;
        lw_x64_rec_transpose_op transpose;
        lw_x64_rec_matmul_op matmul;
        lw_x64_rec_softmax_op softmax;
        lw_x64_rec_concat_op concat;
        lw_x64_rec_slice_op slice;
    } data;
} lw_x64_rec_op;

typedef struct lw_x64_rec_constant {
    void* data;
    uint64_t bytes;
} lw_x64_rec_constant;

typedef struct lw_x64_rec_ctc_tail {
    uint8_t enabled;
    uint8_t packed_weights_borrowed;
    uint8_t reserved[2];
    uint32_t activation_value;
    uint32_t classes;
    uint32_t rows;
    uint32_t inner;
    uint32_t weight_tensor;
    uint32_t bias_tensor;
    float* packed_weights;
    const float* bias;
} lw_x64_rec_ctc_tail;

typedef struct lw_x64_rec_program {
    const lw_model* model;
    lw_cpu_capabilities cpu;
    uint64_t model_signature;
    uint32_t target_width;
    uint32_t time_steps;
    uint32_t class_count;
    uint32_t input_value;
    uint32_t output_value;
    lw_x64_rec_op* ops;
    uint32_t op_count;
    lw_x64_rec_value* values;
    uint32_t value_count;
    uint32_t semantic_consumed;
    uint32_t semantic_elided;
    uint32_t semantic_fused;
    uint32_t unsupported_nodes;
    uint32_t generic_ops;
    uint32_t layout_conversions;
    uint8_t direct_nhwc;
    uint8_t ctc_fused;
    uint16_t reserved;
    uint64_t arena_bytes;
    uint64_t scratch_bytes;
    uint8_t backend_layout;
    uint32_t packed_constant_count;
    lw_x64_rec_constant* constants;
    const struct lw_x64_rec_program* packed_source;
    uint64_t owned_constant_bytes;
    uint64_t borrowed_constant_bytes;
    uint32_t borrowed_constant_count;
    lw_x64_rec_ctc_tail ctc;
    /* Number of additional owners beyond the original compiler caller. */
    lw_atomic_u32 shared_refs;
} lw_x64_rec_program;

typedef struct lw_x64_rec_profile {
    uint64_t total_ns;
    uint64_t pointwise_ns;
    uint64_t dense_ns;
    uint64_t depthwise_ns;
    uint64_t binary_ns;
    uint64_t unary_ns;
    uint64_t reduce_ns;
    uint64_t pool_ns;
    uint64_t transpose_ns;
    uint64_t matmul_ns;
    uint64_t ctc_ns;
    uint64_t ffn_ns;
    uint32_t ffn_calls;
    /* Pointwise epilogue diagnostics; populated only when REC profiling is on. */
    uint64_t pw_epi_ns[7];
    uint64_t pw_epi_calls[7];
    uint64_t pw_epi_macs[7];
} lw_x64_rec_profile;

typedef struct lw_x64_rec_instance {
    const lw_x64_rec_program* program;
    uint8_t* arena;
    uint8_t* scratch;
    float* ctc_scores;
    uint32_t* best_indices;
    float* best_probabilities;
    lw_x64_rec_profile profile;
    /* Borrowed intra-op pool for the CTC head matmul (NULL = serial).
     * Owned by the recognizer; every slot instance of one recognizer
     * shares it, and slots never run concurrently. */
    lw_thread_pool* thread_pool;
    uint32_t intra_op_workers;
    /* Lazily allocated per-worker scratch for sharded dense convs
     * (intra_op_workers * per-op scratch bytes, grown on demand).
     * Always owned by the instance, independent of owns_workspace. */
    uint8_t* dense_shard_scratch;
    uint64_t dense_shard_scratch_bytes;
    /* 0 when arena/scratch/CTC buffers are borrowed from a shared
     * cross-width workspace owned by the recognizer. */
    uint8_t owns_workspace;
} lw_x64_rec_instance;

uint64_t lw_x64_rec_arena_align(uint64_t value, uint64_t alignment);
lw_status lw_x64_rec_arena_alloc(uint64_t* cursor, uint64_t bytes, uint64_t alignment,
                                 uint64_t* out_offset, lw_error* error);
/* Generalized entry: explicit input height (REC callers pass 48; the CLS
 * backbone reuses this lowering with its own fixed input shape). */
lw_x64_rec_compile_result lw_x64_rec_backend_compile_input(
    const lw_model* model, uint32_t input_height, uint32_t target_width,
    lw_x64_rec_compile_strategy strategy, uint32_t allow_ctc_tail,
    lw_x64_rec_program** out_program, lw_error* error);
lw_x64_rec_compile_result lw_x64_rec_backend_compile_ex(
    const lw_model* model, uint32_t target_width,
    lw_x64_rec_compile_strategy strategy,
    lw_x64_rec_program** out_program, lw_error* error);
lw_x64_rec_compile_result lw_x64_rec_backend_compile(
    const lw_model* model, uint32_t target_width, lw_x64_rec_program** out_program,
    lw_error* error);
lw_x64_rec_compile_result lw_x64_rec_backend_compile_shared(
    const lw_model* model, uint32_t target_width, const lw_x64_rec_program* packed_source,
    lw_x64_rec_program** out_program, lw_error* error);
void lw_x64_rec_program_free(lw_x64_rec_program* program);
void lw_x64_rec_program_retain(lw_x64_rec_program* program);
lw_status lw_x64_rec_instance_create(const lw_x64_rec_program* program,
                                     lw_x64_rec_instance** out, lw_error* error);
/* Borrowed-workspace variant: the instance uses the caller's buffers
 * (sized at least to the program's requirements) and never frees
 * them. ctc_rows is the capacity of the three CTC arrays. */
lw_status lw_x64_rec_instance_create_borrowed(
    const lw_x64_rec_program* program,
    uint8_t* arena, uint64_t arena_bytes,
    uint8_t* scratch, uint64_t scratch_bytes,
    float* ctc_scores, uint32_t* best_indices, float* best_probabilities,
    uint64_t ctc_rows, lw_x64_rec_instance** out, lw_error* error);
void lw_x64_rec_instance_free(lw_x64_rec_instance* instance);
/* Attach a borrowed intra-op pool for the CTC head projection
 * (workers <= 1 or NULL pool detaches; serial fallback). */
void lw_x64_rec_instance_set_thread_pool(lw_x64_rec_instance* instance,
                                         lw_thread_pool* pool, uint32_t workers);
float* lw_x64_rec_instance_input(lw_x64_rec_instance* instance, uint64_t* element_count);
lw_status lw_x64_rec_instance_run(lw_x64_rec_instance* instance, lw_error* error);
lw_status lw_x64_rec_instance_run_backbone(lw_x64_rec_instance* instance, lw_error* error);
/* Test-only internal hook: execute one compiled physical op. */
lw_status lw_x64_rec_instance_run_op(lw_x64_rec_instance* instance, uint32_t op_index, lw_error* error);
/* Private contract hook: execute an eligible adjacent pair in cache-local tiles. */
lw_status lw_x64_rec_instance_run_ffn_pair(lw_x64_rec_instance* instance,
                                           uint32_t op_index, lw_error* error);

#endif
