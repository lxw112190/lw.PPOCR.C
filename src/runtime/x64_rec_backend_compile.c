#include "x64_rec_backend_internal.h"
#include "ctc_projection_internal.h"
#include "lwm_read.h"
#include "operator_internal.h"
#include "packed_matmul_internal.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int rec_backend_cpu_supported(lw_cpu_capabilities cpu) {
#if defined(__EMSCRIPTEN__) && defined(__wasm_simd128__)
    (void)cpu;
    return 1;
#else
    return lw_simd_level_is_avx2(cpu.simd) && cpu.has_avx2_fma;
#endif
}

static lw_status make_shape_session(const lw_model* model, uint32_t height, uint32_t width,
                                    lw_session** out, lw_error* error) {
    lw_tensor_desc input;
    lw_tensor_desc_init(&input);
    input.dtype = LW_DTYPE_F32;
    input.rank = 4u;
    input.dimensions[0] = 1;
    input.dimensions[1] = 3;
    input.dimensions[2] = (int32_t)height;
    input.dimensions[3] = (int32_t)width;
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_REC_LAZY_FALLBACK)
    /* The physical compiler consumes resolved tensor metadata, never the
     * canonical execution workspace. Avoid growing the WASM heap to the
     * canonical workspace high-water mark during each width compile. */
    return lw_session_create_metadata_only(model, &input, 1u, NULL, out, error);
#else
    return lw_session_create(model, &input, 1u, NULL, out, error);
#endif
}

static const uint8_t* node_bytes(const lw_model* model, uint32_t index) {
    return model->bytes + (size_t)model->node_offset + (size_t)index * LWM_V0_NODE_SIZE;
}

static const float* constant_f32(const lw_model* model, uint32_t index) {
    const uint8_t* tensor = model->bytes + (size_t)model->tensor_offset +
                            (size_t)index * LWM_V0_TENSOR_SIZE;
    return (const float*)(const void*)(model->bytes + (size_t)lwm_read_u64(tensor + 48u));
}

static int rec_value_is_nhwc_physical(const lw_model* model, const lw_session* session,
                                      uint32_t tensor);

/* Physical extent of the fastest-varying (channel) axis for a broadcast
 * against this output value: NHWC-stored rank-4 values keep C last,
 * canonical-order rank-4 values (attention chain) keep W last, and
 * lower ranks use the canonical trailing axis directly. */
static uint32_t rec_binary_channel_extent(const lw_model* model, const lw_session* session,
                                          const lw_x64_rec_value* out_value,
                                          uint32_t out_index) {
    if (out_value->rank == 0u) return 1u;
    if (out_value->rank == 4u &&
        !rec_value_is_nhwc_physical(model, session, out_index)) {
        return (uint32_t)out_value->dimensions[2];
    }
    return (uint32_t)out_value->dimensions[out_value->rank - 1u];
}
static float read_f32(const uint8_t* p) {
    uint32_t bits = lwm_read_u32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void fill_value(const lw_session* session, uint32_t i, lw_x64_rec_value* value,
                       lw_x64_rec_compile_strategy strategy) {
    const lw_runtime_tensor* tensor = &session->tensors[i];
    uint32_t j;
    memset(value, 0, sizeof(*value));
    value->bytes = tensor->byte_size;
    value->rank = tensor->rank > 4u ? 4u : tensor->rank;
    if (tensor->rank == 4u) {
        if (strategy == LW_X64_REC_COMPILE_NCHW) {
            value->dimensions[0] = tensor->dimensions[0];
            value->dimensions[1] = tensor->dimensions[1];
            value->dimensions[2] = tensor->dimensions[2];
            value->dimensions[3] = tensor->dimensions[3];
            value->layout = LW_X64_REC_LAYOUT_NCHW;
        } else {
            value->dimensions[0] = tensor->dimensions[0];
            value->dimensions[1] = tensor->dimensions[2];
            value->dimensions[2] = tensor->dimensions[3];
            value->dimensions[3] = tensor->dimensions[1];
            value->layout = LW_X64_REC_LAYOUT_NHWC;
        }
    } else {
        for (j = 0u; j < value->rank; ++j) value->dimensions[j] = tensor->dimensions[j];
        value->layout = LW_X64_REC_LAYOUT_SCALAR;
    }
    value->producer = -1;
    value->last_use = -1;
    if ((tensor->flags & LWM_V0_TENSOR_FLAG_CONSTANT) != 0u) value->constant_data = constant_f32(session->model, i);
}

static int add_constant(lw_x64_rec_program* program, void* data, uint64_t bytes) {
    lw_x64_rec_constant* next;
    if (data == NULL || bytes == 0u) return 0;
    next = (lw_x64_rec_constant*)realloc(program->constants,
        (size_t)(program->packed_constant_count + 1u) * sizeof(*next));
    if (next == NULL) return 0;
    program->constants = next;
    program->constants[program->packed_constant_count].data = data;
    program->constants[program->packed_constant_count].bytes = bytes;
    ++program->packed_constant_count;
    program->owned_constant_bytes += bytes;
    return 1;
}

static void* alloc_constant(lw_x64_rec_program* program, uint64_t bytes) {
    void* data;
    if (bytes == 0u || bytes > SIZE_MAX) return NULL;
    data = malloc((size_t)bytes);
    if (data == NULL || !add_constant(program, data, bytes)) {
        free(data);
        return NULL;
    }
    return data;
}

/* A width-specific op can borrow only a pack with the same semantic node and
 * packing contract. The root stays alive through packed_source's reference. */
static const lw_x64_rec_op* find_packed_source_op(const lw_x64_rec_program* program,
                                                  uint32_t semantic_begin) {
    const lw_x64_rec_program* source = program->packed_source;
    uint32_t i;
    if (source == NULL) return NULL;
    for (i = 0u; i < source->op_count; ++i) {
        if (source->ops[i].semantic_begin == semantic_begin) return &source->ops[i];
    }
    return NULL;
}

static void note_borrow(lw_x64_rec_program* program, uint64_t bytes) {
    program->borrowed_constant_bytes += bytes;
    ++program->borrowed_constant_count;
}

static const float* borrow_conv_pack(lw_x64_rec_program* program,
                                     const lw_x64_rec_op* op, uint64_t bytes) {
    const lw_x64_rec_op* source = find_packed_source_op(program, op->semantic_begin);
    if (source == NULL || source->kind != op->kind ||
        source->data.conv.packed_weights == NULL ||
        source->data.conv.kernel_kind != op->data.conv.kernel_kind ||
        source->data.conv.input_channels != op->data.conv.input_channels ||
        source->data.conv.output_channels != op->data.conv.output_channels ||
        source->data.conv.kernel_h != op->data.conv.kernel_h ||
        source->data.conv.kernel_w != op->data.conv.kernel_w ||
        source->data.conv.original_weights != op->data.conv.original_weights)
        return NULL;
    note_borrow(program, bytes);
    return source->data.conv.packed_weights;
}

static const float* borrow_matmul_pack(lw_x64_rec_program* program,
                                       const lw_x64_rec_op* op, uint64_t bytes) {
    const lw_x64_rec_op* source = find_packed_source_op(program, op->semantic_begin);
    if (source == NULL || source->kind != LW_X64_REC_OP_MATMUL ||
        source->data.matmul.packed_weights == NULL ||
        source->data.matmul.inner != op->data.matmul.inner ||
        source->data.matmul.columns != op->data.matmul.columns ||
        source->data.matmul.weights != op->data.matmul.weights)
        return NULL;
    note_borrow(program, bytes);
    return source->data.matmul.packed_weights;
}

static int detect_ctc(const lw_model* model, const lw_session* session, lw_x64_rec_ctc_tail* tail) {
    uint32_t n = model->info.node_count;
    const uint8_t* matmul;
    const uint8_t* add;
    const uint8_t* softmax;
    uint32_t activation;
    uint32_t weights;
    uint32_t output;
    uint32_t add_output;
    if (n < 3u) return 0;
    matmul = node_bytes(model, n - 3u);
    add = node_bytes(model, n - 2u);
    softmax = node_bytes(model, n - 1u);
    if (lwm_read_u16(matmul) != LW_OP_MATMUL || lwm_read_u16(add) != LW_OP_ADD ||
        lwm_read_u16(softmax) != LW_OP_SOFTMAX || lwm_read_u16(matmul + 2u) != 2u ||
        lwm_read_u16(add + 2u) != 2u || lwm_read_u16(softmax + 2u) != 1u) return 0;
    activation = lwm_read_u32(matmul + 8u);
    weights = lwm_read_u32(matmul + 12u);
    output = lwm_read_u32(matmul + 40u);
    add_output = lwm_read_u32(add + 40u);
    if (lwm_read_u32(add + 8u) != output || session->tensors[weights].rank != 2u ||
        add_output != lwm_read_u32(softmax + 8u)) return 0;
    tail->enabled = 1u;
    tail->activation_value = activation;
    tail->weight_tensor = weights;
    tail->bias_tensor = lwm_read_u32(add + 12u);
    tail->classes = (uint32_t)session->tensors[weights].dimensions[1];
    tail->inner = (uint32_t)session->tensors[weights].dimensions[0];
    tail->rows = (uint32_t)session->tensors[activation].dimensions[session->tensors[activation].rank - 2u];
    return 1;
}

static uint64_t value_elements(const lw_x64_rec_value* value) { return value->bytes / sizeof(float); }

static uint8_t choose_nchw_pointwise_kernel(uint32_t height,
                                               uint32_t input_channels,
                                               uint32_t output_channels,
                                               uint32_t spatial) {
    if (output_channels >= 8u && (output_channels & 7u) == 0u && spatial >= 8u &&
        ((height == 6u && input_channels == 512u && output_channels == 1024u) ||
         (height == 3u && ((input_channels == 768u && output_channels == 384u) ||
                           (input_channels == 1536u && output_channels == 768u))))) {
        return LW_X64_REC_NCHW_PW_FMA8;
    }
    return LW_X64_REC_NCHW_PW_FMA4;
}

static uint8_t choose_pointwise_kernel(uint32_t pixels, uint32_t output_channels) {
    if (pixels <= 1u && output_channels % 32u == 0u) return LW_X64_REC_PW_2X32;
    if (pixels <= 1u && output_channels % 16u == 0u) return LW_X64_REC_PW_4X16;
    return LW_X64_REC_PW_6X16;
}

static int fused_gelu_temporaries_private(const lw_model* model, uint32_t node_index,
                                           const lw_fused_gelu_match* match) {
    uint32_t temporary;
    if (model == NULL || match == NULL || node_index > model->info.node_count ||
        model->info.node_count - node_index < 5u) {
        return 0;
    }
    for (temporary = 0u; temporary < 4u; ++temporary) {
        uint32_t tensor = match->outputs[temporary];
        uint32_t node;
        int32_t last_use = -1;
        for (node = 0u; node < model->info.node_count; ++node) {
            const uint8_t* current = node_bytes(model, node);
            uint16_t input_count = lwm_read_u16(current + 2u);
            uint16_t input_index;
            for (input_index = 0u; input_index < input_count; ++input_index) {
                if (lwm_read_u32(current + 8u + (size_t)input_index * sizeof(uint32_t)) == tensor) {
                    last_use = (int32_t)node;
                }
            }
        }
        for (node = 0u; node < model->info.output_count; ++node) {
            if (lwm_read_u32(model->bytes + (size_t)model->output_offset +
                             (size_t)node * sizeof(uint32_t)) == tensor) {
                last_use = (int32_t)model->info.node_count;
            }
        }
        if (last_use != (int32_t)(node_index + temporary + 1u)) {
            return 0;
        }
    }
    return 1;
}

/* Count how many node input slots reference `tensor`; returns UINT32_MAX when
 * the tensor is also a model output (treat as externally consumed). */
static uint32_t tensor_use_count(const lw_model* model, uint32_t tensor) {
    uint32_t uses = 0u;
    uint32_t node;
    uint32_t output;
    for (node = 0u; node < model->info.node_count; ++node) {
        const uint8_t* current = node_bytes(model, node);
        uint16_t input_count = lwm_read_u16(current + 2u);
        uint16_t input_index;
        for (input_index = 0u; input_index < input_count; ++input_index) {
            if (lwm_read_u32(current + 8u + (size_t)input_index * sizeof(uint32_t)) == tensor) {
                ++uses;
            }
        }
    }
    for (output = 0u; output < model->info.output_count; ++output) {
        if (lwm_read_u32(model->bytes + (size_t)model->output_offset +
                         (size_t)output * sizeof(uint32_t)) == tensor) {
            return UINT32_MAX;
        }
    }
    return uses;
}

/* True when `tensor` is consumed exactly once, by node `consumer`. */
static int tensor_single_consumer(const lw_model* model, uint32_t tensor,
                                  uint32_t consumer) {
    const uint8_t* consumer_bytes;
    uint16_t input_count;
    uint16_t input_index;
    uint32_t uses_here = 0u;
    if (tensor_use_count(model, tensor) != 1u) {
        return 0;
    }
    if (consumer >= model->info.node_count) {
        return 0;
    }
    consumer_bytes = node_bytes(model, consumer);
    input_count = lwm_read_u16(consumer_bytes + 2u);
    for (input_index = 0u; input_index < input_count; ++input_index) {
        if (lwm_read_u32(consumer_bytes + 8u + (size_t)input_index * sizeof(uint32_t)) == tensor) {
            ++uses_here;
        }
    }
    return uses_here == 1u;
}

/* True when every consumer of `tensor` lies inside the node range
 * [first, first + span) and the tensor is not a model output. The GELU chain
 * reads its input twice (Div and the second Mul), so a plain single-consumer
 * test would reject it. */
static int tensor_consumed_within(const lw_model* model, uint32_t tensor,
                                  uint32_t first, uint32_t span) {
    uint32_t node;
    uint32_t output;
    int used = 0;
    for (node = 0u; node < model->info.node_count; ++node) {
        const uint8_t* current = node_bytes(model, node);
        uint16_t input_count = lwm_read_u16(current + 2u);
        uint16_t input_index;
        for (input_index = 0u; input_index < input_count; ++input_index) {
            if (lwm_read_u32(current + 8u + (size_t)input_index * sizeof(uint32_t)) == tensor) {
                if (node < first || node >= first + span) {
                    return 0;
                }
                used = 1;
            }
        }
    }
    for (output = 0u; output < model->info.output_count; ++output) {
        if (lwm_read_u32(model->bytes + (size_t)model->output_offset +
                         (size_t)output * sizeof(uint32_t)) == tensor) {
            return 0;
        }
    }
    return used;
}

/* Conv epilogue folding: after a successfully lowered NHWC conv at semantic
 * node `node_index`, greedily absorb a trailing channel-bias Add, a residual
 * Add, and a five-node GELU chain into the conv's fused epilogue. Returns the
 * number of semantic nodes consumed (1 = nothing folded). Folded operations
 * are applied by the conv kernels in exactly the original
 * accumulator -> +bias -> +residual -> activation order, so results stay
 * bit-identical to the unfused op sequence. On a successful fold this also
 * performs the lower_ops bookkeeping for the whole span: producer on the
 * final output and last_use bumps for every consumed input. */
static uint32_t fold_conv_epilogue(const lw_model* model, const lw_session* session,
                                   lw_x64_rec_program* program, uint32_t limit,
                                   uint32_t node_index, lw_x64_rec_op* op,
                                   uint32_t physical) {
    uint32_t current = lwm_read_u32(node_bytes(model, node_index) + 40u);
    uint64_t current_bytes = program->values[current].bytes;
    uint32_t span = 1u;
    uint32_t next = node_index + 1u;
    uint32_t residual_tensor = UINT32_MAX;
    uint32_t final_out = current;
    const float* post_bias = NULL;
    uint16_t activation = LW_NHWC_ACT_NONE;
    int allow_residual_gelu = op->kind != LW_X64_REC_OP_DEPTHWISE;
    if (op->kind == LW_X64_REC_OP_DENSE) return 1u;

    /* 1. Channel-bias Add: conv_out + constant[channels]. */
    if (next < limit) {
        const uint8_t* add = node_bytes(model, next);
        if (lwm_read_u16(add) == LW_OP_ADD && lwm_read_u16(add + 2u) == 2u) {
            uint32_t left = lwm_read_u32(add + 8u);
            uint32_t right = lwm_read_u32(add + 12u);
            uint32_t other = left == current ? right : right == current ? left : UINT32_MAX;
            uint32_t add_out = lwm_read_u32(add + 40u);
            if (other != UINT32_MAX &&
                program->values[other].constant_data != NULL &&
                program->values[other].bytes ==
                    (uint64_t)op->data.conv.output_channels * sizeof(float) &&
                program->values[add_out].bytes == current_bytes &&
                tensor_single_consumer(model, current, next)) {
                post_bias = program->values[other].constant_data;
                current = add_out;
                final_out = add_out;
                ++span;
                ++next;
            }
        }
    }
    /* 2. Residual Add: chain_out + non-constant tensor of the same size. */
    if (allow_residual_gelu && next < limit) {
        const uint8_t* add = node_bytes(model, next);
        if (lwm_read_u16(add) == LW_OP_ADD && lwm_read_u16(add + 2u) == 2u) {
            uint32_t left = lwm_read_u32(add + 8u);
            uint32_t right = lwm_read_u32(add + 12u);
            uint32_t other = left == current ? right : right == current ? left : UINT32_MAX;
            uint32_t add_out = lwm_read_u32(add + 40u);
            if (other != UINT32_MAX &&
                program->values[other].constant_data == NULL &&
                program->values[other].bytes == current_bytes &&
                program->values[add_out].bytes == current_bytes &&
                tensor_single_consumer(model, current, next)) {
                residual_tensor = other;
                current = add_out;
                final_out = add_out;
                ++span;
                ++next;
            }
        }
    }
    /* 3. Five-node GELU chain (Div -> Erf -> Add -> Mul -> Mul). */
    if (allow_residual_gelu && next + 5u <= limit) {
        lw_fused_gelu_match match;
        if (lw_match_fused_gelu(session, next, &match) &&
            match.inputs[0][0] == current &&
            fused_gelu_temporaries_private(model, next, &match) &&
            tensor_consumed_within(model, current, next, 5u)) {
            activation = LW_NHWC_ACT_GELU;
            final_out = match.outputs[4];
            span += 5u;
        }
    }
    if (span == 1u) {
        return 1u;
    }

    op->data.conv.post_bias = post_bias;
    if (residual_tensor != UINT32_MAX) {
        op->data.conv.has_residual = 1u;
        op->data.conv.residual_offset = program->values[residual_tensor].offset;
    }
    op->data.conv.activation = activation;
    op->data.conv.output_offset = program->values[final_out].offset;
    program->values[final_out].producer = (int32_t)physical;
    {
        uint32_t node;
        for (node = node_index; node < node_index + span; ++node) {
            const uint8_t* fused_bytes = node_bytes(model, node);
            uint16_t fused_inputs = lwm_read_u16(fused_bytes + 2u);
            uint16_t input_index;
            for (input_index = 0u; input_index < fused_inputs; ++input_index) {
                uint32_t input = lwm_read_u32(
                    fused_bytes + 8u + (size_t)input_index * sizeof(uint32_t));
                if (program->values[input].last_use < (int32_t)physical) {
                    program->values[input].last_use = (int32_t)physical;
                }
            }
        }
    }
    return span;
}

static lw_status compile_conv(const lw_model* model, const lw_session* session,
                              lw_x64_rec_program* program, const uint8_t* node,
                              lw_x64_rec_op* op, lw_error* error) {
    const lw_runtime_tensor* input = &session->tensors[lwm_read_u32(node + 8u)];
    const lw_runtime_tensor* weight = &session->tensors[lwm_read_u32(node + 12u)];
    const lw_runtime_tensor* output = &session->tensors[lwm_read_u32(node + 40u)];
    const uint8_t* params = model->bytes + (size_t)lwm_read_u64(node + 56u);
    uint32_t groups = lwm_read_u32(params + 4u);
    uint32_t kh = (uint32_t)lwm_read_i32(params + 8u);
    uint32_t kw = (uint32_t)lwm_read_i32(params + 12u);
    uint32_t sh = (uint32_t)lwm_read_i32(params + 16u);
    uint32_t sw = (uint32_t)lwm_read_i32(params + 20u);
    uint64_t count;
    float* packed;
    op->data.conv.input_channels = (uint32_t)input->dimensions[1];
    op->data.conv.input_offset = program->values[lwm_read_u32(node + 8u)].offset;
    op->data.conv.output_offset = program->values[lwm_read_u32(node + 40u)].offset;
    op->data.conv.output_channels = (uint32_t)output->dimensions[1];
    op->data.conv.input_height = (uint32_t)input->dimensions[2];
    op->data.conv.input_width = (uint32_t)input->dimensions[3];
    op->data.conv.output_height = (uint32_t)output->dimensions[2];
    op->data.conv.output_width = (uint32_t)output->dimensions[3];
    op->data.conv.kernel_h = kh; op->data.conv.kernel_w = kw;
    op->data.conv.weight_h = (uint32_t)weight->dimensions[2]; op->data.conv.weight_w = (uint32_t)weight->dimensions[3];
    op->data.conv.stride_h = sh; op->data.conv.stride_w = sw;
    op->data.conv.pad_top = (uint32_t)lwm_read_i32(params + 32u);
    op->data.conv.pad_left = (uint32_t)lwm_read_i32(params + 36u);
    op->data.conv.pad_bottom = (uint32_t)lwm_read_i32(params + 40u);
    op->data.conv.pad_right = (uint32_t)lwm_read_i32(params + 44u);
    op->data.conv.groups = groups;
    if (program->backend_layout == LW_X64_REC_BACKEND_NCHW) {
        const uint32_t weight_index = lwm_read_u32(node + 12u);
        const float* weights = constant_f32(model, weight_index);
        if (weights == NULL) {
            lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "NCHW Conv weights are not constant");
            return LW_STATUS_INVALID_ARGUMENT;
        }
        op->data.conv.original_weights = weights;
        if (lwm_read_u16(node + 2u) >= 3u) {
            op->data.conv.bias = constant_f32(model, lwm_read_u32(node + 16u));
        }
        if (groups == 1u && kh == 1u && kw == 1u && sh == 1u && sw == 1u) {
            if (!lw_packed_conv1x1_weight_count(op->data.conv.input_channels,
                                                op->data.conv.output_channels, &count)) {
                lw_set_error(error, LW_STATUS_UNSUPPORTED, "NCHW pointwise shape is unsupported");
                return LW_STATUS_UNSUPPORTED;
            }
            packed = (float*)alloc_constant(program, count * sizeof(float));
            if (packed == NULL) {
                lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "NCHW pointwise pack allocation failed");
                return LW_STATUS_OUT_OF_MEMORY;
            }
            lw_pack_conv1x1_weights_f32(weights, op->data.conv.input_channels,
                                        op->data.conv.output_channels, packed);
            op->kind = LW_X64_REC_OP_POINTWISE_NCHW;
            op->data.conv.kernel_kind = LW_X64_REC_CONV_POINTWISE;
            op->data.conv.packed_weights = packed;
            op->data.conv.nchw_pointwise_kernel =
                choose_nchw_pointwise_kernel(op->data.conv.input_height,
                                             op->data.conv.input_channels,
                                             op->data.conv.output_channels,
                                             op->data.conv.input_height *
                                                 op->data.conv.input_width);
            return LW_STATUS_OK;
        }
        if (groups == 1u && kh == 3u && kw == 3u && sh == 2u && sw == 2u &&
            op->data.conv.pad_top == 1u && op->data.conv.pad_left == 1u &&
            op->data.conv.pad_bottom == 1u && op->data.conv.pad_right == 1u &&
            lw_packed_conv3x3_stride2_weight_count(op->data.conv.input_channels,
                                                   op->data.conv.output_channels, &count)) {
            packed = (float*)alloc_constant(program, count * sizeof(float));
            if (packed == NULL) {
                lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "NCHW stem pack allocation failed");
                return LW_STATUS_OUT_OF_MEMORY;
            }
            lw_pack_conv3x3_stride2_weights_f32(weights, op->data.conv.input_channels,
                                                op->data.conv.output_channels, packed);
            op->kind = LW_X64_REC_OP_STEM_NCHW;
            op->data.conv.kernel_kind = LW_X64_REC_CONV_DENSE;
            op->data.conv.packed_weights = packed;
            return LW_STATUS_OK;
        }
        if (groups == op->data.conv.input_channels &&
            groups == op->data.conv.output_channels &&
            weight->dimensions[1] == 1u &&
            ((kh == 3u && kw == 3u &&
              op->data.conv.pad_top == 1u && op->data.conv.pad_left == 1u &&
              op->data.conv.pad_bottom == 1u && op->data.conv.pad_right == 1u &&
              ((sh == 1u && sw == 1u) || (sh == 2u && sw == 1u))) ||
             (kh == 1u && kw == 5u && sh == 1u && sw == 1u &&
              op->data.conv.pad_top == 0u && op->data.conv.pad_left == 2u &&
              op->data.conv.pad_bottom == 0u && op->data.conv.pad_right == 2u))) {
            op->kind = LW_X64_REC_OP_DEPTHWISE_NCHW;
            op->data.conv.kernel_kind = LW_X64_REC_CONV_DEPTHWISE;
            return LW_STATUS_OK;
        }
        char message[192]; (void)snprintf(message, sizeof(message), "Tiny NCHW Conv shape is unsupported: groups=%u in=%u out=%u kernel=%ux%u stride=%ux%u pad=%u,%u,%u,%u", groups, op->data.conv.input_channels, op->data.conv.output_channels, kh, kw, sh, sw, op->data.conv.pad_top, op->data.conv.pad_left, op->data.conv.pad_bottom, op->data.conv.pad_right); lw_set_error(error, LW_STATUS_UNSUPPORTED, message);
        return LW_STATUS_UNSUPPORTED;
    }
    if (groups == 1u && kh == 1u && kw == 1u && sh == 1u && sw == 1u &&
        lw_nhwc_dense_packed_weight_count(op->data.conv.input_channels,
                                          op->data.conv.output_channels, 1u, 1u, &count)) {
        op->kind = LW_X64_REC_OP_POINTWISE;
        op->data.conv.kernel_kind = LW_X64_REC_CONV_POINTWISE;
        op->data.conv.pointwise_kernel = choose_pointwise_kernel(
            op->data.conv.input_height * op->data.conv.input_width,
            op->data.conv.output_channels);
    } else if (groups == op->data.conv.input_channels && groups == op->data.conv.output_channels &&
               weight->dimensions[1] == 1 &&
               lw_nhwc_depthwise_packed_weight_count(op->data.conv.input_channels, kh, kw, &count)) {
        op->kind = LW_X64_REC_OP_DEPTHWISE;
        op->data.conv.kernel_kind = LW_X64_REC_CONV_DEPTHWISE;
    } else if (groups == 1u &&
               lw_nhwc_dense_packed_weight_count(op->data.conv.input_channels,
                                                 op->data.conv.output_channels, kh, kw, &count)) {
        op->kind = LW_X64_REC_OP_DENSE;
        op->data.conv.kernel_kind = LW_X64_REC_CONV_DENSE;
    } else {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "Tiny REC Conv shape is not NHWC-lowerable");
        return LW_STATUS_UNSUPPORTED;
    }
    op->data.conv.original_weights = constant_f32(model, lwm_read_u32(node + 12u));
    packed = (float*)borrow_conv_pack(program, op, count * sizeof(float));
    if (packed == NULL) {
        packed = (float*)alloc_constant(program, count * sizeof(float));
        if (packed == NULL) { lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "Conv packed weight allocation failed"); return LW_STATUS_OUT_OF_MEMORY; }
        if (op->data.conv.kernel_kind == LW_X64_REC_CONV_DEPTHWISE)
            lw_pack_nhwc_depthwise_f32(op->data.conv.original_weights, op->data.conv.input_channels, kh, kw, packed);
        else
            lw_pack_nhwc_dense_f32(op->data.conv.original_weights, op->data.conv.input_channels,
                                   op->data.conv.output_channels, kh, kw, packed);
    }
    op->data.conv.packed_weights = packed;
    if (lwm_read_u16(node + 2u) >= 3u) op->data.conv.bias = constant_f32(model, lwm_read_u32(node + 16u));
    op->data.conv.scratch_bytes = 0u;
    if (op->kind == LW_X64_REC_OP_DENSE) {
        op->data.conv.dense_kc = LW_NHWC_DENSE_KC;
        lw_nhwc_dense_desc desc;
        memset(&desc, 0, sizeof(desc));
        desc.batch = 1u; desc.input_channels = op->data.conv.input_channels;
        desc.input_height = op->data.conv.input_height; desc.input_width = op->data.conv.input_width;
        desc.output_channels = op->data.conv.output_channels; desc.output_height = op->data.conv.output_height;
        desc.output_width = op->data.conv.output_width; desc.kernel_h = kh; desc.kernel_w = kw;
        desc.stride_h = sh; desc.stride_w = sw; desc.pad_top = op->data.conv.pad_top; desc.pad_left = op->data.conv.pad_left;
        desc.pad_bottom = op->data.conv.pad_bottom; desc.pad_right = op->data.conv.pad_right;
        desc.dense_kc = op->data.conv.dense_kc;
        if (!lw_nhwc_dense_scratch_bytes(&desc, &op->data.conv.scratch_bytes)) {
            op->data.conv.scalar_fallback = 1u;
            op->data.conv.scratch_bytes = 0u;
        }
        if (op->data.conv.scratch_bytes > program->scratch_bytes) program->scratch_bytes = op->data.conv.scratch_bytes;
    }
    return LW_STATUS_OK;
}

static lw_status compile_node(const lw_model* model, const lw_session* session,
                              lw_x64_rec_program* program, uint32_t node_index,
                              lw_x64_rec_op* op, lw_error* error) {
    const uint8_t* node = node_bytes(model, node_index);
    const uint16_t semantic = lwm_read_u16(node);
    uint32_t input_index = lwm_read_u32(node + 8u);
    uint32_t output_index = lwm_read_u32(node + 40u);
    const lw_runtime_tensor* input = &session->tensors[input_index];
    const lw_runtime_tensor* output = &session->tensors[output_index];
    const lw_x64_rec_value* in_value = &program->values[input_index];
    lw_x64_rec_value* out_value = &program->values[output_index];
    const uint8_t* params = lwm_read_u64(node + 56u) == 0u ? NULL : model->bytes + (size_t)lwm_read_u64(node + 56u);
    uint64_t elements = value_elements(out_value);
    op->semantic_begin = node_index; op->semantic_count = 1u;
    switch (semantic) {
    case LW_OP_CONV: return compile_conv(model, session, program, node, op, error);
    case LW_OP_RELU: op->kind = LW_X64_REC_OP_RELU; break;
    case LW_OP_ERF: op->kind = LW_X64_REC_OP_ERF; break;
    case LW_OP_HARD_SIGMOID: op->kind = LW_X64_REC_OP_HARD_SIGMOID; op->data.unary.alpha = read_f32(params + 4u); op->data.unary.beta = read_f32(params + 8u); break;
    case LW_OP_SIGMOID: op->kind = LW_X64_REC_OP_SIGMOID; break;
    case LW_OP_SQRT: op->kind = LW_X64_REC_OP_SQRT; break;
    case LW_OP_BATCH_NORMALIZATION: {
        uint32_t c = (uint32_t)input->dimensions[1];
        const float* scale = constant_f32(model, lwm_read_u32(node + 12u));
        const float* bias = constant_f32(model, lwm_read_u32(node + 16u));
        const float* mean = constant_f32(model, lwm_read_u32(node + 20u));
        const float* variance = constant_f32(model, lwm_read_u32(node + 24u));
        float epsilon = read_f32(params + 4u);
        const lw_x64_rec_op* source = find_packed_source_op(program, node_index);
        float* mul;
        float* add;
        uint32_t j;
        if (source != NULL && source->kind == LW_X64_REC_OP_AFFINE &&
            source->data.affine.channels == c && source->data.affine.epsilon == epsilon &&
            source->data.affine.scale == scale && source->data.affine.bias == bias &&
            source->data.affine.mean == mean && source->data.affine.variance == variance &&
            source->data.affine.mul != NULL && source->data.affine.add != NULL) {
            mul = (float*)source->data.affine.mul;
            add = (float*)source->data.affine.add;
            note_borrow(program, (uint64_t)c * sizeof(float));
            note_borrow(program, (uint64_t)c * sizeof(float));
        } else {
            mul = (float*)alloc_constant(program, (uint64_t)c * sizeof(float));
            add = (float*)alloc_constant(program, (uint64_t)c * sizeof(float));
            if (mul == NULL || add == NULL) return LW_STATUS_OUT_OF_MEMORY;
            for (j = 0u; j < c; ++j) {
                float factor = scale[j] / sqrtf(variance[j] + epsilon);
                mul[j] = factor;
                add[j] = bias[j] - mean[j] * factor;
            }
        }
        op->kind = LW_X64_REC_OP_AFFINE;
        op->data.affine.input_offset = in_value->offset;
        op->data.affine.output_offset = out_value->offset;
        op->data.affine.mul = mul;
        op->data.affine.add = add;
        op->data.affine.scale = scale;
        op->data.affine.bias = bias;
        op->data.affine.mean = mean;
        op->data.affine.variance = variance;
        op->data.affine.epsilon = epsilon;
        op->data.affine.channel_major = (uint8_t)(program->backend_layout == LW_X64_REC_BACKEND_NCHW || input->rank != 4u);
        op->data.affine.pixels = (uint32_t)(value_elements(in_value) / c);
        op->data.affine.channels = c;
        break;
    }
    case LW_OP_ADD: case LW_OP_MUL: case LW_OP_DIV: case LW_OP_SUB: case LW_OP_POW:
        op->kind = semantic == LW_OP_ADD ? LW_X64_REC_OP_ADD : semantic == LW_OP_MUL ? LW_X64_REC_OP_MUL : semantic == LW_OP_DIV ? LW_X64_REC_OP_DIV : semantic == LW_OP_SUB ? LW_X64_REC_OP_SUB : semantic == LW_OP_POW ? LW_X64_REC_OP_POW : LW_X64_REC_OP_GENERIC_UNSUPPORTED; op->data.binary.operation=semantic; op->data.binary.element_count=elements; op->data.binary.output_offset=out_value->offset; op->data.binary.left_offset=program->values[input_index].offset; op->data.binary.left_constant=program->values[input_index].constant_data; op->data.binary.right_offset=program->values[lwm_read_u32(node+12u)].offset; op->data.binary.right_constant=program->values[lwm_read_u32(node+12u)].constant_data; op->data.binary.broadcast_kind=(program->values[lwm_read_u32(node+12u)].bytes==sizeof(float))?LW_X64_REC_BROADCAST_RIGHT_SCALAR:program->values[lwm_read_u32(node+12u)].bytes==out_value->bytes?LW_X64_REC_BROADCAST_SAME:((program->values[lwm_read_u32(node+12u)].bytes == (uint64_t)rec_binary_channel_extent(model, session, out_value, lwm_read_u32(node + 40u)) * sizeof(float))?LW_X64_REC_BROADCAST_RIGHT_CHANNEL:((program->values[input_index].bytes == (uint64_t)rec_binary_channel_extent(model, session, out_value, lwm_read_u32(node + 40u)) * sizeof(float))?LW_X64_REC_BROADCAST_LEFT_CHANNEL:LW_X64_REC_BROADCAST_GENERAL)); memcpy(op->data.binary.dimensions, out_value->dimensions, sizeof(op->data.binary.dimensions)); op->data.binary.channels = rec_binary_channel_extent(model, session, out_value, lwm_read_u32(node + 40u)); op->data.binary.pixels = op->data.binary.channels != 0u ? (uint32_t)(elements / op->data.binary.channels) : 0u;
        if (program->backend_layout == LW_X64_REC_BACKEND_NCHW && out_value->rank == 4u) {
            op->data.binary.channels = (uint32_t)out_value->dimensions[1];
            op->data.binary.pixels = op->data.binary.channels != 0u ?
                (uint32_t)(elements / op->data.binary.channels) : 0u;
            if (program->values[lwm_read_u32(node + 12u)].bytes ==
                (uint64_t)op->data.binary.channels * sizeof(float)) {
                op->data.binary.broadcast_kind = LW_X64_REC_BROADCAST_RIGHT_CHANNEL;
                op->data.binary.channel_major_nchw = 1u;
            } else if (program->values[input_index].bytes ==
                       (uint64_t)op->data.binary.channels * sizeof(float)) {
                op->data.binary.broadcast_kind = LW_X64_REC_BROADCAST_LEFT_CHANNEL;
                op->data.binary.channel_major_nchw = 1u;
            }
        }
        /* Shape-exact pixel-scalar broadcast ([1,W,C] op [1,W,1]).  Must be
         * evaluated before trusting the byte-based channel detection: when
         * W == channels the byte counts collide. */
        if (out_value->rank >= 2u) {
            /* One scalar per pixel broadcast over the last axis (e.g.
             * [1,W,C] op [1,W,1]). Byte counts alone are ambiguous when
             * pixels == channels, so compare the dimension tuples: same
             * rank, equal leading dims, trailing right dim of 1. The
             * general executor path only handles constant right
             * operands, so route computed ones here. */
            const lw_runtime_tensor* right_tensor =
                &session->tensors[lwm_read_u32(node + 12u)];
            uint32_t out_rank = out_value->rank;
            int pixel_scalar = right_tensor->rank == out_rank &&
                right_tensor->dimensions[out_rank - 1u] == 1 &&
                out_value->dimensions[out_rank - 1u] > 1;
            uint32_t axis;
            for (axis = 0u; axis + 1u < out_rank && pixel_scalar; ++axis) {
                if (right_tensor->dimensions[axis] != out_value->dimensions[axis]) {
                    pixel_scalar = 0;
                }
            }
            if (pixel_scalar) {
                op->data.binary.broadcast_kind = LW_X64_REC_BROADCAST_RIGHT_PIXEL_SCALAR;
            }
        }
        if (program->backend_layout != LW_X64_REC_BACKEND_NCHW && out_value->rank == 4u &&
            op->data.binary.broadcast_kind == LW_X64_REC_BROADCAST_SAME) {
            /* SAME assumes both operands share one physical layout.  The
             * attention chain holds canonical-order rank-4 tensors while
             * the conv chain is NHWC, so a cross-chain elementwise op
             * (e.g. positional-gate add) must remap the NHWC side. */
            const lw_runtime_tensor* out_tensor4 = &session->tensors[lwm_read_u32(node + 40u)];
            int left_nhwc = rec_value_is_nhwc_physical(model, session, input_index);
            int right_nhwc = rec_value_is_nhwc_physical(model, session, lwm_read_u32(node + 12u));
            if (left_nhwc != right_nhwc) {
                op->data.binary.mixed_nhwc = (uint8_t)(right_nhwc != 0 ? 1u : 2u);
                memcpy(op->data.binary.dimensions, out_tensor4->dimensions,
                       sizeof(op->data.binary.dimensions));
            }
        }
        break;
    case LW_OP_REDUCE_MEAN: {
        uint32_t axes_count = params != NULL ? (uint32_t)lwm_read_u16(params + 2u) : 0u;
        int32_t axis0 = axes_count > 0u ? lwm_read_i32(params + 12u) : 0;
        int32_t axis1 = axes_count > 1u ? lwm_read_i32(params + 16u) : 0;
        op->kind=LW_X64_REC_OP_REDUCE_MEAN; op->data.reduce.input_offset=in_value->offset; op->data.reduce.output_offset=out_value->offset; op->data.reduce.batch=(uint32_t)input->dimensions[0]; op->data.reduce.height=(uint32_t)input->dimensions[2]; op->data.reduce.width=(uint32_t)input->dimensions[3]; op->data.reduce.channels=(uint32_t)input->dimensions[1];
        if (input->rank == 4u && axes_count == 2u && axis0 == 2 && axis1 == 3) {
            op->data.reduce.general = 0u;
        } else {
            uint32_t axis_index;
            op->data.reduce.general = 1u;
            op->data.reduce.input_rank = input->rank;
            op->data.reduce.output_rank = output->rank;
            op->data.reduce.axes_count = axes_count;
            op->data.reduce.keep_dimensions = params != NULL ? lwm_read_u32(params + 4u) : 1u;
            op->data.reduce.no_op_with_empty_axes = params != NULL ? lwm_read_u32(params + 8u) : 0u;
            for (axis_index = 0u; axis_index < LW_MAX_DIMS; ++axis_index) {
                op->data.reduce.axes[axis_index] = axis_index < axes_count ?
                    lwm_read_i32(params + 12u + axis_index * 4u) : 0;
            }
            memcpy(op->data.reduce.input_dimensions, input->dimensions, sizeof(op->data.reduce.input_dimensions));
            memcpy(op->data.reduce.output_dimensions, output->dimensions, sizeof(op->data.reduce.output_dimensions));
        }
        break;
    }
    case LW_OP_AVERAGE_POOL: case LW_OP_MAX_POOL: op->kind=semantic==LW_OP_MAX_POOL?LW_X64_REC_OP_MAX_POOL:LW_X64_REC_OP_AVG_POOL; op->data.pool.input_offset=in_value->offset; op->data.pool.output_offset=out_value->offset; op->data.pool.input_dimensions[0]=input->dimensions[0];
        op->data.pool.input_dimensions[1]=program->backend_layout == LW_X64_REC_BACKEND_NCHW ? input->dimensions[1] : input->dimensions[2];
        op->data.pool.input_dimensions[2]=program->backend_layout == LW_X64_REC_BACKEND_NCHW ? input->dimensions[2] : input->dimensions[3];
        op->data.pool.input_dimensions[3]=program->backend_layout == LW_X64_REC_BACKEND_NCHW ? input->dimensions[3] : input->dimensions[1];
        op->data.pool.output_dimensions[0]=output->dimensions[0];
        op->data.pool.output_dimensions[1]=program->backend_layout == LW_X64_REC_BACKEND_NCHW ? output->dimensions[1] : output->dimensions[2];
        op->data.pool.output_dimensions[2]=program->backend_layout == LW_X64_REC_BACKEND_NCHW ? output->dimensions[2] : output->dimensions[3];
        op->data.pool.output_dimensions[3]=program->backend_layout == LW_X64_REC_BACKEND_NCHW ? output->dimensions[3] : output->dimensions[1]; op->data.pool.kernel[0]=lwm_read_i32(params+8u); op->data.pool.kernel[1]=lwm_read_i32(params+12u); op->data.pool.strides[0]=lwm_read_i32(params+16u); op->data.pool.strides[1]=lwm_read_i32(params+20u); op->data.pool.pads[0]=lwm_read_i32(params+24u); op->data.pool.pads[1]=lwm_read_i32(params+28u); op->data.pool.pads[2]=lwm_read_i32(params+32u); op->data.pool.pads[3]=lwm_read_i32(params+36u); op->data.pool.count_include_pad=(uint8_t)lwm_read_u32(params+44u); op->data.pool.is_max=(uint8_t)(semantic==LW_OP_MAX_POOL); break;
    case LW_OP_MATMUL: {
        const lw_runtime_tensor* w = &session->tensors[lwm_read_u32(node + 12u)];
        const lw_runtime_tensor* out_tensor = &session->tensors[lwm_read_u32(node + 40u)];
        uint64_t count = 0u;
        uint32_t batch = 1u;
        uint32_t bi;
        int packable;
        for (bi = 0u; bi + 2u < input->rank; ++bi) {
            batch *= (uint32_t)input->dimensions[bi];
        }
        op->kind = LW_X64_REC_OP_MATMUL;
        op->data.matmul.input_offset = in_value->offset;
        op->data.matmul.output_offset = out_value->offset;
        op->data.matmul.weights_offset = program->values[lwm_read_u32(node + 12u)].offset;
        op->data.matmul.weights = program->values[lwm_read_u32(node + 12u)].constant_data;
        op->data.matmul.batch = batch;
        op->data.matmul.rows = (uint32_t)input->dimensions[input->rank - 2u];
        op->data.matmul.inner = (uint32_t)input->dimensions[input->rank - 1u];
        op->data.matmul.columns = w->rank >= 2u ? (uint32_t)w->dimensions[w->rank - 1u] : 0u;
        op->data.matmul.general = 0u;
        op->data.matmul.input_rank = input->rank;
        op->data.matmul.weights_rank = w->rank;
        op->data.matmul.output_rank = out_tensor->rank;
        memcpy(op->data.matmul.input_dimensions, input->dimensions, sizeof(op->data.matmul.input_dimensions));
        memcpy(op->data.matmul.weights_dimensions, w->dimensions, sizeof(op->data.matmul.weights_dimensions));
        memcpy(op->data.matmul.output_dimensions, out_tensor->dimensions, sizeof(op->data.matmul.output_dimensions));
        /* Mirror the canonical session's packed-matmul eligibility exactly
         * so both paths accumulate in the same order (bit-identical). */
        packable = op->data.matmul.weights != NULL && w->rank == 2u &&
            out_tensor->rank == input->rank &&
            op->data.matmul.rows >= 4u && op->data.matmul.rows % 4u == 0u &&
            op->data.matmul.inner >= 64u && op->data.matmul.columns >= 1024u &&
            w->dimensions[0] == (int32_t)op->data.matmul.inner &&
            out_tensor->dimensions[out_tensor->rank - 2u] == (int32_t)op->data.matmul.rows &&
            out_tensor->dimensions[out_tensor->rank - 1u] == (int32_t)op->data.matmul.columns &&
            lw_packed_matmul_weight_count(op->data.matmul.inner, op->data.matmul.columns, &count);
        if (packable) {
            op->data.matmul.packed_weights = (float*)borrow_matmul_pack(program, op, count * sizeof(float));
            if (op->data.matmul.packed_weights == NULL) {
                op->data.matmul.packed_weights = (float*)alloc_constant(program, count * sizeof(float));
                if (op->data.matmul.packed_weights == NULL) return LW_STATUS_OUT_OF_MEMORY;
                lw_pack_matmul_weights_f32(op->data.matmul.weights, op->data.matmul.inner,
                                           op->data.matmul.columns, op->data.matmul.packed_weights);
            }
        } else {
            op->data.matmul.general = 1u;
        }
        break; }
    case LW_OP_TRANSPOSE: op->kind=LW_X64_REC_OP_TRANSPOSE; op->data.transpose.input_offset=in_value->offset; op->data.transpose.output_offset=out_value->offset; op->data.transpose.rank=input->rank; memcpy(op->data.transpose.input_dimensions,input->dimensions,sizeof(op->data.transpose.input_dimensions)); memcpy(op->data.transpose.output_dimensions,output->dimensions,sizeof(op->data.transpose.output_dimensions)); for(uint32_t j=0;j<input->rank && j<LW_MAX_DIMS;++j)op->data.transpose.permutation[j]=lwm_read_i32(params+4u+j*4u); break;
    case LW_OP_SOFTMAX: op->kind=LW_X64_REC_OP_SOFTMAX; op->data.softmax.input_offset=in_value->offset; op->data.softmax.output_offset=out_value->offset; op->data.softmax.axis=lwm_read_i32(params+4u); op->data.softmax.rank=input->rank; memcpy(op->data.softmax.dimensions,input->dimensions,sizeof(op->data.softmax.dimensions)); break;
    case LW_OP_CONCAT: {
        uint16_t concat_input_count = lwm_read_u16(node + 2u);
        int32_t concat_axis = params != NULL ? lwm_read_i32(params + 4u) : 0;
        uint16_t concat_slot;
        /* Only the channel axis (NCHW axis 1) of rank-4 feature maps is
         * supported: stored NHWC, it is a per-pixel channel copy. Other
         * shapes keep the model on the canonical executor. */
        if (params == NULL || concat_input_count < 2u ||
            concat_input_count > LWM_V0_MAX_NODE_INPUTS || output->rank != 4u ||
            concat_axis != 1 || program->backend_layout != LW_X64_REC_BACKEND_NHWC) {
            op->kind = LW_X64_REC_OP_GENERIC_UNSUPPORTED;
            break;
        }
        op->kind = LW_X64_REC_OP_CONCAT;
        op->data.concat.input_count = concat_input_count;
        op->data.concat.output_offset = out_value->offset;
        op->data.concat.axis = concat_axis;
        op->data.concat.output_channels = (uint32_t)output->dimensions[1];
        op->data.concat.pixels = (uint32_t)((uint64_t)output->dimensions[2] *
                                          (uint64_t)output->dimensions[3]);
        for (concat_slot = 0u; concat_slot < concat_input_count; ++concat_slot) {
            uint32_t concat_input = lwm_read_u32(node + 8u + (size_t)concat_slot * 4u);
            op->data.concat.input_offsets[concat_slot] = program->values[concat_input].offset;
            op->data.concat.input_channels[concat_slot] =
                (uint32_t)session->tensors[concat_input].dimensions[1];
        }
        break;
    }
    case LW_OP_SLICE: {
        uint32_t slice_count = params != NULL ? (uint32_t)lwm_read_u16(params + 2u) : 0u;
        uint32_t slice_index;
        /* Rank-4 values are physically NHWC here, so only slice
         * canonical-layout tensors (rank != 4) via the scalar kernel. */
        if (params == NULL || slice_count == 0u || slice_count > LW_MAX_DIMS ||
            input->rank > LW_MAX_DIMS || input->rank == 4u) {
            op->kind = LW_X64_REC_OP_GENERIC_UNSUPPORTED;
            break;
        }
        op->kind = LW_X64_REC_OP_SLICE;
        op->data.slice.input_offset = in_value->offset;
        op->data.slice.output_offset = out_value->offset;
        op->data.slice.rank = input->rank;
        op->data.slice.output_rank = output->rank;
        op->data.slice.slice_count = slice_count;
        memcpy(op->data.slice.input_dimensions, input->dimensions, sizeof(op->data.slice.input_dimensions));
        memcpy(op->data.slice.output_dimensions, output->dimensions, sizeof(op->data.slice.output_dimensions));
        for (slice_index = 0u; slice_index < LW_MAX_DIMS; ++slice_index) {
            op->data.slice.starts[slice_index] = slice_index < slice_count ? lwm_read_i32(params + 4u + slice_index * 4u) : 0;
            op->data.slice.ends[slice_index] = slice_index < slice_count ? lwm_read_i32(params + 36u + slice_index * 4u) : 0;
            op->data.slice.axes[slice_index] = slice_index < slice_count ? lwm_read_i32(params + 68u + slice_index * 4u) : 0;
            op->data.slice.steps[slice_index] = slice_index < slice_count ? lwm_read_i32(params + 100u + slice_index * 4u) : 0;
        }
        break;
    }
    case LW_OP_RESIZE: op->kind=LW_X64_REC_OP_GENERIC_UNSUPPORTED; break;
    default: op->kind=LW_X64_REC_OP_GENERIC_UNSUPPORTED; break;
    }
    if (op->kind == LW_X64_REC_OP_RELU || op->kind == LW_X64_REC_OP_ERF || op->kind == LW_X64_REC_OP_HARD_SIGMOID || op->kind == LW_X64_REC_OP_SIGMOID || op->kind == LW_X64_REC_OP_SQRT) { op->data.unary.input_offset=in_value->offset; op->data.unary.output_offset=out_value->offset; op->data.unary.element_count=elements; op->data.unary.rank=out_value->rank; memcpy(op->data.unary.dimensions,out_value->dimensions,sizeof(op->data.unary.dimensions)); }
    return LW_STATUS_OK;
}

static uint32_t resolve_alias_root(const lw_x64_rec_program* program, uint32_t index) {
    uint32_t guard = 0u;
    while (program->values[index].alias != 0u && guard++ < program->value_count) {
        index = program->values[index].alias_of;
    }
    return index;
}

typedef struct arena_candidate {
    uint32_t value_index;
    uint64_t bytes;
    int32_t first;
    int32_t last;
    uint64_t offset;
} arena_candidate;

static int candidate_before(const arena_candidate* a, const arena_candidate* b) {
    if (a->bytes != b->bytes) return a->bytes > b->bytes;
    if (a->first != b->first) return a->first < b->first;
    return a->value_index < b->value_index;
}

/* Offline greedy-by-size interval placement: each value gets a lifetime
 * [first,last] in physical-op space; values whose intervals overlap cannot
 * share an offset. Values never produced by a physical op (GELU fusion
 * temporaries, the CTC-excluded tail tensors, dead model tensors) get no slot,
 * which also removes their storage. */
static lw_status plan_arena_offsets(lw_x64_rec_program* program, lw_error* error) {
    arena_candidate* candidates;
    uint32_t count = 0u;
    uint32_t i;
    uint64_t arena_bytes = 0u;
    candidates = (arena_candidate*)calloc(program->value_count, sizeof(*candidates));
    if (candidates == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "REC arena plan allocation failed");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    if (program->ctc_fused) {
        /* The CTC tail reads the activation after the op loop; keep it live
         * until the end so no other value may overlap it. */
        lw_x64_rec_value* activation = &program->values[program->ctc.activation_value];
        if (activation->last_use < (int32_t)program->op_count) {
            activation->last_use = (int32_t)program->op_count;
        }
    }
    for (i = 0u; i < program->value_count; ++i) {
        const lw_x64_rec_value* value = &program->values[i];
        int32_t first;
        int32_t last;
        uint32_t a;
        if (value->alias != 0u || value->constant_data != NULL) continue;
        if ((int32_t)i == (int32_t)program->input_value) {
            /* The caller rewrites the input between runs, so it must never
             * share its offset with any value produced inside the program. */
            first = 0;
            last = (int32_t)program->op_count;
        } else if (value->producer >= 0) {
            first = value->producer;
            last = value->last_use > value->producer ? value->last_use : value->producer;
        } else {
            continue; /* never produced: fusion temporaries, CTC tail, dead tensors */
        }
        for (a = 0u; a < program->value_count; ++a) {
            const lw_x64_rec_value* alias = &program->values[a];
            if (alias->alias != 0u && resolve_alias_root(program, a) == i &&
                alias->last_use > last) {
                last = alias->last_use;
            }
        }
        candidates[count].value_index = i;
        candidates[count].bytes = value->bytes;
        candidates[count].first = first;
        candidates[count].last = last;
        ++count;
    }
    for (i = 1u; i < count; ++i) {
        arena_candidate current = candidates[i];
        uint32_t j = i;
        while (j > 0u && candidate_before(&current, &candidates[j - 1u])) {
            candidates[j] = candidates[j - 1u];
            --j;
        }
        candidates[j] = current;
    }
    for (i = 0u; i < count; ++i) {
        arena_candidate* current = &candidates[i];
        uint64_t offset = 0u;
        int collided = 1;
        while (collided) {
            uint32_t p;
            collided = 0;
            for (p = 0u; p < i; ++p) {
                const arena_candidate* placed = &candidates[p];
                if (placed->first > current->last || current->first > placed->last) continue;
                if (offset < placed->offset + placed->bytes &&
                    placed->offset < offset + current->bytes) {
                    offset = lw_x64_rec_arena_align(placed->offset + placed->bytes, 64u);
                    if (offset == UINT64_MAX) {
                        free(candidates);
                        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS,
                                     "REC arena plan overflows");
                        return LW_STATUS_OUT_OF_BOUNDS;
                    }
                    collided = 1;
                    break;
                }
            }
        }
        offset = lw_x64_rec_arena_align(offset, 64u);
        if (offset == UINT64_MAX || current->bytes > UINT64_MAX - offset) {
            free(candidates);
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "REC arena plan overflows");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        current->offset = offset;
        if (offset + current->bytes > arena_bytes) arena_bytes = offset + current->bytes;
        program->values[current->value_index].offset = offset;
    }
    for (i = 0u; i < program->value_count; ++i) {
        if (program->values[i].alias != 0u) {
            program->values[i].offset =
                program->values[resolve_alias_root(program, i)].offset;
        }
    }
    program->arena_bytes = arena_bytes;
    free(candidates);
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

/* Reset program state owned by a lowering pass so the second pass restarts
 * from the same zero counters and empty constant list. */
static void reset_lowering(lw_x64_rec_program* program, uint32_t limit) {
    uint32_t i;
    for (i = 0u; i < program->packed_constant_count; ++i) free(program->constants[i].data);
    free(program->constants);
    program->constants = NULL;
    program->packed_constant_count = 0u;
    program->owned_constant_bytes = 0u;
    program->borrowed_constant_bytes = 0u;
    program->borrowed_constant_count = 0u;
    memset(program->ops, 0, (size_t)limit * sizeof(*program->ops));
    program->op_count = 0u;
    program->semantic_consumed = 0u;
    program->semantic_elided = 0u;
    program->scratch_bytes = 0u;
    program->unsupported_nodes = 0u;
}

static lw_x64_rec_compile_result lower_ops(const lw_model* model, const lw_session* session,
                                           lw_x64_rec_program* program, uint32_t limit,
                                           lw_error* error);

#if defined(LW_EXPERIMENTAL_REC_FFN_TILING)
/* Discover pairs before placing tensors, then reproduce the same flags after
 * offsets are embedded. Scheduling the second op early extends the first
 * input's lifetime: the projected output must not reuse unread input pixels. */
static void mark_ffn_pairs(lw_x64_rec_program* program) {
    uint32_t i;
    if (program->backend_layout != LW_X64_REC_BACKEND_NHWC || !program->ctc_fused) return;
    for (i = 0u; i + 1u < program->op_count; ++i) {
        lw_x64_rec_op* first = &program->ops[i];
        const lw_x64_rec_op* second = &program->ops[i + 1u];
        const lw_x64_rec_conv_op* a = &first->data.conv;
        const lw_x64_rec_conv_op* b = &second->data.conv;
        uint32_t middle, input, output;
        int is_output = 0;
        if (first->kind != LW_X64_REC_OP_POINTWISE ||
            second->kind != LW_X64_REC_OP_POINTWISE ||
            a->scalar_fallback || b->scalar_fallback || a->has_residual ||
            a->activation != LW_NHWC_ACT_GELU || b->activation != LW_NHWC_ACT_NONE ||
            a->output_channels != b->input_channels ||
            a->input_height != b->input_height || a->input_width != b->input_width ||
            a->input_height != a->output_height || a->input_width != a->output_width ||
            b->input_height != b->output_height || b->input_width != b->output_width ||
            first->semantic_count == 0u) continue;
        middle = lwm_read_u32(node_bytes(program->model,
            first->semantic_begin + first->semantic_count - 1u) + 40u);
        if (middle != lwm_read_u32(node_bytes(program->model, second->semantic_begin) + 8u) ||
            !tensor_single_consumer(program->model, middle, second->semantic_begin) ||
            program->values[middle].alias ||
            program->values[middle].last_use != (int32_t)(i + 1u)) continue;
        for (output = 0u; output < program->model->info.output_count; ++output) {
            if (middle == lwm_read_u32(program->model->bytes +
                (size_t)program->model->output_offset + (size_t)output * sizeof(uint32_t))) {
                is_output = 1;
            }
        }
        if (is_output || middle == program->ctc.activation_value) continue;
        input = lwm_read_u32(node_bytes(program->model, first->semantic_begin) + 8u);
        if (program->values[input].last_use < (int32_t)(i + 1u))
            program->values[input].last_use = (int32_t)(i + 1u);
        first->flags |= LW_X64_REC_OP_FFN_BEGIN;
        ++i;
    }
}
#endif

static lw_x64_rec_compile_result rec_backend_compile_input_impl(const lw_model* model, uint32_t input_height,
                                                                 uint32_t target_width,
                                                                 lw_x64_rec_compile_strategy strategy,
                                                                 uint32_t allow_ctc_tail,
                                                                 const lw_x64_rec_program* packed_source,
                                                                 lw_x64_rec_program** out_program, lw_error* error) {
    lw_session* session=NULL; lw_x64_rec_program* program=NULL; lw_model_info info; lw_status status; uint32_t i, limit; lw_x64_rec_compile_result lowered;
    if(out_program==NULL||model==NULL||target_width==0u||target_width>INT32_MAX||input_height==0u||input_height>INT32_MAX){lw_set_error(error,LW_STATUS_INVALID_ARGUMENT,"REC backend arguments are invalid");return LW_X64_REC_COMPILE_INVALID_GRAPH;} *out_program=NULL;
    if (packed_source != NULL && (packed_source->model != model || packed_source->backend_layout != (uint8_t)strategy || packed_source->packed_source != NULL || packed_source->ops == NULL)) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "REC packed source is incompatible");
        return LW_X64_REC_COMPILE_INVALID_GRAPH;
    }
    lw_model_info_init(&info); status=lw_model_get_info(model,&info); if(status!=LW_STATUS_OK)return LW_X64_REC_COMPILE_INVALID_GRAPH; status=make_shape_session(model,input_height,target_width,&session,error); if(status!=LW_STATUS_OK)return status==LW_STATUS_OUT_OF_MEMORY?LW_X64_REC_COMPILE_OUT_OF_MEMORY:LW_X64_REC_COMPILE_INVALID_GRAPH; program=(lw_x64_rec_program*)calloc(1u,sizeof(*program)); if(program==NULL){lw_session_free(session);lw_set_error(error,LW_STATUS_OUT_OF_MEMORY,"REC program allocation failed");return LW_X64_REC_COMPILE_OUT_OF_MEMORY;} program->model=model;program->cpu=lw_get_cpu_capabilities();program->model_signature=info.content_checksum;program->target_width=target_width;program->backend_layout=(uint8_t)strategy;program->value_count=info.tensor_count;program->direct_nhwc=(uint8_t)(strategy == LW_X64_REC_COMPILE_NHWC);
    if (packed_source != NULL) { program->packed_source = packed_source; lw_x64_rec_program_retain((lw_x64_rec_program*)packed_source); }
    if(!rec_backend_cpu_supported(program->cpu)){lw_session_free(session);lw_x64_rec_program_free(program);lw_set_error(error,LW_STATUS_UNSUPPORTED,"REC compiled backend requires a supported SIMD target");return LW_X64_REC_COMPILE_UNSUPPORTED;} program->ctc_fused=(allow_ctc_tail!=0u&&detect_ctc(model,session,&program->ctc))?1u:0u; limit=program->ctc_fused?info.node_count-3u:info.node_count; program->ops=(lw_x64_rec_op*)calloc(limit,sizeof(*program->ops));program->values=(lw_x64_rec_value*)calloc(program->value_count,sizeof(*program->values));if(program->ops==NULL||program->values==NULL){lw_session_free(session);lw_x64_rec_program_free(program);lw_set_error(error,LW_STATUS_OUT_OF_MEMORY,"REC program tables allocation failed");return LW_X64_REC_COMPILE_OUT_OF_MEMORY;}
    for(i=0u;i<program->value_count;++i){fill_value(session,i,&program->values[i],strategy);if((session->tensors[i].flags&LWM_V0_TENSOR_FLAG_INPUT)!=0u)program->input_value=i;}
    lowered=lower_ops(model,session,program,limit,error);
    if(lowered!=LW_X64_REC_COMPILE_OK){lw_session_free(session);lw_x64_rec_program_free(program);return lowered;}
#if defined(LW_EXPERIMENTAL_REC_FFN_TILING)
    mark_ffn_pairs(program);
#endif
    status=plan_arena_offsets(program,error);
    if(status!=LW_STATUS_OK){lw_session_free(session);lw_x64_rec_program_free(program);return status==LW_STATUS_OUT_OF_MEMORY?LW_X64_REC_COMPILE_OUT_OF_MEMORY:LW_X64_REC_COMPILE_INVALID_GRAPH;}
    reset_lowering(program,limit);
    lowered=lower_ops(model,session,program,limit,error);
    if(lowered!=LW_X64_REC_COMPILE_OK){lw_session_free(session);lw_x64_rec_program_free(program);return lowered;}
#if defined(LW_EXPERIMENTAL_REC_FFN_TILING)
    mark_ffn_pairs(program);
#endif
    program->semantic_consumed+=program->semantic_elided; if(program->ctc_fused){
        uint64_t ctc_count = 0u;
        status=lw_x64_rec_ctc_prepare_shared(model,session,packed_source == NULL ? NULL : &packed_source->ctc,&program->ctc,error);
        if(status!=LW_STATUS_OK){lw_session_free(session);lw_x64_rec_program_free(program);return status==LW_STATUS_OUT_OF_MEMORY?LW_X64_REC_COMPILE_OUT_OF_MEMORY:LW_X64_REC_COMPILE_INVALID_GRAPH;}
        if (lw_packed_matmul_weight_count(program->ctc.inner, program->ctc.classes, &ctc_count)) {
            if (program->ctc.packed_weights_borrowed != 0u) note_borrow(program, ctc_count * sizeof(float));
            else program->owned_constant_bytes += ctc_count * sizeof(float);
        }
    }
    program->class_count=program->ctc.classes;program->time_steps=program->ctc.rows;program->output_value=info.output_count?lwm_read_u32(model->bytes+(size_t)model->output_offset):0u; if(program->arena_bytes==0u){lw_session_free(session);lw_x64_rec_program_free(program);lw_set_error(error,LW_STATUS_INVALID_SHAPE,"REC graph has no runtime arena");return LW_X64_REC_COMPILE_INVALID_GRAPH;} lw_session_free(session);lw_set_error(error,LW_STATUS_OK,"");*out_program=program;return LW_X64_REC_COMPILE_OK;
}

lw_x64_rec_compile_result lw_x64_rec_backend_compile_input(const lw_model* model, uint32_t input_height,
                                                          uint32_t target_width,
                                                          lw_x64_rec_compile_strategy strategy,
                                                          uint32_t allow_ctc_tail,
                                                          lw_x64_rec_program** out_program, lw_error* error) {
    return rec_backend_compile_input_impl(model, input_height, target_width, strategy,
                                          allow_ctc_tail, NULL, out_program, error);
}

/* Lower semantic nodes into the physical op table. Runs twice: the first pass
 * discovers value lifetimes (producer/last_use/fusions/aliases) before arena
 * offsets are planned; the second pass embeds the final offsets. */
/* Walk the semantic producer chain of a tensor to determine the physical
 * layout of its memory in the compiled program: conv-family ops produce
 * physical NHWC, layout-transparent elementwise/unary ops inherit from
 * their (non-constant) input, and everything else (transpose/matmul/
 * slice/reshape chains) produces canonical-order memory. */
static int rec_value_is_nhwc_physical(const lw_model* model, const lw_session* session,
                                      uint32_t tensor) {
    uint32_t current = tensor;
    uint32_t hops;
    for (hops = 0u; hops < 4096u; ++hops) {
        const uint8_t* node = NULL;
        uint16_t semantic;
        uint32_t j;
        for (j = 0u; j < model->info.node_count; ++j) {
            const uint8_t* n = node_bytes(model, j);
            if (lwm_read_u32(n + 40u) == current) { node = n; break; }
        }
        if (node == NULL) return 1; /* graph input: preprocessing emits NHWC */
        semantic = lwm_read_u16(node);
        switch (semantic) {
        case LW_OP_ADD: case LW_OP_MUL: case LW_OP_DIV: case LW_OP_SUB:
        case LW_OP_POW: case LW_OP_ERF: case LW_OP_HARD_SIGMOID:
        case LW_OP_RELU: case LW_OP_SIGMOID: case LW_OP_SQRT:
        case LW_OP_SOFTMAX: {
            /* layout-transparent: inherit from the non-constant input */
            uint32_t next = lwm_read_u32(node + 8u);
            if (lwm_read_u16(node + 2u) == 2u &&
                (session->tensors[next].flags & LWM_V0_TENSOR_FLAG_CONSTANT) != 0u) {
                next = lwm_read_u32(node + 12u);
            }
            current = next;
            break;
        }
        case LW_OP_CONV: case LW_OP_CONV_TRANSPOSE:
        case LW_OP_BATCH_NORMALIZATION: case LW_OP_REDUCE_MEAN:
        case LW_OP_AVERAGE_POOL: case LW_OP_MAX_POOL:
        case LW_OP_CONCAT: case LW_OP_RESIZE:
            return 1;
        default:
            return 0;
        }
    }
    return 0;
}

static lw_x64_rec_compile_result lower_ops(const lw_model* model, const lw_session* session,
                                           lw_x64_rec_program* program, uint32_t limit,
                                           lw_error* error) {
    uint32_t i;
    uint32_t physical = 0u;
    lw_status status = LW_STATUS_OK;
    for(i=0u;i<limit;++i){const uint8_t* node=node_bytes(model,i);uint16_t semantic=lwm_read_u16(node);
    {
        lw_fused_gelu_match match;
        if (i + 5u <= limit && lw_match_fused_gelu(session, i, &match) &&
            fused_gelu_temporaries_private(model, i, &match) && physical < limit) {
            lw_x64_rec_op* op = &program->ops[physical];
            const lw_x64_rec_value* source = &program->values[match.inputs[0][0]];
            const lw_x64_rec_value* output = &program->values[match.outputs[4]];
            memset(op, 0, sizeof(*op));
            op->kind = LW_X64_REC_OP_GELU;
            op->semantic_begin = i;
            op->semantic_count = 5u;
            op->data.unary.input_offset = source->offset;
            op->data.unary.output_offset = output->offset;
            op->data.unary.element_count = output->bytes / sizeof(float);
            op->data.unary.rank = output->rank;
            memcpy(op->data.unary.dimensions, output->dimensions,
                   sizeof(op->data.unary.dimensions));
            program->values[match.outputs[4]].producer = (int32_t)physical;
            for (uint32_t fused_node = i; fused_node < i + 5u; ++fused_node) {
                const uint8_t* fused_bytes = node_bytes(model, fused_node);
                uint16_t fused_inputs = lwm_read_u16(fused_bytes + 2u);
                for (uint16_t input_index = 0u; input_index < fused_inputs; ++input_index) {
                    uint32_t input = lwm_read_u32(
                        fused_bytes + 8u + (size_t)input_index * sizeof(uint32_t));
                    if (program->values[input].last_use < (int32_t)physical) {
                        program->values[input].last_use = (int32_t)physical;
                    }
                }
            }
            ++physical;
            program->semantic_consumed += 5u;
            i += 4u;
            continue;
        }
    }
    if (program->backend_layout == LW_X64_REC_BACKEND_NCHW &&
        (semantic == LW_OP_RESHAPE || semantic == LW_OP_SQUEEZE ||
         semantic == LW_OP_UNSQUEEZE)) {
        uint32_t in = lwm_read_u32(node + 8u);
        uint32_t out = lwm_read_u32(node + 40u);
        program->values[out].alias = 1u;
        program->values[out].alias_of = in;
        program->values[out].offset = program->values[in].offset;
        ++program->semantic_elided;
        program->values[out].producer = (int32_t)i;
        continue;
    }
    if(semantic==LW_OP_RESHAPE||semantic==LW_OP_SQUEEZE||semantic==LW_OP_UNSQUEEZE){uint32_t in=lwm_read_u32(node+8u),out=lwm_read_u32(node+40u); const lw_runtime_tensor* in_tensor=&session->tensors[in]; const lw_runtime_tensor* out_tensor=&session->tensors[out]; if(in_tensor->rank==3u && out_tensor->rank==4u && physical<limit){ lw_x64_rec_op* repack=&program->ops[physical++]; memset(repack,0,sizeof(*repack)); repack->kind=LW_X64_REC_OP_TRANSPOSE; repack->semantic_begin=i; repack->semantic_count=1u; repack->data.transpose.input_offset=program->values[in].offset; repack->data.transpose.output_offset=program->values[out].offset; repack->data.transpose.rank=4u; if(out_tensor->dimensions[1]==in_tensor->dimensions[1]){ repack->data.transpose.input_dimensions[0]=in_tensor->dimensions[0]; repack->data.transpose.input_dimensions[1]=in_tensor->dimensions[1]; repack->data.transpose.input_dimensions[2]=out_tensor->dimensions[2]; repack->data.transpose.input_dimensions[3]=out_tensor->dimensions[3]; repack->data.transpose.output_dimensions[0]=out_tensor->dimensions[0]; repack->data.transpose.output_dimensions[1]=out_tensor->dimensions[2]; repack->data.transpose.output_dimensions[2]=out_tensor->dimensions[3]; repack->data.transpose.output_dimensions[3]=out_tensor->dimensions[1]; repack->data.transpose.permutation[0]=0; repack->data.transpose.permutation[1]=2; repack->data.transpose.permutation[2]=3; repack->data.transpose.permutation[3]=1; }else{ /* Contiguous sequence-to-map repack whose permuted encoding is not  * self-consistent (e.g. [1,W,C] -> [1,1,W,C]); both layouts hold the  * same bytes, so emit the identity view and let the executor  * degenerate to a plain copy. */ repack->data.transpose.input_dimensions[0]=out_tensor->dimensions[0]; repack->data.transpose.input_dimensions[1]=out_tensor->dimensions[1]; repack->data.transpose.input_dimensions[2]=out_tensor->dimensions[2]; repack->data.transpose.input_dimensions[3]=out_tensor->dimensions[3]; repack->data.transpose.output_dimensions[0]=out_tensor->dimensions[0]; repack->data.transpose.output_dimensions[1]=out_tensor->dimensions[1]; repack->data.transpose.output_dimensions[2]=out_tensor->dimensions[2]; repack->data.transpose.output_dimensions[3]=out_tensor->dimensions[3]; repack->data.transpose.permutation[0]=0; repack->data.transpose.permutation[1]=1; repack->data.transpose.permutation[2]=2; repack->data.transpose.permutation[3]=3; } program->values[out].producer=(int32_t)(physical-1u); if(program->values[in].last_use<(int32_t)(physical-1u))program->values[in].last_use=(int32_t)(physical-1u); program->semantic_consumed++; continue; } if(in_tensor->rank==4u && out_tensor->rank==3u && !rec_value_is_nhwc_physical(model,session,in)){ /* Canonical-order rank-4 input (attention chain): the reshape is a pure reinterpretation, elide it. */ program->values[out].alias=1u;program->values[out].alias_of=in;program->values[out].offset=program->values[in].offset;++program->semantic_elided;program->values[out].producer=(int32_t)i;continue; } if(in_tensor->rank==4u && out_tensor->rank==3u && physical<limit){ lw_x64_rec_op* repack=&program->ops[physical++]; memset(repack,0,sizeof(*repack)); repack->kind=LW_X64_REC_OP_TRANSPOSE; repack->semantic_begin=i; repack->semantic_count=1u; repack->data.transpose.input_offset=program->values[in].offset; repack->data.transpose.output_offset=program->values[out].offset; repack->data.transpose.rank=4u; repack->data.transpose.input_dimensions[0]=in_tensor->dimensions[0]; repack->data.transpose.input_dimensions[1]=in_tensor->dimensions[2]; repack->data.transpose.input_dimensions[2]=in_tensor->dimensions[3]; repack->data.transpose.input_dimensions[3]=in_tensor->dimensions[1]; repack->data.transpose.output_dimensions[0]=out_tensor->dimensions[0]; repack->data.transpose.output_dimensions[1]=in_tensor->dimensions[1]; repack->data.transpose.output_dimensions[2]=in_tensor->dimensions[2]; repack->data.transpose.output_dimensions[3]=in_tensor->dimensions[3]; repack->data.transpose.permutation[0]=0; repack->data.transpose.permutation[1]=3; repack->data.transpose.permutation[2]=1; repack->data.transpose.permutation[3]=2; program->values[out].producer=(int32_t)(physical-1u); if(program->values[in].last_use<(int32_t)(physical-1u))program->values[in].last_use=(int32_t)(physical-1u); program->semantic_consumed++; continue; } program->values[out].alias=1u;program->values[out].alias_of=in;program->values[out].offset=program->values[in].offset;++program->semantic_elided;program->values[out].producer=(int32_t)i;continue;} if(physical>=limit)return LW_X64_REC_COMPILE_INVALID_GRAPH; {lw_x64_rec_op* op=&program->ops[physical];memset(op,0,sizeof(*op));status=compile_node(model,session,program,i,op,error);if(status!=LW_STATUS_OK||op->kind==LW_X64_REC_OP_GENERIC_UNSUPPORTED){++program->unsupported_nodes;if (status == LW_STATUS_OK || error == NULL || error->message[0] == 0) { char message[128]; (void)snprintf(message,sizeof(message),"REC node %u (op %u) lowering failed (status %d, kind %u)",(unsigned)i,(unsigned)semantic,(int)status,(unsigned)op->kind); lw_set_error(error,status!=LW_STATUS_OK?status:LW_STATUS_UNSUPPORTED,message); }return LW_X64_REC_COMPILE_UNSUPPORTED;}op->semantic_begin=i;op->semantic_count=1u;if(program->backend_layout==LW_X64_REC_BACKEND_NHWC&&op->data.conv.scalar_fallback==0u&&(op->kind==LW_X64_REC_OP_POINTWISE||op->kind==LW_X64_REC_OP_DENSE||op->kind==LW_X64_REC_OP_DEPTHWISE)){uint32_t fold_span=fold_conv_epilogue(model,session,program,limit,i,op,physical);if(fold_span>1u){op->semantic_count=(uint16_t)fold_span;program->semantic_consumed+=fold_span;++physical;i+=fold_span-1u;continue;}}program->semantic_consumed++;program->values[lwm_read_u32(node+40u)].producer=(int32_t)physical;for(uint32_t j=0u;j<lwm_read_u16(node+2u);++j){uint32_t in=lwm_read_u32(node+8u+j*4u);if(program->values[in].last_use<(int32_t)physical)program->values[in].last_use=(int32_t)physical;}++physical;}}
    program->op_count=physical;
    return LW_X64_REC_COMPILE_OK;
}

lw_x64_rec_compile_result lw_x64_rec_backend_compile_ex(
    const lw_model* model, uint32_t target_width,
    lw_x64_rec_compile_strategy strategy,
    lw_x64_rec_program** out_program, lw_error* error) {
#if defined(__EMSCRIPTEN__)
    if (strategy != LW_X64_REC_COMPILE_NHWC) {
        if (out_program != NULL) *out_program = NULL;
        lw_set_error(error, LW_STATUS_UNSUPPORTED,
                     "WASM compiled REC supports NHWC layout only");
        return LW_X64_REC_COMPILE_UNSUPPORTED;
    }
#endif
    return lw_x64_rec_backend_compile_input(model, 48u, target_width, strategy, 1u,
                                            out_program, error);
}

lw_x64_rec_compile_result lw_x64_rec_backend_compile(
    const lw_model* model, uint32_t target_width,
    lw_x64_rec_program** out_program, lw_error* error) {
    return lw_x64_rec_backend_compile_input(model, 48u, target_width,
                                            LW_X64_REC_COMPILE_NHWC, 1u, out_program, error);
}

lw_x64_rec_compile_result lw_x64_rec_backend_compile_shared(
    const lw_model* model, uint32_t target_width, const lw_x64_rec_program* packed_source,
    lw_x64_rec_program** out_program, lw_error* error) {
    if (packed_source == NULL) {
        if (out_program != NULL) *out_program = NULL;
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "REC packed source is null");
        return LW_X64_REC_COMPILE_INVALID_GRAPH;
    }
    return rec_backend_compile_input_impl(model, 48u, target_width,
                                          LW_X64_REC_COMPILE_NHWC, 1u, packed_source,
                                          out_program, error);
}

void lw_x64_rec_program_retain(lw_x64_rec_program* program) {
    uint32_t current;
    if (program == NULL) return;
    current = lw_atomic_u32_load_acquire(&program->shared_refs);
    while (current != UINT32_MAX &&
           !lw_atomic_u32_compare_exchange_acq_rel(&program->shared_refs, &current,
                                                    current + 1u)) {}
}

void lw_x64_rec_program_free(lw_x64_rec_program* program) {
    uint32_t current;
    uint32_t i;
    const lw_x64_rec_program* packed_source;
    if (program == NULL) return;
    current = lw_atomic_u32_load_acquire(&program->shared_refs);
    while (current != 0u) {
        if (lw_atomic_u32_compare_exchange_acq_rel(&program->shared_refs, &current,
                                                   current - 1u)) return;
    }
    packed_source = program->packed_source;
    lw_x64_rec_ctc_free(&program->ctc);
    for (i = 0u; i < program->packed_constant_count; ++i) free(program->constants[i].data);
    free(program->constants);
    free(program->ops);
    free(program->values);
    free(program);
    lw_x64_rec_program_free((lw_x64_rec_program*)packed_source);
}
