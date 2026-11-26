#include "x64_det_backend_internal.h"
#include "lwm_read.h"
#include "operator_internal.h"
#include "layout_planner_internal.h"
#include "../kernels/packed_conv_internal.h"

#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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
#if defined(__EMSCRIPTEN__) && defined(LW_WASM_COMPILED_DET)
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

static float read_f32(const uint8_t* p) {
    uint32_t bits = lwm_read_u32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint64_t value_elements(const lw_x64_det_value* value) { return value->bytes / sizeof(float); }

/* Effective layout of a tensor's primary slot: whatever layout its producer
 * node runs in. A direct-NHWC graph input is already in its arena layout. */
static uint8_t det_tensor_eff_layout(const lw_session* session,
                                     const lw_x64_det_program* program, uint32_t tensor_index) {
    int32_t birth;
    if (session == NULL || program == NULL || tensor_index >= program->value_count) {
        return LW_X64_DET_LAYOUT_NCHW;
    }
    if (program->direct_nhwc != 0u &&
        (session->tensors[tensor_index].flags & LWM_V0_TENSOR_FLAG_INPUT) != 0u) {
        return LW_X64_DET_LAYOUT_NHWC;
    }
    birth = session->tensors[tensor_index].birth_node;
    if (birth < 0 || (uint32_t)birth >= program->model->info.node_count) {
        return LW_X64_DET_LAYOUT_NCHW;
    }
    return program->effective_node_layout[birth];
}

static void fill_value(const lw_session* session, uint32_t i, lw_x64_det_value* value,
                       const lw_x64_det_program* program) {
    const lw_runtime_tensor* tensor = &session->tensors[i];
    uint32_t j;
    memset(value, 0, sizeof(*value));
    value->bytes = tensor->byte_size;
    value->rank = tensor->rank > 4u ? 4u : tensor->rank;
    value->layout = LW_X64_DET_LAYOUT_NCHW;
    if (tensor->rank == 4u && det_tensor_eff_layout(session, program, i) == LW_X64_DET_LAYOUT_NHWC) {
        /* Physical order NHWC: [N,H,W,C]. */
        value->dimensions[0] = tensor->dimensions[0];
        value->dimensions[1] = tensor->dimensions[2];
        value->dimensions[2] = tensor->dimensions[3];
        value->dimensions[3] = tensor->dimensions[1];
        value->layout = LW_X64_DET_LAYOUT_NHWC;
    } else {
        for (j = 0u; j < value->rank; ++j) value->dimensions[j] = tensor->dimensions[j];
    }
    value->producer = -1;
    value->last_use = -1;
    value->alt_producer = -1;
    value->alt_last_use = -1;
    if ((tensor->flags & LWM_V0_TENSOR_FLAG_CONSTANT) != 0u) {
        value->constant_data = constant_f32(session->model, i);
    }
}

static int add_constant(lw_x64_det_program* program, void* data, uint64_t bytes) {
    lw_x64_det_constant* next;
    if (data == NULL || bytes == 0u) return 0;
    next = (lw_x64_det_constant*)realloc(program->constants,
        (size_t)(program->packed_constant_count + 1u) * sizeof(*next));
    if (next == NULL) return 0;
    program->constants = next;
    program->constants[program->packed_constant_count].data = data;
    program->constants[program->packed_constant_count].bytes = bytes;
    ++program->packed_constant_count;
    return 1;
}

static void* alloc_constant(lw_x64_det_program* program, uint64_t bytes) {
    void* data;
    if (bytes == 0u || bytes > SIZE_MAX) return NULL;
    data = malloc((size_t)bytes);
    if (data == NULL || !add_constant(program, data, bytes)) {
        free(data);
        return NULL;
    }
    return data;
}

static uint8_t choose_nchw_pointwise_kernel(uint32_t height, uint32_t input_channels,
                                            uint32_t output_channels, uint32_t spatial) {
    if (output_channels >= 8u && (output_channels & 7u) == 0u && spatial >= 8u &&
        ((height == 6u && input_channels == 512u && output_channels == 1024u) ||
         (height == 3u && ((input_channels == 768u && output_channels == 384u) ||
                           (input_channels == 1536u && output_channels == 768u))))) {
        return LW_X64_DET_NCHW_PW_FMA8;
    }
    return LW_X64_DET_NCHW_PW_FMA4;
}

static uint8_t choose_pointwise_kernel(uint32_t pixels, uint32_t output_channels) {
    if (pixels <= 1u && output_channels % 32u == 0u) return LW_X64_DET_PW_2X32;
    if (pixels <= 1u && output_channels % 16u == 0u) return LW_X64_DET_PW_4X16;
    return LW_X64_DET_PW_6X16;
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

static lw_x64_det_op* push_op(lw_x64_det_program* program, uint32_t* count, uint16_t kind,
                              uint32_t semantic_begin, uint16_t semantic_count) {
    lw_x64_det_op* op = &program->ops[*count];
    memset(op, 0, sizeof(*op));
    op->kind = kind;
    op->semantic_begin = semantic_begin;
    op->semantic_count = semantic_count;
    ++*count;
    return op;
}

static int det_conv_is_depthwise(const lw_runtime_tensor* input, const lw_runtime_tensor* weight,
                                 uint32_t groups) {
    return groups == (uint32_t)input->dimensions[1] &&
           groups == (uint32_t)weight->dimensions[0] && weight->dimensions[1] == 1u;
}

static uint64_t det_slot_offset(lw_x64_det_program* program, uint32_t tensor_index,
                                uint8_t consumer_layout);

static lw_status compile_conv(const lw_model* model, const lw_session* session,
                              lw_x64_det_program* program, const uint8_t* node,
                              lw_x64_det_op* op, int materialize_constants, lw_error* error) {
    const lw_runtime_tensor* input = &session->tensors[lwm_read_u32(node + 8u)];
    const lw_runtime_tensor* weight = &session->tensors[lwm_read_u32(node + 12u)];
    const lw_runtime_tensor* output = &session->tensors[lwm_read_u32(node + 40u)];
    const uint8_t* params = model->bytes + (size_t)lwm_read_u64(node + 56u);
    uint32_t groups = lwm_read_u32(params + 4u);
    uint32_t kh = (uint32_t)lwm_read_i32(params + 8u);
    uint32_t kw = (uint32_t)lwm_read_i32(params + 12u);
    uint32_t sh = (uint32_t)lwm_read_i32(params + 16u);
    uint32_t sw = (uint32_t)lwm_read_i32(params + 20u);
    uint8_t node_layout = program->values[lwm_read_u32(node + 40u)].layout;
    uint64_t count;
    float* packed;
    op->data.conv.input_channels = (uint32_t)input->dimensions[1];
    op->data.conv.input_offset = det_slot_offset(program, lwm_read_u32(node + 8u), node_layout);
    op->data.conv.output_offset = program->values[lwm_read_u32(node + 40u)].offset;
    op->data.conv.output_channels = (uint32_t)output->dimensions[1];
    op->data.conv.input_height = (uint32_t)input->dimensions[2];
    op->data.conv.input_width = (uint32_t)input->dimensions[3];
    op->data.conv.output_height = (uint32_t)output->dimensions[2];
    op->data.conv.output_width = (uint32_t)output->dimensions[3];
    op->data.conv.kernel_h = kh; op->data.conv.kernel_w = kw;
    op->data.conv.weight_h = (uint32_t)weight->dimensions[2];
    op->data.conv.weight_w = (uint32_t)weight->dimensions[3];
    op->data.conv.stride_h = sh; op->data.conv.stride_w = sw;
    op->data.conv.pad_top = (uint32_t)lwm_read_i32(params + 32u);
    op->data.conv.pad_left = (uint32_t)lwm_read_i32(params + 36u);
    op->data.conv.pad_bottom = (uint32_t)lwm_read_i32(params + 40u);
    op->data.conv.pad_right = (uint32_t)lwm_read_i32(params + 44u);
    op->data.conv.groups = groups;
    if (node_layout == LW_X64_DET_LAYOUT_NCHW) {
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
        op->kind = LW_X64_DET_OP_CONV_NCHW;
        op->data.conv.nchw_conv_kernel = LW_X64_DET_NCHW_CONV_SCALAR;
        if (groups == 1u && kh == 1u && kw == 1u && sh == 1u && sw == 1u) {
            if (!lw_packed_conv1x1_weight_count(op->data.conv.input_channels,
                                                op->data.conv.output_channels, &count)) {
                lw_set_error(error, LW_STATUS_UNSUPPORTED, "NCHW pointwise shape is unsupported");
                return LW_STATUS_UNSUPPORTED;
            }
            if (materialize_constants) {
                packed = (float*)alloc_constant(program, count * sizeof(float));
                if (packed == NULL) {
                    lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "NCHW pointwise pack allocation failed");
                    return LW_STATUS_OUT_OF_MEMORY;
                }
                lw_pack_conv1x1_weights_f32(weights, op->data.conv.input_channels,
                                            op->data.conv.output_channels, packed);
                op->data.conv.packed_weights = packed;
            }
            op->data.conv.kernel_kind = LW_X64_DET_CONV_POINTWISE;
            op->data.conv.nchw_conv_kernel = LW_X64_DET_NCHW_CONV_POINTWISE_PACKED;
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
            op->data.conv.pad_bottom == 1u && op->data.conv.pad_right == 1u) {
            op->data.conv.kernel_kind = LW_X64_DET_CONV_DENSE;
            op->data.conv.nchw_conv_kernel = LW_X64_DET_NCHW_CONV_3X3S2;
            return LW_STATUS_OK;
        }
        if (groups == 1u && kh == 3u && kw == 3u && sh == 1u && sw == 1u &&
            op->data.conv.pad_top == 1u && op->data.conv.pad_left == 1u &&
            op->data.conv.pad_bottom == 1u && op->data.conv.pad_right == 1u) {
            op->data.conv.kernel_kind = LW_X64_DET_CONV_DENSE;
            op->data.conv.nchw_conv_kernel = LW_X64_DET_NCHW_CONV_3X3_UNIT;
            return LW_STATUS_OK;
        }
        if (groups == 1u && kh == 2u && kw == 2u && sh == 1u && sw == 1u &&
            op->data.conv.pad_top == 0u && op->data.conv.pad_left == 0u &&
            op->data.conv.pad_bottom == 1u && op->data.conv.pad_right == 1u) {
            op->data.conv.kernel_kind = LW_X64_DET_CONV_DENSE;
            op->data.conv.nchw_conv_kernel = LW_X64_DET_NCHW_CONV_2X2_PADEND1;
            return LW_STATUS_OK;
        }
        if (det_conv_is_depthwise(input, weight, groups) &&
            kh == 3u && kw == 3u &&
            op->data.conv.pad_top == 1u && op->data.conv.pad_left == 1u &&
            op->data.conv.pad_bottom == 1u && op->data.conv.pad_right == 1u &&
            ((sh == 1u && sw == 1u) || (sh == 2u && sw == 1u))) {
            op->data.conv.kernel_kind = LW_X64_DET_CONV_DEPTHWISE;
            op->data.conv.nchw_conv_kernel = sh == 2u
                ? LW_X64_DET_NCHW_CONV_DEPTHWISE3X3_S2X1
                : LW_X64_DET_NCHW_CONV_DEPTHWISE3X3;
            return LW_STATUS_OK;
        }
        if (det_conv_is_depthwise(input, weight, groups) &&
            kh == 5u && kw == 5u && sh == 1u && sw == 1u &&
            op->data.conv.pad_top == 2u && op->data.conv.pad_left == 2u &&
            op->data.conv.pad_bottom == 2u && op->data.conv.pad_right == 2u) {
            op->data.conv.kernel_kind = LW_X64_DET_CONV_DEPTHWISE;
            op->data.conv.nchw_conv_kernel = LW_X64_DET_NCHW_CONV_DEPTHWISE5X5;
            return LW_STATUS_OK;
        }
        /* Generic scalar fallback keeps every remaining shape correct. */
        op->data.conv.kernel_kind = det_conv_is_depthwise(input, weight, groups)
            ? LW_X64_DET_CONV_DEPTHWISE : LW_X64_DET_CONV_DENSE;
        return LW_STATUS_OK;
    }
    if (groups == 1u && kh == 1u && kw == 1u && sh == 1u && sw == 1u &&
        lw_nhwc_dense_packed_weight_count(op->data.conv.input_channels,
                                          op->data.conv.output_channels, 1u, 1u, &count)) {
        op->kind = LW_X64_DET_OP_POINTWISE;
        op->data.conv.kernel_kind = LW_X64_DET_CONV_POINTWISE;
        op->data.conv.pointwise_kernel = choose_pointwise_kernel(
            op->data.conv.input_height * op->data.conv.input_width,
            op->data.conv.output_channels);
    } else if (groups == op->data.conv.input_channels &&
               groups == op->data.conv.output_channels && weight->dimensions[1] == 1u &&
               lw_nhwc_depthwise_packed_weight_count(op->data.conv.input_channels, kh, kw,
                                                     &count)) {
        op->kind = LW_X64_DET_OP_DEPTHWISE;
        op->data.conv.kernel_kind = LW_X64_DET_CONV_DEPTHWISE;
    } else if (groups == 1u &&
               lw_nhwc_dense_packed_weight_count(op->data.conv.input_channels,
                                                 op->data.conv.output_channels, kh, kw, &count)) {
        op->kind = LW_X64_DET_OP_DENSE;
        op->data.conv.kernel_kind = LW_X64_DET_CONV_DENSE;
    } else {
        lw_set_error(error, LW_STATUS_UNSUPPORTED, "Tiny DET Conv shape is not NHWC-lowerable");
        return LW_STATUS_UNSUPPORTED;
    }
    op->data.conv.original_weights = constant_f32(model, lwm_read_u32(node + 12u));
    if (materialize_constants) {
        packed = (float*)alloc_constant(program, count * sizeof(float));
        if (packed == NULL) {
            lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "Conv packed weight allocation failed");
            return LW_STATUS_OUT_OF_MEMORY;
        }
        if (op->data.conv.kernel_kind == LW_X64_DET_CONV_DEPTHWISE) {
            lw_pack_nhwc_depthwise_f32(op->data.conv.original_weights,
                                       op->data.conv.input_channels, kh, kw, packed);
        } else {
            lw_pack_nhwc_dense_f32(op->data.conv.original_weights, op->data.conv.input_channels,
                                   op->data.conv.output_channels, kh, kw, packed);
        }
        op->data.conv.packed_weights = packed;
    }
    if (lwm_read_u16(node + 2u) >= 3u) {
        op->data.conv.bias = constant_f32(model, lwm_read_u32(node + 16u));
    }
    op->data.conv.scratch_bytes = 0u;
    if (op->kind == LW_X64_DET_OP_DENSE) {
        lw_nhwc_dense_desc desc;
        op->data.conv.dense_kc = LW_NHWC_DENSE_KC;
        memset(&desc, 0, sizeof(desc));
        desc.batch = 1u; desc.input_channels = op->data.conv.input_channels;
        desc.input_height = op->data.conv.input_height; desc.input_width = op->data.conv.input_width;
        desc.output_channels = op->data.conv.output_channels;
        desc.output_height = op->data.conv.output_height; desc.output_width = op->data.conv.output_width;
        desc.kernel_h = kh; desc.kernel_w = kw;
        desc.stride_h = sh; desc.stride_w = sw;
        desc.pad_top = op->data.conv.pad_top; desc.pad_left = op->data.conv.pad_left;
        desc.pad_bottom = op->data.conv.pad_bottom; desc.pad_right = op->data.conv.pad_right;
        desc.dense_kc = op->data.conv.dense_kc;
        if (!lw_nhwc_dense_scratch_bytes(&desc, &op->data.conv.scratch_bytes)) {
            op->data.conv.scalar_fallback = 1u;
            op->data.conv.scratch_bytes = 0u;
        }
        if (op->data.conv.scratch_bytes > program->scratch_bytes) {
            program->scratch_bytes = op->data.conv.scratch_bytes;
        }
    }
    return LW_STATUS_OK;
}

static uint8_t node_eff_layout(const lw_x64_det_program* program, uint32_t node_index) {
    return node_index < program->model->info.node_count
        ? program->effective_node_layout[node_index] : LW_X64_DET_LAYOUT_NCHW;
}

/* Offsets for one physical op input. The primary slot holds the producer's
 * layout; when the consumer runs in the other layout, a LAYOUT_CONVERT op is
 * materialized (once per value) and the consumer reads the alternate slot. */
static uint64_t det_slot_offset(lw_x64_det_program* program, uint32_t tensor_index,
                                uint8_t consumer_layout) {
    const lw_x64_det_value* value = &program->values[tensor_index];
    if (value->layout == consumer_layout || value->alt_producer < 0) return value->offset;
    return value->alt_offset;
}

static lw_status compile_node(const lw_model* model, const lw_session* session,
                              lw_x64_det_program* program, uint32_t node_index,
                              lw_x64_det_op* op, int materialize_constants, lw_error* error) {
    const uint8_t* node = node_bytes(model, node_index);
    const uint16_t semantic = lwm_read_u16(node);
    uint32_t input_index = lwm_read_u32(node + 8u);
    uint32_t output_index = lwm_read_u32(node + 40u);
    const lw_runtime_tensor* input = &session->tensors[input_index];
    const lw_runtime_tensor* output = &session->tensors[output_index];
    const lw_x64_det_value* out_value = &program->values[output_index];
    const uint8_t* params = lwm_read_u64(node + 56u) == 0u
        ? NULL : model->bytes + (size_t)lwm_read_u64(node + 56u);
    uint64_t elements = value_elements(out_value);
    uint8_t layout = out_value->layout;
    op->semantic_begin = node_index;
    op->semantic_count = 1u;
    switch (semantic) {
    case LW_OP_CONV:
        return compile_conv(model, session, program, node, op, materialize_constants, error);
    case LW_OP_RELU: op->kind = LW_X64_DET_OP_RELU; break;
    case LW_OP_ERF: op->kind = LW_X64_DET_OP_ERF; break;
    case LW_OP_HARD_SIGMOID:
        op->kind = LW_X64_DET_OP_HARD_SIGMOID;
        op->data.unary.alpha = read_f32(params + 4u);
        op->data.unary.beta = read_f32(params + 8u);
        break;
    case LW_OP_SIGMOID: op->kind = LW_X64_DET_OP_SIGMOID_NCHW; break;
    case LW_OP_BATCH_NORMALIZATION: {
        uint32_t c = (uint32_t)input->dimensions[1];
        op->kind = LW_X64_DET_OP_AFFINE;
        op->data.affine.input_offset = det_slot_offset(program, input_index, layout);
        op->data.affine.output_offset = out_value->offset;
        op->data.affine.scale = constant_f32(model, lwm_read_u32(node + 12u));
        op->data.affine.bias = constant_f32(model, lwm_read_u32(node + 16u));
        op->data.affine.mean = constant_f32(model, lwm_read_u32(node + 20u));
        op->data.affine.variance = constant_f32(model, lwm_read_u32(node + 24u));
        op->data.affine.epsilon = read_f32(params + 4u);
        op->data.affine.channel_major = (uint8_t)(layout == LW_X64_DET_LAYOUT_NCHW || input->rank != 4u);
        op->data.affine.pixels = (uint32_t)(value_elements(&program->values[input_index]) / c);
        op->data.affine.channels = c;
        if (materialize_constants) {
            float* mul = (float*)alloc_constant(program, (uint64_t)c * sizeof(float));
            float* add = (float*)alloc_constant(program, (uint64_t)c * sizeof(float));
            if (mul == NULL || add == NULL) return LW_STATUS_OUT_OF_MEMORY;
            for (uint32_t j = 0u; j < c; ++j) {
                float scale = op->data.affine.scale[j] /
                    sqrtf(op->data.affine.variance[j] + op->data.affine.epsilon);
                mul[j] = scale;
                add[j] = op->data.affine.bias[j] - op->data.affine.mean[j] * scale;
            }
            op->data.affine.mul = mul;
            op->data.affine.add = add;
        }
        break;
    }
    case LW_OP_ADD: case LW_OP_MUL: case LW_OP_DIV: case LW_OP_SUB: case LW_OP_POW:
        op->kind = semantic == LW_OP_ADD ? LW_X64_DET_OP_ADD
            : semantic == LW_OP_MUL ? LW_X64_DET_OP_MUL
            : semantic == LW_OP_DIV ? LW_X64_DET_OP_DIV
            : LW_X64_DET_OP_GENERIC_UNSUPPORTED;
        op->data.binary.operation = semantic;
        op->data.binary.element_count = elements;
        op->data.binary.layout = layout;
        op->data.binary.output_offset = out_value->offset;
        op->data.binary.left_offset = det_slot_offset(program, input_index, layout);
        op->data.binary.left_constant = program->values[input_index].constant_data;
        op->data.binary.right_offset = det_slot_offset(program, lwm_read_u32(node + 12u), layout);
        op->data.binary.right_constant = program->values[lwm_read_u32(node + 12u)].constant_data;
        op->data.binary.broadcast_kind =
            program->values[lwm_read_u32(node + 12u)].bytes == sizeof(float)
                ? LW_X64_DET_BROADCAST_RIGHT_SCALAR
            : program->values[lwm_read_u32(node + 12u)].bytes == out_value->bytes
                ? LW_X64_DET_BROADCAST_SAME
            : (program->values[lwm_read_u32(node + 12u)].bytes ==
                   (uint64_t)out_value->dimensions[3] * sizeof(float))
                ? LW_X64_DET_BROADCAST_RIGHT_CHANNEL
            : ((program->values[input_index].bytes ==
                   (uint64_t)out_value->dimensions[3] * sizeof(float))
                ? LW_X64_DET_BROADCAST_LEFT_CHANNEL
                : LW_X64_DET_BROADCAST_GENERAL);
        memcpy(op->data.binary.dimensions, out_value->dimensions,
               sizeof(op->data.binary.dimensions));
        op->data.binary.channels = out_value->rank > 0u
            ? (uint32_t)out_value->dimensions[out_value->rank - 1u] : 1u;
        op->data.binary.pixels = op->data.binary.channels != 0u
            ? (uint32_t)(elements / op->data.binary.channels) : 0u;
        if (layout == LW_X64_DET_LAYOUT_NCHW && out_value->rank == 4u) {
            op->data.binary.channels = (uint32_t)out_value->dimensions[1];
            op->data.binary.pixels = op->data.binary.channels != 0u
                ? (uint32_t)(elements / op->data.binary.channels) : 0u;
            if (program->values[lwm_read_u32(node + 12u)].bytes ==
                (uint64_t)op->data.binary.channels * sizeof(float)) {
                op->data.binary.broadcast_kind = LW_X64_DET_BROADCAST_RIGHT_CHANNEL;
            } else if (program->values[input_index].bytes ==
                       (uint64_t)op->data.binary.channels * sizeof(float)) {
                op->data.binary.broadcast_kind = LW_X64_DET_BROADCAST_LEFT_CHANNEL;
            }
        }
        break;
    case LW_OP_REDUCE_MEAN:
        op->kind = LW_X64_DET_OP_REDUCE_MEAN;
        op->data.reduce.input_offset = det_slot_offset(program, input_index, layout);
        op->data.reduce.output_offset = out_value->offset;
        op->data.reduce.batch = (uint32_t)input->dimensions[0];
        op->data.reduce.height = (uint32_t)input->dimensions[2];
        op->data.reduce.width = (uint32_t)input->dimensions[3];
        op->data.reduce.channels = (uint32_t)input->dimensions[1];
        op->data.reduce.layout = layout;
        break;
    case LW_OP_AVERAGE_POOL: case LW_OP_MAX_POOL:
        op->kind = semantic == LW_OP_MAX_POOL ? LW_X64_DET_OP_MAX_POOL : LW_X64_DET_OP_AVG_POOL;
        op->data.pool.input_offset = det_slot_offset(program, input_index, layout);
        op->data.pool.output_offset = out_value->offset;
        op->data.pool.input_dimensions[0] = input->dimensions[0];
        op->data.pool.input_dimensions[1] = layout == LW_X64_DET_LAYOUT_NCHW ? input->dimensions[1] : input->dimensions[2];
        op->data.pool.input_dimensions[2] = layout == LW_X64_DET_LAYOUT_NCHW ? input->dimensions[2] : input->dimensions[3];
        op->data.pool.input_dimensions[3] = layout == LW_X64_DET_LAYOUT_NCHW ? input->dimensions[3] : input->dimensions[1];
        op->data.pool.output_dimensions[0] = output->dimensions[0];
        op->data.pool.output_dimensions[1] = layout == LW_X64_DET_LAYOUT_NCHW ? output->dimensions[1] : output->dimensions[2];
        op->data.pool.output_dimensions[2] = layout == LW_X64_DET_LAYOUT_NCHW ? output->dimensions[2] : output->dimensions[3];
        op->data.pool.output_dimensions[3] = layout == LW_X64_DET_LAYOUT_NCHW ? output->dimensions[3] : output->dimensions[1];
        op->data.pool.kernel[0] = lwm_read_i32(params + 8u);
        op->data.pool.kernel[1] = lwm_read_i32(params + 12u);
        op->data.pool.strides[0] = lwm_read_i32(params + 16u);
        op->data.pool.strides[1] = lwm_read_i32(params + 20u);
        op->data.pool.pads[0] = lwm_read_i32(params + 24u);
        op->data.pool.pads[1] = lwm_read_i32(params + 28u);
        op->data.pool.pads[2] = lwm_read_i32(params + 32u);
        op->data.pool.pads[3] = lwm_read_i32(params + 36u);
        op->data.pool.count_include_pad = (uint8_t)lwm_read_u32(params + 44u);
        op->data.pool.is_max = (uint8_t)(semantic == LW_OP_MAX_POOL);
        op->data.pool.layout = layout;
        break;
    case LW_OP_CONV_TRANSPOSE: {
        /* Only the planner-capable 2x2/stride-2/no-pad shape (the
         * probability-map head) is lowerable. */
        if (lwm_read_u32(params + 4u) != 1u || lwm_read_i32(params + 8u) != 2 ||
            lwm_read_i32(params + 12u) != 2 || lwm_read_i32(params + 16u) != 2 ||
            lwm_read_i32(params + 20u) != 2 || lwm_read_i32(params + 24u) != 1 ||
            lwm_read_i32(params + 28u) != 1 || lwm_read_i32(params + 32u) != 0 ||
            lwm_read_i32(params + 36u) != 0 || lwm_read_i32(params + 40u) != 0 ||
            lwm_read_i32(params + 44u) != 0) {
            lw_set_error(error, LW_STATUS_UNSUPPORTED,
                         "DET ConvTranspose shape is not lowerable");
            return LW_STATUS_UNSUPPORTED;
        }
        op->data.conv_transpose.input_offset = det_slot_offset(program, input_index, layout);
        op->data.conv_transpose.output_offset = out_value->offset;
        op->data.conv_transpose.weights = constant_f32(model, lwm_read_u32(node + 12u));
        op->data.conv_transpose.bias = lwm_read_u16(node + 2u) >= 3u
            ? constant_f32(model, lwm_read_u32(node + 16u)) : NULL;
        op->data.conv_transpose.input_dimensions[0] = input->dimensions[0];
        op->data.conv_transpose.input_dimensions[1] = input->dimensions[1];
        op->data.conv_transpose.input_dimensions[2] = input->dimensions[2];
        op->data.conv_transpose.input_dimensions[3] = input->dimensions[3];
        op->data.conv_transpose.output_dimensions[0] = output->dimensions[0];
        op->data.conv_transpose.output_dimensions[1] = output->dimensions[1];
        op->data.conv_transpose.output_dimensions[2] = output->dimensions[2];
        op->data.conv_transpose.output_dimensions[3] = output->dimensions[3];
        op->data.conv_transpose.layout = layout;
        if (layout == LW_X64_DET_LAYOUT_NHWC) {
            uint32_t oc = (uint32_t)output->dimensions[1];
            if (oc != 1u && (oc & 7u) != 0u) {
                lw_set_error(error, LW_STATUS_UNSUPPORTED,
                             "DET ConvTranspose output channels are not NHWC-lowerable");
                return LW_STATUS_UNSUPPORTED;
            }
            if (oc != 1u) {
                uint64_t count = 0u;
                if (!lw_nhwc_convtranspose_packed_weight_count(
                        op->data.conv_transpose.input_dimensions[1], oc, &count)) {
                    lw_set_error(error, LW_STATUS_UNSUPPORTED,
                                 "DET ConvTranspose pack shape is unsupported");
                    return LW_STATUS_UNSUPPORTED;
                }
                if (materialize_constants) {
                    float* packed = (float*)alloc_constant(program, count * sizeof(float));
                    if (packed == NULL) {
                        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY,
                                     "DET ConvTranspose pack allocation failed");
                        return LW_STATUS_OUT_OF_MEMORY;
                    }
                    lw_pack_nhwc_convtranspose2x2_f32(
                        op->data.conv_transpose.weights,
                        (uint32_t)op->data.conv_transpose.input_dimensions[1], oc, packed);
                    op->data.conv_transpose.packed_weights = packed;
                }
                op->kind = LW_X64_DET_OP_CONV_TRANSPOSE_NHWC;
            } else {
                op->kind = LW_X64_DET_OP_CONV_TRANSPOSE_NHWC_C1;
            }
        } else {
            op->kind = LW_X64_DET_OP_CONV_TRANSPOSE_NCHW;
        }
        break;
    }
    case LW_OP_RESIZE: {
        op->kind = layout == LW_X64_DET_LAYOUT_NHWC
            ? LW_X64_DET_OP_RESIZE_NHWC : LW_X64_DET_OP_RESIZE_NCHW;
        op->data.resize.input_offset = det_slot_offset(program, input_index, layout);
        op->data.resize.output_offset = out_value->offset;
        op->data.resize.rank = input->rank > 4u ? 4u : input->rank;
        op->data.resize.input_dimensions[0] = input->dimensions[0];
        op->data.resize.input_dimensions[1] = input->dimensions[1];
        op->data.resize.input_dimensions[2] = input->dimensions[2];
        op->data.resize.input_dimensions[3] = input->dimensions[3];
        op->data.resize.output_dimensions[0] = output->dimensions[0];
        op->data.resize.output_dimensions[1] = output->dimensions[1];
        op->data.resize.output_dimensions[2] = output->dimensions[2];
        op->data.resize.output_dimensions[3] = output->dimensions[3];
        op->data.resize.scales[0] = read_f32(params + 4u);
        op->data.resize.scales[1] = read_f32(params + 8u);
        op->data.resize.scales[2] = read_f32(params + 12u);
        op->data.resize.scales[3] = read_f32(params + 16u);
        op->data.resize.layout = layout;
        break;
    }
    case LW_OP_CONCAT: {
        uint16_t input_count = lwm_read_u16(node + 2u);
        uint16_t slot;
        if (input_count > LWM_V0_MAX_NODE_INPUTS) {
            lw_set_error(error, LW_STATUS_UNSUPPORTED, "DET Concat has too many inputs");
            return LW_STATUS_UNSUPPORTED;
        }
        op->kind = layout == LW_X64_DET_LAYOUT_NHWC
            ? LW_X64_DET_OP_CONCAT_NHWC : LW_X64_DET_OP_CONCAT_NCHW;
        op->data.concat.input_count = input_count;
        op->data.concat.output_offset = out_value->offset;
        op->data.concat.axis = lwm_read_i32(params + 4u);
        op->data.concat.output_rank = output->rank > 4u ? 4u : output->rank;
        op->data.concat.layout = layout;
        for (slot = 0u; slot < input_count; ++slot) {
            const lw_runtime_tensor* concat_input =
                &session->tensors[lwm_read_u32(node + 8u + (size_t)slot * 4u)];
            op->data.concat.input_offsets[slot] =
                det_slot_offset(program, lwm_read_u32(node + 8u + (size_t)slot * 4u), layout);
            op->data.concat.input_ranks[slot] = concat_input->rank > 4u ? 4u : concat_input->rank;
            memcpy(op->data.concat.input_dimensions[slot], concat_input->dimensions,
                   sizeof(op->data.concat.input_dimensions[slot]));
        }
        memcpy(op->data.concat.output_dimensions, output->dimensions,
               sizeof(op->data.concat.output_dimensions));
        break;
    }
    default:
        op->kind = LW_X64_DET_OP_GENERIC_UNSUPPORTED;
        break;
    }
    if (op->kind == LW_X64_DET_OP_RELU || op->kind == LW_X64_DET_OP_ERF ||
        op->kind == LW_X64_DET_OP_HARD_SIGMOID || op->kind == LW_X64_DET_OP_SIGMOID_NCHW) {
        op->data.unary.input_offset = det_slot_offset(program, input_index, layout);
        op->data.unary.output_offset = out_value->offset;
        op->data.unary.element_count = elements;
        op->data.unary.rank = out_value->rank;
        op->data.unary.layout = layout;
        memcpy(op->data.unary.dimensions, out_value->dimensions,
               sizeof(op->data.unary.dimensions));
    }
    return LW_STATUS_OK;
}

static uint32_t resolve_alias_root(const lw_x64_det_program* program, uint32_t index) {
    uint32_t guard = 0u;
    while (program->values[index].alias != 0u && guard++ < program->value_count) {
        index = program->values[index].alias_of;
    }
    return index;
}

/* Lowerability: every planner-NHWC node except the island-exit set. Sigmoid
 * stays NCHW-scalar forever (canonical parity and neutral tensors make it
 * conversion-free); SQRT is not in the lowering switch. The planner's shape
 * rules gate the rest. */
static int det_node_nhwc_lowerable(uint16_t operation) {
    switch (operation) {
    case LW_OP_CONV:
    case LW_OP_ADD:
    case LW_OP_MUL:
    case LW_OP_DIV:
    case LW_OP_SUB:
    case LW_OP_POW:
    case LW_OP_RELU:
    case LW_OP_ERF:
    case LW_OP_HARD_SIGMOID:
    case LW_OP_BATCH_NORMALIZATION:
    case LW_OP_REDUCE_MEAN:
    case LW_OP_AVERAGE_POOL:
    case LW_OP_MAX_POOL:
    case LW_OP_RESIZE:
    case LW_OP_CONV_TRANSPOSE:
    case LW_OP_CONCAT:
        return 1;
    default:
        return 0;
    }
}

/* Cross the planner's NHWC classification with the v1 lowerability set and
 * force matched GELU patterns to a uniform layout. */
static lw_status plan_effective_layouts(const lw_model* model, const lw_session* session,
                                        const lw_layout_plan* plan,
                                        lw_x64_det_program* program, lw_error* error) {
    uint32_t node_index;
    program->effective_node_layout = (uint8_t*)calloc(model->info.node_count, sizeof(uint8_t));
    if (program->effective_node_layout == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "DET effective layout allocation failed");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    for (node_index = 0u; node_index < model->info.node_count; ++node_index) {
        const uint8_t* node = node_bytes(model, node_index);
        uint16_t operation = lwm_read_u16(node);
        if (plan->node_layout[node_index] == LW_FAST_LAYOUT_NHWC &&
            det_node_nhwc_lowerable(operation)) {
            program->effective_node_layout[node_index] = LW_X64_DET_LAYOUT_NHWC;
            ++program->nhwc_effective_nodes;
        } else {
            if (plan->node_layout[node_index] == LW_FAST_LAYOUT_NHWC &&
                operation != LW_OP_SIGMOID && operation != LW_OP_SQRT) {
                /* Planner/backend divergence: the planner classified an op
                 * NHWC that this backend cannot lower in any version. */
                char message[128];
                (void)snprintf(message, sizeof(message),
                               "DET node %" PRIu32 " (op %u) is planner-NHWC but not lowerable",
                               (unsigned)node_index, (unsigned)operation);
                lw_set_error(error, LW_STATUS_UNSUPPORTED, message);
                return LW_STATUS_UNSUPPORTED;
            }
            program->effective_node_layout[node_index] = LW_X64_DET_LAYOUT_NCHW;
            ++program->nchw_effective_nodes;
        }
    }
    for (node_index = 0u; node_index + 5u <= model->info.node_count; ++node_index) {
        lw_fused_gelu_match match;
        if (lw_match_fused_gelu(session, node_index, &match)) {
            uint8_t layout = program->effective_node_layout[node_index];
            uint32_t fused;
            for (fused = node_index + 1u; fused < node_index + 5u; ++fused) {
                program->effective_node_layout[fused] = layout;
            }
            node_index += 4u;
        }
    }
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

typedef struct det_arena_candidate {
    uint32_t value_index;
    uint64_t bytes;
    int32_t first;
    int32_t last;
    uint64_t offset;
    uint8_t is_alt;
} det_arena_candidate;

static int det_candidate_before(const det_arena_candidate* a, const det_arena_candidate* b) {
    if (a->bytes != b->bytes) return a->bytes > b->bytes;
    if (a->first != b->first) return a->first < b->first;
    if (a->value_index != b->value_index) return a->value_index < b->value_index;
    return a->is_alt < b->is_alt;
}

/* Copy of the proven REC greedy-by-size interval planner, extended with
 * alternate-layout slots: values read across an effective-layout boundary get
 * a second candidate whose lifetime spans its LAYOUT_CONVERT op to the last
 * alternate-layout consumer. */
static lw_status plan_arena_offsets(lw_x64_det_program* program, lw_error* error) {
    det_arena_candidate* candidates;
    uint32_t count = 0u;
    uint32_t i;
    uint64_t arena_bytes = 0u;
    candidates = (det_arena_candidate*)calloc(program->value_count * 2u, sizeof(*candidates));
    if (candidates == NULL) {
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "DET arena plan allocation failed");
        return LW_STATUS_OUT_OF_MEMORY;
    }
    for (i = 0u; i < program->value_count; ++i) {
        const lw_x64_det_value* value = &program->values[i];
        int32_t first;
        int32_t last;
        uint32_t a;
        if (value->alias != 0u || value->constant_data != NULL) continue;
        if ((int32_t)i == (int32_t)program->input_value) {
            first = 0;
            last = (int32_t)program->op_count;
        } else if (value->producer >= 0) {
            first = value->producer;
            last = value->last_use > value->producer ? value->last_use : value->producer;
        } else {
            continue;
        }
        for (a = 0u; a < program->value_count; ++a) {
            const lw_x64_det_value* alias = &program->values[a];
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
        if (value->alt_producer >= 0) {
            candidates[count].value_index = i;
            candidates[count].bytes = value->bytes;
            candidates[count].first = value->alt_producer;
            candidates[count].last = value->alt_last_use > value->alt_producer
                ? value->alt_last_use : value->alt_producer;
            candidates[count].is_alt = 1u;
            ++count;
        }
    }
    for (i = 1u; i < count; ++i) {
        det_arena_candidate current = candidates[i];
        uint32_t j = i;
        while (j > 0u && det_candidate_before(&current, &candidates[j - 1u])) {
            candidates[j] = candidates[j - 1u];
            --j;
        }
        candidates[j] = current;
    }
    for (i = 0u; i < count; ++i) {
        det_arena_candidate* current = &candidates[i];
        uint64_t offset = 0u;
        int collided = 1;
        while (collided) {
            uint32_t p;
            collided = 0;
            for (p = 0u; p < i; ++p) {
                const det_arena_candidate* placed = &candidates[p];
                if (placed->first > current->last || current->first > placed->last) continue;
                if (offset < placed->offset + placed->bytes &&
                    placed->offset < offset + current->bytes) {
                    offset = lw_x64_rec_arena_align(placed->offset + placed->bytes, 64u);
                    if (offset == UINT64_MAX) {
                        free(candidates);
                        lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS,
                                     "DET arena plan overflows");
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
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "DET arena plan overflows");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        current->offset = offset;
        if (offset + current->bytes > arena_bytes) arena_bytes = offset + current->bytes;
        if (current->is_alt) {
            program->values[current->value_index].alt_offset = offset;
        } else {
            program->values[current->value_index].offset = offset;
        }
    }
    for (i = 0u; i < program->value_count; ++i) {
        if (program->values[i].alias != 0u) {
            const lw_x64_det_value* root = &program->values[resolve_alias_root(program, i)];
            program->values[i].offset = root->offset;
            program->values[i].alt_offset = root->alt_offset;
        }
    }
    program->arena_bytes = arena_bytes;
    free(candidates);
    lw_set_error(error, LW_STATUS_OK, "");
    return LW_STATUS_OK;
}

static void reset_lowering(lw_x64_det_program* program, uint32_t limit) {
    uint32_t i;
    memset(program->ops, 0, (size_t)limit * sizeof(*program->ops));
    program->op_count = 0u;
    program->semantic_consumed = 0u;
    program->semantic_elided = 0u;
    program->scratch_bytes = 0u;
    program->unsupported_nodes = 0u;
    for (i = 0u; i < program->value_count; ++i) {
        program->values[i].producer = -1;
        program->values[i].last_use = -1;
        program->values[i].alt_producer = -1;
        program->values[i].alt_last_use = -1;
    }
}

static lw_x64_det_compile_result lower_ops(const lw_model* model, const lw_session* session,
                                           lw_x64_det_program* program, uint32_t limit,
                                           int materialize_constants, lw_error* error);

static uint32_t det_tensor_consumer_count(const lw_model* model, uint32_t tensor_index) {
    uint32_t count = 0u;
    uint32_t node;
    for (node = 0u; node < model->info.node_count; ++node) {
        const uint8_t* current = node_bytes(model, node);
        uint16_t input_count = lwm_read_u16(current + 2u);
        uint16_t input_index;
        for (input_index = 0u; input_index < input_count; ++input_index) {
            if (lwm_read_u32(current + 8u + (size_t)input_index * sizeof(uint32_t)) ==
                tensor_index) {
                ++count;
            }
        }
    }
    return count;
}

/* Every consumer of the tensor must be one of the `span` nodes starting at
 * `first`, and the tensor must not be a graph output. The GELU chain reads
 * its input twice, so a plain single-consumer test is too strict. */
static int det_tensor_consumed_within(const lw_model* model, uint32_t tensor, uint32_t first,
                                      uint32_t span) {
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

static int det_input_is_neutral(const lw_session* session, uint32_t tensor_index) {
    return tensor_index < session->model->info.tensor_count &&
           lw_layout_tensor_is_neutral(session, tensor_index);
}

/* Layout conversion only applies to rank-4 non-neutral tensors: neutral
 * tensors are layout-identical and other ranks are layout-agnostic. */
static int det_tensor_convertible(const lw_session* session, uint32_t tensor_index) {
    return tensor_index < session->model->info.tensor_count &&
           session->tensors[tensor_index].rank == 4u &&
           !det_input_is_neutral(session, tensor_index);
}

/* Materialize LAYOUT_CONVERT ops for every activation input of `node_index`
 * whose storage layout differs from the node's effective layout. Neutral
 * tensors ([N,1,H,W] or [N,C,1,1]) are layout-identical and skipped. */
static lw_status materialize_input_converts(const lw_model* model, const lw_session* session,
                                            lw_x64_det_program* program, uint32_t node_index,
                                            uint32_t* physical, uint32_t limit,
                                            lw_error* error) {
    const uint8_t* node = node_bytes(model, node_index);
    uint16_t input_count = lwm_read_u16(node + 2u);
    uint8_t node_layout = node_eff_layout(program, node_index);
    uint16_t input_slot;
    for (input_slot = 0u; input_slot < input_count; ++input_slot) {
        uint32_t tensor_index = lwm_read_u32(node + 8u + (size_t)input_slot * 4u);
        lw_x64_det_value* value;
        if (tensor_index >= program->value_count) continue;
        value = &program->values[tensor_index];
        if (value->alias != 0u) {
            value = &program->values[resolve_alias_root(program, tensor_index)];
        }
        if (value->constant_data != NULL || value->layout == node_layout ||
            !det_tensor_convertible(session, tensor_index)) {
            continue;
        }
        if (value->alt_producer >= 0) {
            if (value->alt_layout != node_layout) {
                lw_set_error(error, LW_STATUS_UNSUPPORTED,
                             "DET tensor requires three layouts");
                return LW_STATUS_UNSUPPORTED;
            }
            continue;
        }
        if (*physical >= limit) {
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "DET physical op table overflow");
            return LW_STATUS_OUT_OF_BOUNDS;
        }
        {
            const lw_runtime_tensor* tensor = &session->tensors[tensor_index];
            lw_x64_det_op* convert = push_op(program, physical, LW_X64_DET_OP_LAYOUT_CONVERT,
                                             node_index, 0u);
            convert->data.convert.input_offset = value->offset;
            convert->data.convert.output_offset = value->alt_offset;
            convert->data.convert.batch = (uint32_t)tensor->dimensions[0];
            convert->data.convert.channels = (uint32_t)tensor->dimensions[1];
            convert->data.convert.height = (uint32_t)tensor->dimensions[2];
            convert->data.convert.width = (uint32_t)tensor->dimensions[3];
            convert->data.convert.to_nhwc = (uint8_t)(node_layout == LW_X64_DET_LAYOUT_NHWC);
        }
        value->alt_producer = (int32_t)(*physical - 1u);
        value->alt_layout = node_layout;
        if (value->last_use < value->alt_producer) value->last_use = value->alt_producer;
    }
    return LW_STATUS_OK;
}

/* Update input read lifetimes for one semantic node whose physical op sits at
 * `physical`. Primary-slot reads extend last_use; alternate-slot reads extend
 * alt_last_use. */
static void update_input_uses(const lw_model* model, const lw_session* session,
                              lw_x64_det_program* program, uint32_t node_index,
                              int32_t physical) {
    const uint8_t* node = node_bytes(model, node_index);
    uint16_t input_count = lwm_read_u16(node + 2u);
    uint8_t node_layout = node_eff_layout(program, node_index);
    uint16_t input_slot;
    for (input_slot = 0u; input_slot < input_count; ++input_slot) {
        uint32_t tensor_index = lwm_read_u32(node + 8u + (size_t)input_slot * 4u);
        lw_x64_det_value* value;
        if (tensor_index >= program->value_count) continue;
        value = &program->values[tensor_index];
        if (value->alias != 0u) {
            value = &program->values[resolve_alias_root(program, tensor_index)];
        }
        if (value->constant_data != NULL) continue;
        if (value->layout != node_layout && value->alt_producer >= 0 &&
            det_tensor_convertible(session, tensor_index)) {
            if (value->alt_last_use < physical) value->alt_last_use = physical;
        } else if (value->last_use < physical) {
            value->last_use = physical;
        }
    }
}

static lw_x64_det_compile_result lower_ops(const lw_model* model, const lw_session* session,
                                           lw_x64_det_program* program, uint32_t limit,
                                           int materialize_constants, lw_error* error) {
    uint32_t i;
    uint32_t physical = 0u;
    lw_status status = LW_STATUS_OK;
    for (i = 0u; i < model->info.node_count; ++i) {
        const uint8_t* node = node_bytes(model, i);
        uint16_t semantic = lwm_read_u16(node);
        {
            lw_fused_gelu_match match;
            if (i + 5u <= model->info.node_count && lw_match_fused_gelu(session, i, &match) &&
                fused_gelu_temporaries_private(model, i, &match) && physical < limit) {
                uint8_t layout = node_eff_layout(program, i);
                uint32_t fused_node;
                lw_x64_det_op* op;
                const lw_x64_det_value* output = &program->values[match.outputs[4]];
                for (fused_node = i; fused_node < i + 5u; ++fused_node) {
                    status = materialize_input_converts(model, session, program, fused_node,
                                                        &physical, limit, error);
                    if (status != LW_STATUS_OK) return status == LW_STATUS_OUT_OF_MEMORY
                        ? LW_X64_DET_COMPILE_OUT_OF_MEMORY : LW_X64_DET_COMPILE_UNSUPPORTED;
                }
                if (physical >= limit) {
                    lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "DET physical op table overflow");
                    return LW_X64_DET_COMPILE_INVALID_GRAPH;
                }
                op = &program->ops[physical];
                memset(op, 0, sizeof(*op));
                op->kind = LW_X64_DET_OP_GELU;
                op->semantic_begin = i;
                op->semantic_count = 5u;
                op->data.unary.input_offset =
                    det_slot_offset(program, match.inputs[0][0], layout);
                op->data.unary.output_offset = output->offset;
                op->data.unary.element_count = output->bytes / sizeof(float);
                op->data.unary.rank = output->rank;
                op->data.unary.layout = layout;
                memcpy(op->data.unary.dimensions, output->dimensions,
                       sizeof(op->data.unary.dimensions));
                program->values[match.outputs[4]].producer = (int32_t)physical;
                for (fused_node = i; fused_node < i + 5u; ++fused_node) {
                    update_input_uses(model, session, program, fused_node, (int32_t)physical);
                }
                ++physical;
                program->semantic_consumed += 5u;
                i += 4u;
                continue;
            }
        }
        if (semantic == LW_OP_RESHAPE || semantic == LW_OP_SQUEEZE ||
            semantic == LW_OP_UNSQUEEZE) {
            uint32_t in = lwm_read_u32(node + 8u);
            uint32_t out = lwm_read_u32(node + 40u);
            program->values[out].alias = 1u;
            program->values[out].alias_of = in;
            program->values[out].offset = program->values[in].offset;
            ++program->semantic_elided;
            program->values[out].producer = (int32_t)i;
            continue;
        }
        status = materialize_input_converts(model, session, program, i, &physical, limit, error);
        if (status != LW_STATUS_OK) {
            return status == LW_STATUS_OUT_OF_MEMORY ? LW_X64_DET_COMPILE_OUT_OF_MEMORY
                                                     : LW_X64_DET_COMPILE_UNSUPPORTED;
        }
        if (physical >= limit) {
            lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "DET physical op table overflow");
            return LW_X64_DET_COMPILE_INVALID_GRAPH;
        }
        {
            lw_x64_det_op* op = &program->ops[physical];
            memset(op, 0, sizeof(*op));
            status = compile_node(model, session, program, i, op, materialize_constants, error);
            if (status != LW_STATUS_OK || op->kind == LW_X64_DET_OP_GENERIC_UNSUPPORTED) {
                ++program->unsupported_nodes;
                if (status == LW_STATUS_OK || error == NULL || error->message[0] == 0) {
                    char message[128];
                    (void)snprintf(message, sizeof(message),
                                   "DET node %u (op %u) lowering failed (status %d, kind %u)",
                                   (unsigned)i, (unsigned)semantic, (int)status,
                                   (unsigned)op->kind);
                    lw_set_error(error, status != LW_STATUS_OK ? status : LW_STATUS_UNSUPPORTED,
                                 message);
                }
                return LW_X64_DET_COMPILE_UNSUPPORTED;
            }
            program->semantic_consumed++;
            program->values[lwm_read_u32(node + 40u)].producer = (int32_t)physical;
            update_input_uses(model, session, program, i, (int32_t)physical);
            ++physical;
            /* Conv epilogue folding: greedily absorb a trailing channel-bias
             * Add, a residual Add, and a final Relu or five-node exact-GELU
             * chain into an NHWC pointwise/dense conv's fused epilogue. The
             * kernels apply accumulator -> +post_bias -> +residual ->
             * activation in exactly the original op order, so results stay
             * bit-identical to the unfused sequence. Elided intermediate
             * tensors never materialize (producer/last_use = -1). */
            if (semantic == LW_OP_CONV) {
                lw_x64_det_op* conv_op = &program->ops[physical - 1u];
                uint32_t conv_output = lwm_read_u32(node + 40u);
                if ((conv_op->kind == LW_X64_DET_OP_POINTWISE ||
                     conv_op->kind == LW_X64_DET_OP_DENSE) &&
                    conv_op->data.conv.activation == LW_NHWC_ACT_NONE &&
                    program->values[conv_output].layout == LW_X64_DET_LAYOUT_NHWC) {
                    uint32_t current = conv_output;
                    uint64_t current_bytes = program->values[current].bytes;
                    uint32_t cursor = i + 1u;
                    uint32_t consumed = 0u;
                    uint32_t residual_tensor = UINT32_MAX;
                    uint32_t final_out = current;
                    const float* post_bias = NULL;
                    uint16_t activation = LW_NHWC_ACT_NONE;
                    uint32_t intermediates[3];
                    uint32_t intermediate_count = 0u;
                    /* Channel-bias Add and residual Add, in either order, at
                     * most one of each. Both must be NHWC-effective so no
                     * layout convert would have sat between conv and Add. */
                    while (cursor < model->info.node_count &&
                           (post_bias == NULL || residual_tensor == UINT32_MAX)) {
                        const uint8_t* add = node_bytes(model, cursor);
                        uint32_t left;
                        uint32_t right;
                        uint32_t other;
                        uint32_t add_out;
                        uint32_t root;
                        lw_x64_det_value* other_value;
                        if (lwm_read_u16(add) != LW_OP_ADD ||
                            lwm_read_u16(add + 2u) != 2u ||
                            node_eff_layout(program, cursor) != LW_X64_DET_LAYOUT_NHWC ||
                            det_tensor_consumer_count(model, current) != 1u) {
                            break;
                        }
                        left = lwm_read_u32(add + 8u);
                        right = lwm_read_u32(add + 12u);
                        other = left == current ? right
                            : right == current ? left : UINT32_MAX;
                        if (other == UINT32_MAX || other == current) break;
                        add_out = lwm_read_u32(add + 40u);
                        if (add_out == current ||
                            program->values[add_out].bytes != current_bytes ||
                            program->values[add_out].alias != 0u) {
                            break;
                        }
                        root = resolve_alias_root(program, other);
                        other_value = &program->values[root];
                        if (other_value->constant_data != NULL) {
                            if (post_bias != NULL ||
                                other_value->bytes !=
                                    (uint64_t)conv_op->data.conv.output_channels *
                                        sizeof(float)) {
                                break;
                            }
                            post_bias = other_value->constant_data;
                        } else {
                            if (residual_tensor != UINT32_MAX ||
                                other_value->bytes != current_bytes ||
                                other_value->layout != LW_X64_DET_LAYOUT_NHWC) {
                                break;
                            }
                            residual_tensor = root;
                        }
                        intermediates[intermediate_count++] = current;
                        current = add_out;
                        final_out = add_out;
                        ++consumed;
                        ++cursor;
                    }
                    /* Relu or five-node exact-GELU chain after the adds. The
                     * GELU chain reads its input twice (Div and outer Mul),
                     * so require every consumer to live inside the chain. */
                    if (cursor < model->info.node_count) {
                        const uint8_t* next = node_bytes(model, cursor);
                        if (lwm_read_u16(next) == LW_OP_RELU &&
                            lwm_read_u32(next + 8u) == current &&
                            det_tensor_consumer_count(model, current) == 1u) {
                            intermediates[intermediate_count++] = current;
                            activation = LW_NHWC_ACT_RELU;
                            final_out = lwm_read_u32(next + 40u);
                            ++consumed;
                        } else {
                            lw_fused_gelu_match match;
                            if (cursor + 5u <= model->info.node_count &&
                                lw_match_fused_gelu(session, cursor, &match) &&
                                match.inputs[0][0] == current &&
                                fused_gelu_temporaries_private(model, cursor, &match) &&
                                det_tensor_consumed_within(model, current, cursor, 5u)) {
                                intermediates[intermediate_count++] = current;
                                activation = LW_NHWC_ACT_GELU;
                                final_out = match.outputs[4];
                                consumed += 5u;
                            }
                        }
                    }
                    if (consumed > 0u) {
                        uint32_t k;
                        conv_op->data.conv.post_bias = post_bias;
                        if (residual_tensor != UINT32_MAX) {
                            conv_op->data.conv.has_residual = 1u;
                            conv_op->data.conv.residual_offset =
                                det_slot_offset(program, residual_tensor,
                                                LW_X64_DET_LAYOUT_NHWC);
                            if (program->values[residual_tensor].last_use <
                                (int32_t)(physical - 1u)) {
                                program->values[residual_tensor].last_use =
                                    (int32_t)(physical - 1u);
                            }
                        }
                        conv_op->data.conv.activation = activation;
                        conv_op->data.conv.output_offset =
                            program->values[final_out].offset;
                        conv_op->semantic_count = (uint16_t)(1u + consumed);
                        for (k = 0u; k < intermediate_count; ++k) {
                            program->values[intermediates[k]].producer = -1;
                            program->values[intermediates[k]].last_use = -1;
                        }
                        program->values[final_out].producer = (int32_t)(physical - 1u);
                        program->semantic_elided += consumed;
                        i += consumed;
                        continue;
                    }
                }
            }
        }
    }
    /* Graph-output tensors are read by the caller in NCHW; if a producer left
     * one in NHWC (possible once Phase 2 makes the tail NHWC), materialize a
     * trailing convert. Neutral outputs are layout-identical and skipped. */
    {
        uint32_t t;
        for (t = 0u; t < program->value_count; ++t) {
            lw_x64_det_value* value = &program->values[t];
            int32_t birth;
            if ((session->tensors[t].flags & LWM_V0_TENSOR_FLAG_OUTPUT) == 0u ||
                det_input_is_neutral(session, t)) {
                continue;
            }
            birth = session->tensors[t].birth_node;
            if (value->layout != LW_X64_DET_LAYOUT_NHWC || birth < 0 ||
                node_eff_layout(program, (uint32_t)birth) != LW_X64_DET_LAYOUT_NHWC) {
                continue;
            }
            if (physical >= limit) {
                lw_set_error(error, LW_STATUS_OUT_OF_BOUNDS, "DET physical op table overflow");
                return LW_X64_DET_COMPILE_INVALID_GRAPH;
            }
            {
                const lw_runtime_tensor* tensor = &session->tensors[t];
                lw_x64_det_op* convert = push_op(program, &physical,
                                                 LW_X64_DET_OP_LAYOUT_CONVERT,
                                                 (uint32_t)birth, 0u);
                convert->data.convert.input_offset = value->offset;
                convert->data.convert.output_offset = value->alt_offset;
                convert->data.convert.batch = (uint32_t)tensor->dimensions[0];
                convert->data.convert.channels = (uint32_t)tensor->dimensions[1];
                convert->data.convert.height = (uint32_t)tensor->dimensions[2];
                convert->data.convert.width = (uint32_t)tensor->dimensions[3];
                convert->data.convert.to_nhwc = 0u;
            }
            value->alt_producer = (int32_t)(physical - 1u);
            value->alt_layout = LW_X64_DET_LAYOUT_NCHW;
            if (value->last_use < value->alt_producer) value->last_use = value->alt_producer;
        }
    }
    program->op_count = physical;
    return LW_X64_DET_COMPILE_OK;
}

lw_x64_det_compile_result lw_x64_det_backend_compile_ex(
    const lw_model* model, uint32_t height, uint32_t width,
    lw_x64_det_compile_strategy strategy, lw_x64_det_program** out_program, lw_error* error) {
    lw_session* session = NULL;
    lw_x64_det_program* program = NULL;
    lw_model_info info;
    lw_status status;
    uint32_t i;
    uint32_t capacity;
    lw_x64_det_compile_result lowered;
    const int direct_input = strategy == LW_X64_DET_COMPILE_NHWC_DIRECT;
    if (direct_input) strategy = LW_X64_DET_COMPILE_NHWC;
    if (out_program == NULL || model == NULL || height == 0u || width == 0u ||
        height > INT32_MAX || width > INT32_MAX) {
        lw_set_error(error, LW_STATUS_INVALID_ARGUMENT, "DET backend arguments are invalid");
        return LW_X64_DET_COMPILE_INVALID_GRAPH;
    }
    *out_program = NULL;
    lw_model_info_init(&info);
    status = lw_model_get_info(model, &info);
    if (status != LW_STATUS_OK) return LW_X64_DET_COMPILE_INVALID_GRAPH;
    status = make_shape_session(model, height, width, &session, error);
    if (status != LW_STATUS_OK) {
        return status == LW_STATUS_OUT_OF_MEMORY ? LW_X64_DET_COMPILE_OUT_OF_MEMORY
                                                 : LW_X64_DET_COMPILE_INVALID_GRAPH;
    }
    program = (lw_x64_det_program*)calloc(1u, sizeof(*program));
    if (program == NULL) {
        lw_session_free(session);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "DET program allocation failed");
        return LW_X64_DET_COMPILE_OUT_OF_MEMORY;
    }
    program->model = model;
    program->cpu = lw_get_cpu_capabilities();
    program->model_signature = info.content_checksum;
    program->input_height = height;
    program->input_width = width;
    program->backend_layout = (uint8_t)strategy;
    program->value_count = info.tensor_count;
#if defined(__EMSCRIPTEN__) && defined(__wasm_simd128__)
    if (strategy != LW_X64_DET_COMPILE_NHWC) {
        lw_session_free(session);
        lw_x64_det_program_free(program);
        lw_set_error(error, LW_STATUS_UNSUPPORTED,
                     "WASM compiled DET supports NHWC layout only");
        return LW_X64_DET_COMPILE_UNSUPPORTED;
    }
#else
    if (!lw_simd_level_is_avx2(program->cpu.simd) || !program->cpu.has_avx2_fma) {
        lw_session_free(session);
        lw_x64_det_program_free(program);
        lw_set_error(error, LW_STATUS_UNSUPPORTED,
                     "standalone x64 DET backend requires AVX2 and FMA");
        return LW_X64_DET_COMPILE_UNSUPPORTED;
    }
#endif
    program->effective_node_layout = (uint8_t*)calloc(info.node_count, sizeof(uint8_t));
    if (program->effective_node_layout == NULL) {
        lw_session_free(session);
        lw_x64_det_program_free(program);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "DET effective layout allocation failed");
        return LW_X64_DET_COMPILE_OUT_OF_MEMORY;
    }
    if (strategy == LW_X64_DET_COMPILE_NHWC) {
        lw_layout_planner_options options;
        lw_layout_plan plan;
        memset(&options, 0, sizeof(options));
        options.allow_direct_nhwc_graph_input = 1u;
        memset(&plan, 0, sizeof(plan));
        status = lw_layout_plan_build(session, &options, &plan, error);
        if (status != LW_STATUS_OK) {
            lw_layout_plan_free(&plan);
            lw_session_free(session);
            lw_x64_det_program_free(program);
            return LW_X64_DET_COMPILE_INVALID_GRAPH;
        }
        program->direct_nhwc = direct_input && plan.graph_input_direct_nhwc;
        status = plan_effective_layouts(model, session, &plan, program, error);
        lw_layout_plan_free(&plan);
        if (status != LW_STATUS_OK) {
            lw_session_free(session);
            lw_x64_det_program_free(program);
            return status == LW_STATUS_OUT_OF_MEMORY ? LW_X64_DET_COMPILE_OUT_OF_MEMORY
                                                     : LW_X64_DET_COMPILE_UNSUPPORTED;
        }
    } else {
        program->nchw_effective_nodes = info.node_count;
    }
    /* Physical ops: one per semantic node at most, plus one shared convert per
     * value at most. */
    capacity = info.node_count + info.tensor_count + 16u;
    program->ops = (lw_x64_det_op*)calloc(capacity, sizeof(*program->ops));
    program->values = (lw_x64_det_value*)calloc(program->value_count,
                                                sizeof(*program->values));
    if (program->ops == NULL || program->values == NULL) {
        lw_session_free(session);
        lw_x64_det_program_free(program);
        lw_set_error(error, LW_STATUS_OUT_OF_MEMORY, "DET program tables allocation failed");
        return LW_X64_DET_COMPILE_OUT_OF_MEMORY;
    }
    for (i = 0u; i < program->value_count; ++i) {
        fill_value(session, i, &program->values[i], program);
        if ((session->tensors[i].flags & LWM_V0_TENSOR_FLAG_INPUT) != 0u) {
            program->input_value = i;
        }
    }
    /* Arena planning needs operator shapes and lifetimes only. Pack the model
     * constants once, when lowering the executable ops with final offsets. */
    lowered = lower_ops(model, session, program, capacity, 0, error);
    if (lowered != LW_X64_DET_COMPILE_OK) {
        lw_session_free(session);
        lw_x64_det_program_free(program);
        return lowered;
    }
    status = plan_arena_offsets(program, error);
    if (status != LW_STATUS_OK) {
        lw_session_free(session);
        lw_x64_det_program_free(program);
        return status == LW_STATUS_OUT_OF_MEMORY ? LW_X64_DET_COMPILE_OUT_OF_MEMORY
                                                 : LW_X64_DET_COMPILE_INVALID_GRAPH;
    }
    reset_lowering(program, capacity);
    lowered = lower_ops(model, session, program, capacity, 1, error);
    if (lowered != LW_X64_DET_COMPILE_OK) {
        lw_session_free(session);
        lw_x64_det_program_free(program);
        return lowered;
    }
    program->semantic_consumed += program->semantic_elided;
    program->output_value = info.output_count
        ? lwm_read_u32(model->bytes + (size_t)model->output_offset) : 0u;
    for (i = 0u; i < program->op_count; ++i) {
        if (program->ops[i].kind == LW_X64_DET_OP_LAYOUT_CONVERT) {
            ++program->layout_conversions;
        }
    }
    if (program->arena_bytes == 0u) {
        lw_session_free(session);
        lw_x64_det_program_free(program);
        lw_set_error(error, LW_STATUS_INVALID_SHAPE, "DET graph has no runtime arena");
        return LW_X64_DET_COMPILE_INVALID_GRAPH;
    }
    lw_session_free(session);
    lw_set_error(error, LW_STATUS_OK, "");
    *out_program = program;
    return LW_X64_DET_COMPILE_OK;
}

void lw_x64_det_program_free(lw_x64_det_program* program) {
    uint32_t i;
    if (program == NULL) return;
    for (i = 0u; i < program->packed_constant_count; ++i) free(program->constants[i].data);
    free(program->constants);
    free(program->ops);
    free(program->values);
    free(program->effective_node_layout);
    free(program);
}
