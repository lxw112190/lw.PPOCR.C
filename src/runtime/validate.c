#include "model_internal.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

typedef struct lwm_header {
    uint32_t flags;
    uint32_t tensor_count;
    uint32_t node_count;
    uint32_t input_count;
    uint32_t output_count;
    uint64_t input_offset;
    uint64_t output_offset;
    uint64_t tensor_offset;
    uint64_t node_offset;
    uint64_t param_offset;
    uint64_t param_size;
    uint64_t string_offset;
    uint64_t string_size;
    uint64_t weight_offset;
    uint64_t weight_size;
    uint64_t file_size;
    uint64_t workspace_size;
    uint64_t checksum;
} lwm_header;

static uint16_t read_u16(const uint8_t* p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t read_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int32_t read_i32(const uint8_t* p) {
    return (int32_t)read_u32(p);
}

static float read_f32(const uint8_t* p) {
    uint32_t bits = read_u32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint64_t read_u64(const uint8_t* p) {
    return (uint64_t)read_u32(p) | ((uint64_t)read_u32(p + 4) << 32);
}

static int is_aligned8(uint64_t value) {
    return (value & UINT64_C(7)) == 0u;
}

static int range_valid(uint64_t offset, uint64_t size, uint64_t file_size) {
    return offset <= file_size && size <= file_size - offset;
}

static int table_range_valid(uint64_t offset, uint32_t count, uint32_t record_size, uint64_t file_size) {
    uint64_t size = (uint64_t)count * (uint64_t)record_size;
    return range_valid(offset, size, file_size);
}

static uint64_t table_end(uint64_t offset, uint32_t count, uint32_t record_size) {
    return offset + (uint64_t)count * (uint64_t)record_size;
}

static lw_status fail(lw_error* error, lw_status status, const char* message) {
    lw_set_error(error, status, message);
    return status;
}

static uint32_t dtype_size(uint32_t dtype) {
    switch (dtype) {
        case 1u: return 4u;
        case 2u: return 4u;
        case 3u: return 8u;
        case 4u: return 1u;
        default: return 0u;
    }
}

static uint32_t expected_param_size(uint16_t op) {
    switch (op) {
        case 1u: return 64u;
        case 6u: return 16u;
        case 7u: return 24u;
        case 8u: return 48u;
        case 10u: return 64u;
        case 11u: return 40u;
        case 12u: return 40u;
        case 13u: return 40u;
        case 15u: return 16u;
        case 17u: return 16u;
        case 18u: return 64u;
        case 19u: return 64u;
        case 20u: return 32u;
        default: return 0u;
    }
}

static int param_in_section(uint64_t offset, uint32_t size, const lwm_header* h) {
    return offset >= h->param_offset && range_valid(offset, size, h->param_offset + h->param_size);
}

static lw_status validate_params(const uint8_t* p, uint16_t op, uint32_t size, lw_error* error) {
    uint32_t i;
    if (size == 0u) {
        return LW_STATUS_OK;
    }
    if (read_u16(p) != 1u) {
        return fail(error, LW_STATUS_UNSUPPORTED_VERSION, "unsupported operator parameter version");
    }
    if (op == 1u || op == 18u) {
        if (read_u16(p + 2) != 2u || read_u32(p + 4) == 0u ||
            read_u32(p + 4) > INT32_MAX) {
            return fail(error, LW_STATUS_INVALID_FORMAT, "invalid convolution parameter record");
        }
        for (i = 0u; i < 6u; ++i) {
            if (read_i32(p + 8u + i * 4u) <= 0) {
                return fail(error, LW_STATUS_INVALID_FORMAT, "convolution kernel, stride, or dilation is invalid");
            }
        }
        for (i = 0u; i < 4u; ++i) {
            if (read_i32(p + 32u + i * 4u) < 0 || read_u32(p + 48u + i * 4u) != 0u) {
                return fail(error, LW_STATUS_INVALID_FORMAT, "convolution padding or reserved field is invalid");
            }
        }
    } else if (op == 6u) {
        if (read_u16(p + 2) != 0u || read_u32(p + 12) != 0u) {
            return fail(error, LW_STATUS_INVALID_FORMAT, "invalid HardSigmoid parameter record");
        }
    } else if (op == 7u) {
        if (read_u16(p + 2) != 0u || read_u32(p + 12) != 0u || read_u32(p + 16) != 0u || read_u32(p + 20) != 0u) {
            return fail(error, LW_STATUS_INVALID_FORMAT, "invalid BatchNormalization parameter record");
        }
    } else if (op == 8u) {
        if (read_u16(p + 2) > LWM_V0_MAX_DIMS || read_u32(p + 4) > 1u || read_u32(p + 8) > 1u || read_u32(p + 44) != 0u) {
            return fail(error, LW_STATUS_INVALID_FORMAT, "invalid ReduceMean parameter record");
        }
    } else if (op == 10u || op == 19u) {
        if (read_u16(p + 2) != 0u || read_u32(p + 4) != 2u ||
            read_u32(p + 40) > 1u || read_u32(p + 44) > (op == 10u ? 1u : 0u)) {
            return fail(error, LW_STATUS_INVALID_FORMAT, "invalid pool parameter record");
        }
        for (i = 0u; i < 4u; ++i) {
            if (read_i32(p + 8u + i * 4u) <= 0) {
                return fail(error, LW_STATUS_INVALID_FORMAT, "pool kernel or stride is invalid");
            }
        }
        for (i = 0u; i < 4u; ++i) {
            if (read_i32(p + 24u + i * 4u) < 0) {
                return fail(error, LW_STATUS_INVALID_FORMAT, "pool padding is invalid");
            }
        }
        for (i = 0u; i < 4u; ++i) {
            if (read_u32(p + 48u + i * 4u) != 0u) {
                return fail(error, LW_STATUS_INVALID_FORMAT, "pool reserved field is non-zero");
            }
        }
    } else if (op == 11u || op == 12u || op == 13u) {
        if (read_u16(p + 2) > LWM_V0_MAX_DIMS || read_u32(p + 36) != 0u) {
            return fail(error, LW_STATUS_INVALID_FORMAT, "invalid axes parameter record");
        }
    } else if (op == 15u) {
        if (read_u16(p + 2) != 0u || read_u32(p + 8) != 0u || read_u32(p + 12) != 0u) {
            return fail(error, LW_STATUS_INVALID_FORMAT, "invalid Softmax parameter record");
        }
    } else if (op == 17u) {
        if (read_u16(p + 2) != 0u || read_u32(p + 8) != 0u ||
            read_u32(p + 12) != 0u) {
            return fail(error, LW_STATUS_INVALID_FORMAT, "invalid Concat parameter record");
        }
    } else if (op == 20u) {
        if (read_u16(p + 2) != 4u || read_u32(p + 20) != 0u ||
            read_u32(p + 24) != 0u || read_u32(p + 28) != 0u) {
            return fail(error, LW_STATUS_INVALID_FORMAT, "invalid Resize parameter record");
        }
        for (i = 0u; i < 4u; ++i) {
            float scale = read_f32(p + 4u + i * 4u);
            if (!isfinite(scale) || scale <= 0.0f) {
                return fail(error, LW_STATUS_INVALID_FORMAT, "Resize scale is invalid");
            }
        }
    }
    return LW_STATUS_OK;
}

static uint64_t compute_checksum(const uint8_t* bytes, size_t count) {
    uint64_t value = UINT64_C(14695981039346656037);
    size_t i;
    for (i = 0u; i < count; ++i) {
        uint8_t byte = (i >= LWM_V0_CHECKSUM_OFFSET && i < LWM_V0_CHECKSUM_OFFSET + 8u) ? 0u : bytes[i];
        value ^= byte;
        value *= UINT64_C(1099511628211);
    }
    return value;
}

lw_status lw_validate_lwm_v0(lw_model* model, lw_error* error) {
    const uint8_t* b = model->bytes;
    lwm_header h;
    uint32_t i;
    if (memcmp(b, "LWM0", 4u) != 0) {
        return fail(error, LW_STATUS_INVALID_FORMAT, "invalid LWM magic");
    }
    if (read_u16(b + 4) != 0u || read_u16(b + 6) != 1u) {
        return fail(error, LW_STATUS_UNSUPPORTED_VERSION, "unsupported LWM format version");
    }
    if (read_u32(b + 8) != LWM_V0_HEADER_SIZE) {
        return fail(error, LW_STATUS_INVALID_FORMAT, "invalid LWM header size");
    }
    h.flags = read_u32(b + 12);
    h.tensor_count = read_u32(b + 16);
    h.node_count = read_u32(b + 20);
    h.input_count = read_u32(b + 24);
    h.output_count = read_u32(b + 28);
    h.input_offset = read_u64(b + 32);
    h.output_offset = read_u64(b + 40);
    h.tensor_offset = read_u64(b + 48);
    h.node_offset = read_u64(b + 56);
    h.param_offset = read_u64(b + 64);
    h.param_size = read_u64(b + 72);
    h.string_offset = read_u64(b + 80);
    h.string_size = read_u64(b + 88);
    h.weight_offset = read_u64(b + 96);
    h.weight_size = read_u64(b + 104);
    h.file_size = read_u64(b + 112);
    h.workspace_size = read_u64(b + 120);
    h.checksum = read_u64(b + 128);

    if (h.flags != LWM_V0_HEADER_FLAG_NO_MEMORY_PLAN || h.workspace_size != 0u) {
        return fail(error, LW_STATUS_UNSUPPORTED, "unsupported LWM header flags or workspace plan");
    }
    if (read_u64(b + 136) != 0u || read_u64(b + 144) != 0u || read_u64(b + 152) != 0u) {
        return fail(error, LW_STATUS_INVALID_FORMAT, "non-zero LWM header reserved field");
    }
    if (h.file_size != model->byte_count) {
        return fail(error, LW_STATUS_OUT_OF_BOUNDS, "LWM file_size does not match the actual file");
    }
    if (!is_aligned8(h.input_offset) || !is_aligned8(h.output_offset) || !is_aligned8(h.tensor_offset) ||
        !is_aligned8(h.node_offset) || !is_aligned8(h.param_offset) || !is_aligned8(h.string_offset) ||
        !is_aligned8(h.weight_offset)) {
        return fail(error, LW_STATUS_INVALID_FORMAT, "LWM section offset is not 8-byte aligned");
    }
    if (!table_range_valid(h.input_offset, h.input_count, 4u, h.file_size) ||
        !table_range_valid(h.output_offset, h.output_count, 4u, h.file_size) ||
        !table_range_valid(h.tensor_offset, h.tensor_count, LWM_V0_TENSOR_SIZE, h.file_size) ||
        !table_range_valid(h.node_offset, h.node_count, LWM_V0_NODE_SIZE, h.file_size) ||
        !range_valid(h.param_offset, h.param_size, h.file_size) ||
        !range_valid(h.string_offset, h.string_size, h.file_size) ||
        !range_valid(h.weight_offset, h.weight_size, h.file_size)) {
        return fail(error, LW_STATUS_OUT_OF_BOUNDS, "LWM section range exceeds file bounds");
    }
    if (h.input_offset < LWM_V0_HEADER_SIZE ||
        table_end(h.input_offset, h.input_count, 4u) > h.output_offset ||
        table_end(h.output_offset, h.output_count, 4u) > h.tensor_offset ||
        table_end(h.tensor_offset, h.tensor_count, LWM_V0_TENSOR_SIZE) > h.node_offset ||
        table_end(h.node_offset, h.node_count, LWM_V0_NODE_SIZE) > h.param_offset ||
        h.param_offset + h.param_size > h.string_offset ||
        h.string_offset + h.string_size > h.weight_offset ||
        h.weight_offset + h.weight_size != h.file_size) {
        return fail(error, LW_STATUS_OUT_OF_BOUNDS, "LWM sections overlap or are out of order");
    }
    if (h.checksum == 0u || compute_checksum(b, model->byte_count) != h.checksum) {
        return fail(error, LW_STATUS_CHECKSUM_MISMATCH, "LWM content checksum mismatch");
    }

    for (i = 0u; i < h.tensor_count; ++i) {
        const uint8_t* t = b + (size_t)h.tensor_offset + (size_t)i * LWM_V0_TENSOR_SIZE;
        uint32_t dtype = read_u32(t);
        uint32_t rank = read_u32(t + 4);
        uint32_t flags = read_u32(t + 40);
        uint64_t data_offset = read_u64(t + 48);
        uint64_t data_size = read_u64(t + 56);
        uint64_t workspace_offset = read_u64(t + 64);
        uint64_t workspace_size = read_u64(t + 72);
        uint64_t element_count = 1u;
        uint32_t j;
        int dynamic = 0;
        uint32_t item_size = dtype_size(dtype);
        if (item_size == 0u || rank > LWM_V0_MAX_DIMS || (flags & ~7u) != 0u || read_u32(t + 44) != 0u) {
            return fail(error, LW_STATUS_INVALID_FORMAT, "invalid tensor type, rank, flags, or reserved field");
        }
        for (j = 0u; j < LWM_V0_MAX_DIMS; ++j) {
            int32_t dim = read_i32(t + 8u + j * 4u);
            if (j < rank) {
                if (dim == -1) {
                    dynamic = 1;
                } else if (dim <= 0 || element_count > UINT64_MAX / (uint32_t)dim) {
                    return fail(error, LW_STATUS_OUT_OF_BOUNDS, "invalid or overflowing tensor dimensions");
                } else {
                    element_count *= (uint32_t)dim;
                }
            } else if (dim != 0) {
                return fail(error, LW_STATUS_INVALID_FORMAT, "unused tensor dimension is non-zero");
            }
        }
        if ((flags & LWM_V0_TENSOR_FLAG_CONSTANT) != 0u) {
            if (dynamic || !is_aligned8(data_offset) || data_offset < h.weight_offset ||
                !range_valid(data_offset, data_size, h.weight_offset + h.weight_size) ||
                element_count > UINT64_MAX / item_size || data_size != element_count * item_size) {
                return fail(error, LW_STATUS_OUT_OF_BOUNDS, "invalid constant tensor data range or size");
            }
        } else if (data_offset != 0u || data_size != 0u) {
            return fail(error, LW_STATUS_INVALID_FORMAT, "non-constant tensor contains file data");
        }
        if (workspace_offset != LWM_V0_NO_WORKSPACE || workspace_size != 0u) {
            return fail(error, LW_STATUS_UNSUPPORTED, "v0.1 tensor contains an unsupported workspace plan");
        }
    }

    for (i = 0u; i < h.node_count; ++i) {
        const uint8_t* n = b + (size_t)h.node_offset + (size_t)i * LWM_V0_NODE_SIZE;
        uint16_t op = read_u16(n);
        uint16_t input_count = read_u16(n + 2);
        uint16_t output_count = read_u16(n + 4);
        uint64_t param_offset = read_u64(n + 56);
        uint32_t param_size = read_u32(n + 64);
        uint32_t expected_size;
        uint32_t j;
        lw_status param_status;
        if (op == 0u || op > 21u || input_count > LWM_V0_MAX_NODE_INPUTS ||
            output_count == 0u || output_count > LWM_V0_MAX_NODE_OUTPUTS ||
            read_u16(n + 6) != 0u || read_u32(n + 68) != 0u) {
            return fail(error, LW_STATUS_INVALID_FORMAT, "invalid node operator, arity, flags, or reserved field");
        }
        for (j = 0u; j < LWM_V0_MAX_NODE_INPUTS; ++j) {
            uint32_t index = read_u32(n + 8u + j * 4u);
            if ((j < input_count && index >= h.tensor_count) || (j >= input_count && index != 0u)) {
                return fail(error, LW_STATUS_OUT_OF_BOUNDS, "invalid node input tensor index");
            }
        }
        for (j = 0u; j < LWM_V0_MAX_NODE_OUTPUTS; ++j) {
            uint32_t index = read_u32(n + 40u + j * 4u);
            if ((j < output_count && index >= h.tensor_count) || (j >= output_count && index != 0u)) {
                return fail(error, LW_STATUS_OUT_OF_BOUNDS, "invalid node output tensor index");
            }
        }
        expected_size = expected_param_size(op);
        if (param_size != expected_size || (param_size == 0u && param_offset != 0u) ||
            (param_size != 0u && (!is_aligned8(param_offset) || !param_in_section(param_offset, param_size, &h)))) {
            return fail(error, LW_STATUS_OUT_OF_BOUNDS, "invalid node parameter range or size");
        }
        param_status = validate_params(param_size == 0u ? NULL : b + (size_t)param_offset, op, param_size, error);
        if (param_status != LW_STATUS_OK) {
            return param_status;
        }
    }

    for (i = 0u; i < h.input_count; ++i) {
        uint32_t index = read_u32(b + (size_t)h.input_offset + (size_t)i * 4u);
        const uint8_t* tensor;
        if (index >= h.tensor_count) {
            return fail(error, LW_STATUS_OUT_OF_BOUNDS, "invalid graph input tensor index");
        }
        tensor = b + (size_t)h.tensor_offset + (size_t)index * LWM_V0_TENSOR_SIZE;
        if ((read_u32(tensor + 40) & LWM_V0_TENSOR_FLAG_INPUT) == 0u) {
            return fail(error, LW_STATUS_INVALID_FORMAT, "graph input tensor flag is missing");
        }
    }
    for (i = 0u; i < h.output_count; ++i) {
        uint32_t index = read_u32(b + (size_t)h.output_offset + (size_t)i * 4u);
        const uint8_t* tensor;
        if (index >= h.tensor_count) {
            return fail(error, LW_STATUS_OUT_OF_BOUNDS, "invalid graph output tensor index");
        }
        tensor = b + (size_t)h.tensor_offset + (size_t)index * LWM_V0_TENSOR_SIZE;
        if ((read_u32(tensor + 40) & LWM_V0_TENSOR_FLAG_OUTPUT) == 0u) {
            return fail(error, LW_STATUS_INVALID_FORMAT, "graph output tensor flag is missing");
        }
    }

    memset(&model->info, 0, sizeof(model->info));
    model->info.struct_size = (uint32_t)sizeof(model->info);
    model->info.format_major = 0u;
    model->info.format_minor = 1u;
    model->info.tensor_count = h.tensor_count;
    model->info.node_count = h.node_count;
    model->info.input_count = h.input_count;
    model->info.output_count = h.output_count;
    model->info.file_size = h.file_size;
    model->info.weight_size = h.weight_size;
    model->info.workspace_size = h.workspace_size;
    model->info.content_checksum = h.checksum;
    model->input_offset = h.input_offset;
    model->output_offset = h.output_offset;
    model->tensor_offset = h.tensor_offset;
    model->node_offset = h.node_offset;
    model->param_offset = h.param_offset;
    model->param_size = h.param_size;
    model->weight_offset = h.weight_offset;
    model->weight_size = h.weight_size;
    return LW_STATUS_OK;
}
