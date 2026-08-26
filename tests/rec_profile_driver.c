#if !defined(_WIN32) && !defined(__APPLE__)
#  define _POSIX_C_SOURCE 200809L
#endif

#include "executor_internal.h"
#include "lwm_read.h"
#include "lw_infer.h"
#include "model_internal.h"
#include "session_internal.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#elif defined(__APPLE__)
#  include <mach/mach_time.h>
#else
#  include <time.h>
#endif

static uint64_t profile_clock(void* context) {
#if defined(_WIN32)
    LARGE_INTEGER counter;
    const LARGE_INTEGER* frequency = (const LARGE_INTEGER*)context;
    if (frequency == NULL || frequency->QuadPart <= 0 || !QueryPerformanceCounter(&counter)) {
        return 0u;
    }
    return (uint64_t)((double)counter.QuadPart * 1000000000.0 / (double)frequency->QuadPart);
#elif defined(__APPLE__)
    const mach_timebase_info_data_t* timebase = (const mach_timebase_info_data_t*)context;
    uint64_t ticks = mach_absolute_time();
    if (timebase == NULL || timebase->denom == 0u) {
        return 0u;
    }
    return (uint64_t)((double)ticks * (double)timebase->numer / (double)timebase->denom);
#else
    (void)context;
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0u;
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
#endif
}

static int parse_positive_i32(const char* text, int32_t* value) {
    char* end = NULL;
    long parsed;
    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
        return 0;
    }
    *value = (int32_t)parsed;
    return 1;
}

static void print_tensor_dimensions(const lw_runtime_tensor* tensor) {
    uint32_t dimension;
    putchar('[');
    for (dimension = 0u; dimension < tensor->rank; ++dimension) {
        if (dimension != 0u) {
            putchar(',');
        }
        printf("%d", tensor->dimensions[dimension]);
    }
    putchar(']');
}

int main(int argc, char** argv) {
    static const char* names[LW_EXECUTION_PROFILE_OPERATOR_CAPACITY] = {
        "unknown",    "Conv",      "Add",         "Mul",
        "Div",        "Erf",       "HardSigmoid", "BatchNormalization",
        "ReduceMean", "Relu",      "AveragePool", "Squeeze",
        "Transpose",  "Unsqueeze", "MatMul",      "Softmax"};
    lw_model* model = NULL;
    lw_session* session = NULL;
    lw_tensor_desc input_desc;
    lw_tensor_desc output_desc;
    lw_execution_profile profile;
    lw_error error;
    lw_status status;
    int32_t width;
    int32_t iterations;
    uint64_t input_count;
    uint64_t output_count = 1u;
    float* input = NULL;
    float* output = NULL;
    uint64_t index;
    int32_t iteration;
    int exit_code = 1;
#if defined(_WIN32)
    LARGE_INTEGER clock_context;
#elif defined(__APPLE__)
    mach_timebase_info_data_t clock_context;
#endif
    if (argc != 4 || !parse_positive_i32(argv[2], &width) || width < 7 ||
        !parse_positive_i32(argv[3], &iterations) || iterations > 10000) {
        fprintf(stderr, "usage: rec-profile-driver rec.lwm width iterations\n");
        return 2;
    }
    lw_error_init(&error);
    status = lw_model_load(argv[1], NULL, &model, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "model load failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_tensor_desc_init(&input_desc);
    input_desc.dtype = LW_DTYPE_F32;
    input_desc.rank = 4u;
    input_desc.dimensions[0] = 1;
    input_desc.dimensions[1] = 3;
    input_desc.dimensions[2] = 48;
    input_desc.dimensions[3] = width;
    lw_error_init(&error);
    status = lw_session_create(model, &input_desc, 1u, NULL, &session, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "session create failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_tensor_desc_init(&output_desc);
    if (lw_session_get_output_desc(session, 0u, &output_desc) != LW_STATUS_OK) {
        fprintf(stderr, "unable to read output descriptor\n");
        goto cleanup;
    }
    for (index = 0u; index < output_desc.rank; ++index) {
        output_count *= (uint32_t)output_desc.dimensions[index];
    }
    input_count = UINT64_C(3) * 48u * (uint32_t)width;
    if (input_count > SIZE_MAX / sizeof(float) || output_count > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "profile tensor size exceeds address space\n");
        goto cleanup;
    }
    input = (float*)malloc((size_t)input_count * sizeof(*input));
    output = (float*)malloc((size_t)output_count * sizeof(*output));
    if (input == NULL || output == NULL) {
        fprintf(stderr, "profile buffer allocation failed\n");
        goto cleanup;
    }
    for (index = 0u; index < input_count; ++index) {
        input[index] = (float)((int32_t)((index * 17u) % 257u) - 128) / 127.0f;
    }
    lw_error_init(&error);
    status = lw_execute_session_f32(session, input, input_count, output, output_count, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "profile warm-up failed: %s: %s\n", lw_status_string(status),
                error.message);
        goto cleanup;
    }
    memset(&profile, 0, sizeof(profile));
    profile.struct_size = (uint32_t)sizeof(profile);
    profile.clock = profile_clock;
#if defined(_WIN32)
    if (!QueryPerformanceFrequency(&clock_context) || clock_context.QuadPart <= 0) {
        fprintf(stderr, "unable to initialize the profile clock\n");
        goto cleanup;
    }
    profile.clock_context = &clock_context;
#elif defined(__APPLE__)
    if (mach_timebase_info(&clock_context) != KERN_SUCCESS || clock_context.denom == 0u) {
        fprintf(stderr, "unable to initialize the profile clock\n");
        goto cleanup;
    }
    profile.clock_context = &clock_context;
#endif
    for (iteration = 0; iteration < iterations; ++iteration) {
        lw_error_init(&error);
        status = lw_execute_session_f32_profiled(session, input, input_count, output, output_count,
                                                 &profile, &error);
        if (status != LW_STATUS_OK) {
            fprintf(stderr, "profile execution failed: %s: %s\n", lw_status_string(status),
                    error.message);
            goto cleanup;
        }
    }
    printf("{\"width\":%d,\"iterations\":%d,\"operators\":[", width, iterations);
    /* REC v0 uses operator IDs 1 through 15. Keep this focused report stable
     * even though the shared profiler also covers DET/CLS IDs through 21. */
    for (index = 1u; index <= 15u; ++index) {
        if (index != 1u) {
            putchar(',');
        }
        printf("{\"id\":%llu,\"name\":\"%s\",\"nanoseconds\":%llu,"
               "\"invocations\":%llu}",
               (unsigned long long)index, names[index],
               (unsigned long long)profile.operator_nanoseconds[index],
               (unsigned long long)profile.operator_invocations[index]);
    }
    printf("],\"conv_nodes\":[");
    {
        uint32_t node_index;
        int first = 1;
        for (node_index = 0u; node_index < model->info.node_count; ++node_index) {
            const uint8_t* node =
                model->bytes + (size_t)model->node_offset + (size_t)node_index * LWM_V0_NODE_SIZE;
            uint32_t input_index;
            uint32_t weight_index;
            uint32_t output_index;
            uint64_t param_offset;
            const uint8_t* params;
            const lw_runtime_tensor* input_tensor;
            const lw_runtime_tensor* weight_tensor;
            const lw_runtime_tensor* output_tensor;
            if (lwm_read_u16(node) != 1u) {
                continue;
            }
            input_index = lwm_read_u32(node + 8u);
            weight_index = lwm_read_u32(node + 12u);
            output_index = lwm_read_u32(node + 40u);
            param_offset = lwm_read_u64(node + 56u);
            params = model->bytes + (size_t)param_offset;
            input_tensor = &session->tensors[input_index];
            weight_tensor = &session->tensors[weight_index];
            output_tensor = &session->tensors[output_index];
            if (!first) {
                putchar(',');
            }
            first = 0;
            printf("{\"node\":%u,\"nanoseconds\":%llu,\"invocations\":%llu,"
                   "\"input\":[%d,%d,%d,%d],\"weights\":[%d,%d,%d,%d],"
                   "\"output\":[%d,%d,%d,%d],\"group\":%u,"
                   "\"kernel\":[%d,%d],\"strides\":[%d,%d],"
                   "\"dilations\":[%d,%d],\"pads\":[%d,%d,%d,%d]}",
                   node_index, (unsigned long long)profile.node_nanoseconds[node_index],
                   (unsigned long long)profile.node_invocations[node_index],
                   input_tensor->dimensions[0], input_tensor->dimensions[1],
                   input_tensor->dimensions[2], input_tensor->dimensions[3],
                   weight_tensor->dimensions[0], weight_tensor->dimensions[1],
                   weight_tensor->dimensions[2], weight_tensor->dimensions[3],
                   output_tensor->dimensions[0], output_tensor->dimensions[1],
                   output_tensor->dimensions[2], output_tensor->dimensions[3],
                   lwm_read_u32(params + 4u), lwm_read_i32(params + 8u), lwm_read_i32(params + 12u),
                   lwm_read_i32(params + 16u), lwm_read_i32(params + 20u),
                   lwm_read_i32(params + 24u), lwm_read_i32(params + 28u),
                   lwm_read_i32(params + 32u), lwm_read_i32(params + 36u),
                   lwm_read_i32(params + 40u), lwm_read_i32(params + 44u));
        }
    }
    printf("],\"binary_nodes\":[");
    {
        uint32_t node_index;
        int first = 1;
        for (node_index = 0u; node_index < model->info.node_count; ++node_index) {
            const uint8_t* node =
                model->bytes + (size_t)model->node_offset + (size_t)node_index * LWM_V0_NODE_SIZE;
            uint16_t operation = lwm_read_u16(node);
            uint32_t left_index;
            uint32_t right_index;
            uint32_t output_index;
            const lw_runtime_tensor* left_tensor;
            const lw_runtime_tensor* right_tensor;
            const lw_runtime_tensor* output_tensor;
            if (operation < 2u || operation > 4u) {
                continue;
            }
            left_index = lwm_read_u32(node + 8u);
            right_index = lwm_read_u32(node + 12u);
            output_index = lwm_read_u32(node + 40u);
            left_tensor = &session->tensors[left_index];
            right_tensor = &session->tensors[right_index];
            output_tensor = &session->tensors[output_index];
            if (!first) {
                putchar(',');
            }
            first = 0;
            printf("{\"node\":%u,\"operation\":\"%s\","
                   "\"nanoseconds\":%llu,\"invocations\":%llu,\"left\":",
                   node_index, names[operation],
                   (unsigned long long)profile.node_nanoseconds[node_index],
                   (unsigned long long)profile.node_invocations[node_index]);
            print_tensor_dimensions(left_tensor);
            printf(",\"right\":");
            print_tensor_dimensions(right_tensor);
            printf(",\"output\":");
            print_tensor_dimensions(output_tensor);
            printf(",\"left_constant\":%s,\"right_constant\":%s}",
                   (left_tensor->flags & LWM_V0_TENSOR_FLAG_CONSTANT) != 0u ? "true" : "false",
                   (right_tensor->flags & LWM_V0_TENSOR_FLAG_CONSTANT) != 0u ? "true" : "false");
        }
    }
    printf("],\"matmul_nodes\":[");
    {
        uint32_t node_index;
        int first = 1;
        for (node_index = 0u; node_index < model->info.node_count; ++node_index) {
            const uint8_t* node =
                model->bytes + (size_t)model->node_offset + (size_t)node_index * LWM_V0_NODE_SIZE;
            uint32_t input_index;
            uint32_t weight_index;
            uint32_t output_index;
            const lw_runtime_tensor* input_tensor;
            const lw_runtime_tensor* weight_tensor;
            const lw_runtime_tensor* output_tensor;
            uint64_t batch_count = 1u;
            uint32_t dimension;
            if (lwm_read_u16(node) != 14u) {
                continue;
            }
            input_index = lwm_read_u32(node + 8u);
            weight_index = lwm_read_u32(node + 12u);
            output_index = lwm_read_u32(node + 40u);
            input_tensor = &session->tensors[input_index];
            weight_tensor = &session->tensors[weight_index];
            output_tensor = &session->tensors[output_index];
            for (dimension = 0u; dimension + 2u < input_tensor->rank; ++dimension) {
                batch_count *= (uint32_t)input_tensor->dimensions[dimension];
            }
            if (!first) {
                putchar(',');
            }
            first = 0;
            printf("{\"node\":%u,\"nanoseconds\":%llu,\"invocations\":%llu,"
                   "\"batch_count\":%llu,\"rows\":%d,"
                   "\"inner_dimension\":%d,\"columns\":%d,\"input\":",
                   node_index, (unsigned long long)profile.node_nanoseconds[node_index],
                   (unsigned long long)profile.node_invocations[node_index],
                   (unsigned long long)batch_count,
                   input_tensor->dimensions[input_tensor->rank - 2u],
                   input_tensor->dimensions[input_tensor->rank - 1u], weight_tensor->dimensions[1]);
            print_tensor_dimensions(input_tensor);
            printf(",\"weights\":");
            print_tensor_dimensions(weight_tensor);
            printf(",\"output\":");
            print_tensor_dimensions(output_tensor);
            putchar('}');
        }
    }
    printf("]}\n");
    exit_code = 0;
cleanup:
    free(output);
    free(input);
    lw_session_free(session);
    lw_model_free(model);
    return exit_code;
}
