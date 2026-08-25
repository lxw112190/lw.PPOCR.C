#if !defined(_WIN32) && !defined(__APPLE__)
#  define _POSIX_C_SOURCE 200809L
#endif

#include "executor_internal.h"
#include "lw_infer.h"

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
    (void)context;
#if defined(_WIN32)
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (!QueryPerformanceFrequency(&frequency) ||
        !QueryPerformanceCounter(&counter) || frequency.QuadPart <= 0) {
        return 0u;
    }
    return (uint64_t)((double)counter.QuadPart * 1000000000.0 /
                      (double)frequency.QuadPart);
#elif defined(__APPLE__)
    mach_timebase_info_data_t timebase;
    uint64_t ticks = mach_absolute_time();
    if (mach_timebase_info(&timebase) != KERN_SUCCESS || timebase.denom == 0u) {
        return 0u;
    }
    return (uint64_t)((double)ticks * (double)timebase.numer /
                      (double)timebase.denom);
#else
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0u;
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
        (uint64_t)value.tv_nsec;
#endif
}

static int parse_positive_i32(const char* text, int32_t* value) {
    char* end = NULL;
    long parsed;
    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed <= 0 || parsed > INT32_MAX) {
        return 0;
    }
    *value = (int32_t)parsed;
    return 1;
}

int main(int argc, char** argv) {
    static const char* names[LW_EXECUTION_PROFILE_OPERATOR_CAPACITY] = {
        "unknown", "Conv", "Add", "Mul", "Div", "Erf", "HardSigmoid",
        "BatchNormalization", "ReduceMean", "Relu", "AveragePool", "Squeeze",
        "Transpose", "Unsqueeze", "MatMul", "Softmax"};
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
    if (argc != 4 || !parse_positive_i32(argv[2], &width) || width < 7 ||
        !parse_positive_i32(argv[3], &iterations) || iterations > 10000) {
        fprintf(stderr, "usage: rec-profile-driver rec.lwm width iterations\n");
        return 2;
    }
    lw_error_init(&error);
    status = lw_model_load(argv[1], NULL, &model, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "model load failed: %s: %s\n",
                lw_status_string(status), error.message);
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
        fprintf(stderr, "session create failed: %s: %s\n",
                lw_status_string(status), error.message);
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
    if (input_count > SIZE_MAX / sizeof(float) ||
        output_count > SIZE_MAX / sizeof(float)) {
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
    status = lw_execute_session_f32(
        session, input, input_count, output, output_count, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "profile warm-up failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    memset(&profile, 0, sizeof(profile));
    profile.struct_size = (uint32_t)sizeof(profile);
    profile.clock = profile_clock;
    for (iteration = 0; iteration < iterations; ++iteration) {
        lw_error_init(&error);
        status = lw_execute_session_f32_profiled(
            session, input, input_count, output, output_count, &profile, &error);
        if (status != LW_STATUS_OK) {
            fprintf(stderr, "profile execution failed: %s: %s\n",
                    lw_status_string(status), error.message);
            goto cleanup;
        }
    }
    printf("{\"width\":%d,\"iterations\":%d,\"operators\":[", width, iterations);
    for (index = 1u; index < LW_EXECUTION_PROFILE_OPERATOR_CAPACITY; ++index) {
        if (index != 1u) {
            putchar(',');
        }
        printf("{\"id\":%llu,\"name\":\"%s\",\"nanoseconds\":%llu,"
               "\"invocations\":%llu}",
               (unsigned long long)index, names[index],
               (unsigned long long)profile.operator_nanoseconds[index],
               (unsigned long long)profile.operator_invocations[index]);
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
