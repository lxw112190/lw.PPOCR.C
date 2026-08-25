#include "executor_internal.h"
#include "lw_infer.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int write_output(const char* path, const float* values, size_t count) {
    FILE* file = fopen(path, "wb");
    size_t written;
    if (file == NULL) {
        fprintf(stderr, "unable to open output file: %s\n", path);
        return 0;
    }
    written = fwrite(values, sizeof(*values), count, file);
    if (fclose(file) != 0 || written != count) {
        fprintf(stderr, "unable to write complete output file\n");
        return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    lw_model* model = NULL;
    lw_session* session = NULL;
    lw_tensor_desc input_desc;
    lw_tensor_desc output_desc;
    lw_error error;
    lw_status status;
    char* end = NULL;
    long width_value;
    uint64_t input_count;
    uint64_t output_count = 1u;
    float* input = NULL;
    float* output = NULL;
    float* repeated = NULL;
    uint64_t index;
    int exit_code = 1;

    if (argc != 4) {
        fprintf(stderr, "usage: rec-graph-driver rec.lwm width output.f32\n");
        return 2;
    }
    errno = 0;
    width_value = strtol(argv[2], &end, 10);
    if (errno != 0 || end == argv[2] || *end != '\0' || width_value < 7 || width_value > INT32_MAX) {
        fprintf(stderr, "invalid REC width\n");
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
    input_desc.dimensions[3] = (int32_t)width_value;
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
    input_count = UINT64_C(1) * 3u * 48u * (uint32_t)width_value;
    if (input_count > SIZE_MAX / sizeof(float) || output_count > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "test tensor size exceeds address space\n");
        goto cleanup;
    }
    input = (float*)malloc((size_t)input_count * sizeof(*input));
    output = (float*)malloc((size_t)output_count * sizeof(*output));
    repeated = (float*)malloc((size_t)output_count * sizeof(*repeated));
    if (input == NULL || output == NULL || repeated == NULL) {
        fprintf(stderr, "test buffer allocation failed\n");
        goto cleanup;
    }
    for (index = 0u; index < input_count; ++index) {
        input[index] = (float)((int32_t)((index * 17u) % 257u) - 128) / 127.0f;
    }

    lw_error_init(&error);
    status = lw_execute_session_f32(session, input, input_count - 1u, output, output_count, &error);
    if (status != LW_STATUS_INVALID_SHAPE) {
        fprintf(stderr, "executor accepted an incorrect input element count\n");
        goto cleanup;
    }
    lw_error_init(&error);
    status = lw_execute_session_f32(session, input, input_count, output, output_count, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "graph execution failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_error_init(&error);
    status = lw_execute_session_f32(session, input, input_count, repeated, output_count, &error);
    if (status != LW_STATUS_OK || memcmp(output, repeated, (size_t)output_count * sizeof(*output)) != 0) {
        fprintf(stderr, "repeated graph execution is not deterministic\n");
        goto cleanup;
    }
    if (!write_output(argv[3], output, (size_t)output_count)) {
        goto cleanup;
    }
    printf("shape=1,%d,6906 elements=%llu\n", output_desc.dimensions[1],
           (unsigned long long)output_count);
    exit_code = 0;

cleanup:
    free(repeated);
    free(output);
    free(input);
    lw_session_free(session);
    lw_model_free(model);
    return exit_code;
}
