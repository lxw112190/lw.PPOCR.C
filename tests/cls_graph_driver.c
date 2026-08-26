#include "executor_internal.h"
#include "lw_infer.h"

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
    enum { INPUT_WIDTH = 160, INPUT_HEIGHT = 80, OUTPUT_COUNT = 2 };
    const uint64_t input_count = UINT64_C(3) * INPUT_HEIGHT * INPUT_WIDTH;
    lw_model* model = NULL;
    lw_session* session = NULL;
    lw_tensor_desc input_desc;
    lw_tensor_desc output_desc;
    lw_error error;
    lw_status status;
    float* input = NULL;
    float output[OUTPUT_COUNT];
    float repeated[OUTPUT_COUNT];
    uint64_t index;
    int exit_code = 1;

    if (argc != 3) {
        fprintf(stderr, "usage: cls-graph-driver cls.lwm output.f32\n");
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
    input_desc.dimensions[2] = INPUT_HEIGHT;
    input_desc.dimensions[3] = INPUT_WIDTH;
    lw_error_init(&error);
    status = lw_session_create(model, &input_desc, 1u, NULL, &session, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "session create failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_tensor_desc_init(&output_desc);
    status = lw_session_get_output_desc(session, 0u, &output_desc);
    if (status != LW_STATUS_OK || output_desc.dtype != LW_DTYPE_F32 || output_desc.rank != 2u ||
        output_desc.dimensions[0] != 1 || output_desc.dimensions[1] != OUTPUT_COUNT) {
        fprintf(stderr, "unexpected classifier output descriptor\n");
        goto cleanup;
    }
    if (input_count > SIZE_MAX / sizeof(*input)) {
        fprintf(stderr, "test tensor size exceeds address space\n");
        goto cleanup;
    }
    input = (float*)malloc((size_t)input_count * sizeof(*input));
    if (input == NULL) {
        fprintf(stderr, "test buffer allocation failed\n");
        goto cleanup;
    }
    for (index = 0u; index < input_count; ++index) {
        input[index] = (float)((int32_t)((index * 19u) % 263u) - 131) / 131.0f;
    }
    lw_error_init(&error);
    status = lw_execute_session_f32(session, input, input_count - 1u, output, OUTPUT_COUNT, &error);
    if (status != LW_STATUS_INVALID_SHAPE) {
        fprintf(stderr, "executor accepted an incorrect input element count\n");
        goto cleanup;
    }
    lw_error_init(&error);
    status = lw_execute_session_f32(session, input, input_count, output, OUTPUT_COUNT, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "graph execution failed: %s: %s\n", lw_status_string(status),
                error.message);
        goto cleanup;
    }
    lw_error_init(&error);
    status = lw_execute_session_f32(session, input, input_count, repeated, OUTPUT_COUNT, &error);
    if (status != LW_STATUS_OK || memcmp(output, repeated, sizeof(output)) != 0) {
        fprintf(stderr, "repeated graph execution is not deterministic\n");
        goto cleanup;
    }
    if (!write_output(argv[2], output, OUTPUT_COUNT)) {
        goto cleanup;
    }
    printf("shape=1,2 label=%d score=%.9g\n", output[1] > output[0] ? 1 : 0,
           (double)(output[1] > output[0] ? output[1] : output[0]));
    exit_code = 0;

cleanup:
    free(input);
    lw_session_free(session);
    lw_model_free(model);
    return exit_code;
}
