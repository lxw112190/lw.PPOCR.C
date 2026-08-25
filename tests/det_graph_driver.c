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
        return 0;
    }
    written = fwrite(values, sizeof(*values), count, file);
    return fclose(file) == 0 && written == count;
}

int main(int argc, char** argv) {
    lw_model* model = NULL;
    lw_session* session = NULL;
    lw_tensor_desc input_desc;
    lw_tensor_desc output_desc;
    lw_error error;
    lw_status status;
    float* input = NULL;
    float* output = NULL;
    float* repeated = NULL;
    uint64_t input_count;
    uint64_t output_count;
    uint64_t index;
    long height;
    long width;
    int exit_code = 1;
    if (argc != 5) {
        fprintf(stderr, "usage: det-graph-driver det.lwm height width output.f32\n");
        return 2;
    }
    height = strtol(argv[2], NULL, 10);
    width = strtol(argv[3], NULL, 10);
    if (height <= 0 || height > INT32_MAX || width <= 0 || width > INT32_MAX) {
        fprintf(stderr, "invalid input dimensions\n");
        return 2;
    }
    input_count = UINT64_C(3) * (uint32_t)height * (uint32_t)width;
    output_count = (uint64_t)(uint32_t)height * (uint32_t)width;
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
    input_desc.dimensions[2] = (int32_t)height;
    input_desc.dimensions[3] = (int32_t)width;
    status = lw_session_create(model, &input_desc, 1u, NULL, &session, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "session create failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_tensor_desc_init(&output_desc);
    status = lw_session_get_output_desc(session, 0u, &output_desc);
    if (status != LW_STATUS_OK || output_desc.dtype != LW_DTYPE_F32 ||
        output_desc.rank != 4u || output_desc.dimensions[0] != 1 ||
        output_desc.dimensions[1] != 1 || output_desc.dimensions[2] != height ||
        output_desc.dimensions[3] != width) {
        fprintf(stderr, "unexpected detector output descriptor\n");
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
        input[index] = (float)((int32_t)((index * 23u) % 269u) - 134) / 134.0f;
    }
    status = lw_execute_session_f32(session, input, input_count, output, output_count, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "graph execution failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    status = lw_execute_session_f32(session, input, input_count, repeated, output_count, &error);
    if (status != LW_STATUS_OK ||
        memcmp(output, repeated, (size_t)output_count * sizeof(*output)) != 0) {
        fprintf(stderr, "repeated graph execution is not deterministic\n");
        goto cleanup;
    }
    if (!write_output(argv[4], output, (size_t)output_count)) {
        fprintf(stderr, "unable to write output\n");
        goto cleanup;
    }
    printf("shape=1,1,%ld,%ld first=%.9g\n", height, width, (double)output[0]);
    exit_code = 0;
cleanup:
    free(repeated);
    free(output);
    free(input);
    lw_session_free(session);
    lw_model_free(model);
    return exit_code;
}
