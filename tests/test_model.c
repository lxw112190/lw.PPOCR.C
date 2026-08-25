#include "lw_infer.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    lw_model* model = NULL;
    lw_error error;
    lw_model_options options;
    lw_model_info info;
    lw_status status;
    if (argc != 2) {
        fprintf(stderr, "expected path to rec.lwm\n");
        return 2;
    }
    lw_error_init(&error);
    lw_model_options_init(&options);
    options.max_file_size = 100u;
    status = lw_model_load(argv[1], &options, &model, &error);
    if (status != LW_STATUS_OUT_OF_BOUNDS || model != NULL) {
        fprintf(stderr, "model file limit was not enforced\n");
        return 1;
    }
    lw_error_init(&error);
    status = lw_model_load(argv[1], NULL, &model, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "%s: %s\n", lw_status_string(status), error.message);
        return 1;
    }
    memset(&info, 0, sizeof(info));
    info.struct_size = (uint32_t)sizeof(info);
    status = lw_model_get_info(model, &info);
    if (status != LW_STATUS_OK || info.format_major != 0u || info.format_minor != 1u ||
        info.tensor_count != 282u || info.node_count != 161u ||
        info.input_count != 1u || info.output_count != 1u ||
        info.file_size == 0u || info.weight_size == 0u || info.workspace_size != 0u ||
        info.content_checksum == 0u) {
        fprintf(stderr, "unexpected model metadata\n");
        lw_model_free(model);
        return 1;
    }
    lw_model_free(model);
    return 0;
}
