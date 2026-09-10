#include "lw_infer.h"
#include "ppm_image.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    lw_example_ppm_image image;
    lw_recognizer_options options;
    lw_recognizer_info info;
    lw_recognition_result result;
    lw_recognizer* recognizer = NULL;
    lw_error error;
    char* text = NULL;
    lw_status status;
    int exit_code = 1;

    memset(&image, 0, sizeof(image));
    if (argc != 4) {
        fprintf(stderr, "usage: lw-abi-v1-consumer <rec.lwm> <dictionary.txt> <image.ppm>\n");
        return 2;
    }
    if (!lw_example_ppm_image_load_bgr(argv[3], &image)) {
        fprintf(stderr, "invalid P6 PPM image: %s\n", argv[3]);
        goto cleanup;
    }
    lw_recognizer_options_init(&options);
    if (options.struct_size != sizeof(options) || options.reserved0 != 0u ||
        options.reserved1 != 0u) {
        fprintf(stderr, "unexpected recognizer option layout\n");
        goto cleanup;
    }
    options.target_width = 960u;
    lw_error_init(&error);
    status = lw_recognizer_create(argv[1], argv[2], &options, &recognizer, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "create failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_recognizer_info_init(&info);
    status = lw_recognizer_get_info(recognizer, &info);
    if (status != LW_STATUS_OK || info.struct_size != sizeof(info) ||
        info.target_width != 960u || info.max_text_capacity == 0u ||
        info.max_text_capacity > SIZE_MAX) {
        fprintf(stderr, "unexpected recognizer info\n");
        goto cleanup;
    }
    text = (char*)malloc((size_t)info.max_text_capacity);
    if (text == NULL) {
        fprintf(stderr, "unable to allocate text buffer\n");
        goto cleanup;
    }
    lw_recognition_result_init(&result);
    lw_error_init(&error);
    status = lw_recognizer_recognize_bgr_u8(
        recognizer, image.pixels, image.byte_count, image.width, image.height,
        image.width * 3u, text, info.max_text_capacity, &result, &error);
    if (status != LW_STATUS_OK || result.struct_size != sizeof(result) ||
        result.required_text_capacity == 0u || text[0] == '\0') {
        fprintf(stderr, "recognition failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    printf("abi_v1_consumer=ok chars=%u target_width=%u\n", result.emitted_count,
           info.target_width);
    exit_code = 0;
cleanup:
    free(text);
    lw_recognizer_free(recognizer);
    lw_example_ppm_image_free(&image);
    return exit_code;
}