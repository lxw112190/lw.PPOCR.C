#include "lw_infer.h"
#include "ppm_image.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#  include <shellapi.h>
#endif

static int demo_main(int argc, char** argv) {
    lw_classifier* classifier = NULL;
    lw_classification_result classification;
    lw_error error;
    lw_example_ppm_image image;
    lw_status status;
    int result = 1;
    memset(&image, 0, sizeof(image));
    if (argc != 3) {
        fprintf(stderr, "usage: lw-classify-ppm <cls.lwm> <image.ppm>\n");
        return 2;
    }
    if (!lw_example_ppm_image_load_bgr(argv[2], &image) ||
        image.width > UINT32_MAX / 3u) {
        fprintf(stderr, "invalid P6 PPM image: %s\n", argv[2]);
        goto cleanup;
    }
    lw_error_init(&error);
    status = lw_classifier_create(argv[1], NULL, &classifier, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "create failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_classification_result_init(&classification);
    lw_error_init(&error);
    status = lw_classifier_classify_bgr_u8(
        classifier, image.pixels, image.byte_count, image.width, image.height,
        image.width * 3u, &classification, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "classification failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    printf("label=%u orientation=%u score=%.8f image=%ux%u resized_width=%u\n",
           classification.label, classification.orientation_degrees,
           classification.score, image.width, image.height,
           classification.resized_width);
    result = 0;

cleanup:
    lw_example_ppm_image_free(&image);
    lw_classifier_free(classifier);
    return result;
}

#if defined(_WIN32)
int main(void) {
    wchar_t** wide_argv;
    char** utf8_argv;
    int argc;
    int index;
    int result = 2;
    SetConsoleOutputCP(CP_UTF8);
    wide_argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (wide_argv == NULL || argc <= 0) {
        return 2;
    }
    utf8_argv = (char**)calloc((size_t)argc, sizeof(*utf8_argv));
    if (utf8_argv == NULL) {
        LocalFree(wide_argv);
        return 2;
    }
    for (index = 0; index < argc; ++index) {
        int bytes = WideCharToMultiByte(
            CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1,
            NULL, 0, NULL, NULL);
        if (bytes <= 0) {
            goto cleanup;
        }
        utf8_argv[index] = (char*)malloc((size_t)bytes);
        if (utf8_argv[index] == NULL ||
            WideCharToMultiByte(
                CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1,
                utf8_argv[index], bytes, NULL, NULL) <= 0) {
            goto cleanup;
        }
    }
    result = demo_main(argc, utf8_argv);
cleanup:
    for (index = 0; index < argc; ++index) {
        free(utf8_argv[index]);
    }
    free(utf8_argv);
    LocalFree(wide_argv);
    return result;
}
#else
int main(int argc, char** argv) {
    return demo_main(argc, argv);
}
#endif
