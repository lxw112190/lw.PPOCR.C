#include "lw_infer.h"
#include "ppm_image.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#  include <shellapi.h>
#endif

static int demo_main(int argc, char** argv) {
    lw_recognizer* recognizer = NULL;
    lw_recognizer_info info;
    lw_recognition_result recognition;
    lw_error error;
    lw_example_ppm_image image;
    char* text = NULL;
    lw_status status;
    int result = 1;
    memset(&image, 0, sizeof(image));
    if (argc != 4) {
        fprintf(stderr, "usage: lw-recognize-ppm <rec.lwm> <dictionary.txt> <image.ppm>\n");
        return 2;
    }
    if (!lw_example_ppm_image_load_bgr(argv[3], &image) ||
        image.width > UINT32_MAX / 3u) {
        fprintf(stderr, "invalid P6 PPM image: %s\n", argv[3]);
        goto cleanup;
    }
    lw_error_init(&error);
    status = lw_recognizer_create(argv[1], argv[2], NULL, &recognizer, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "create failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_recognizer_info_init(&info);
    if (lw_recognizer_get_info(recognizer, &info) != LW_STATUS_OK ||
        info.max_text_capacity == 0u || info.max_text_capacity > SIZE_MAX) {
        fprintf(stderr, "unable to query recognizer information\n");
        goto cleanup;
    }
    text = (char*)malloc((size_t)info.max_text_capacity);
    if (text == NULL) {
        fprintf(stderr, "unable to allocate text buffer\n");
        goto cleanup;
    }
    lw_recognition_result_init(&recognition);
    lw_error_init(&error);
    status = lw_recognizer_recognize_bgr_u8(
        recognizer, image.pixels, image.byte_count, image.width, image.height,
        image.width * 3u, text, info.max_text_capacity, &recognition, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "recognition failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    printf("text=%s\n", text);
    printf("score=%.8f chars=%u image=%ux%u resized_width=%u time_steps=%u\n",
           recognition.score, recognition.emitted_count, image.width, image.height,
           recognition.resized_width, recognition.time_steps);
    result = 0;
cleanup:
    free(text);
    lw_example_ppm_image_free(&image);
    lw_recognizer_free(recognizer);
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
