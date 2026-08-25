#include "lw_infer.h"
#include "ppm_image.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#  include <shellapi.h>
#endif

static int demo_main(int argc, char** argv) {
    lw_detector* detector = NULL;
    lw_detector_info info;
    lw_detection_result detection;
    lw_detection_box* boxes = NULL;
    lw_error error;
    lw_example_ppm_image image;
    lw_status status;
    uint32_t index;
    int return_code = 1;
    memset(&image, 0, sizeof(image));
    if (argc != 3) {
        fprintf(stderr, "usage: lw-detect-ppm <det.lwm> <image.ppm>\n");
        return 2;
    }
    if (!lw_example_ppm_image_load_bgr(argv[2], &image) ||
        image.width > UINT32_MAX / 3u) {
        fprintf(stderr, "invalid P6 PPM image: %s\n", argv[2]);
        goto cleanup;
    }
    lw_error_init(&error);
    status = lw_detector_create(argv[1], NULL, &detector, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "create failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_detector_info_init(&info);
    status = lw_detector_get_info(detector, &info);
    if (status != LW_STATUS_OK || info.max_candidates == 0u ||
        info.max_candidates > SIZE_MAX / sizeof(*boxes)) {
        fprintf(stderr, "unable to read detector limits\n");
        goto cleanup;
    }
    boxes = (lw_detection_box*)calloc(info.max_candidates, sizeof(*boxes));
    if (boxes == NULL) {
        fprintf(stderr, "unable to allocate detection box buffer\n");
        goto cleanup;
    }
    lw_detection_result_init(&detection);
    lw_error_init(&error);
    status = lw_detector_detect_bgr_u8(
        detector, image.pixels, image.byte_count, image.width, image.height,
        image.width * 3u, boxes, info.max_candidates, &detection, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "detection failed: %s: %s\n",
                lw_status_string(status), error.message);
        goto cleanup;
    }
    printf("boxes=%u image=%ux%u resized=%ux%u\n", detection.box_count,
           image.width, image.height, detection.resized_width,
           detection.resized_height);
    for (index = 0u; index < detection.box_count; ++index) {
        const lw_detection_box* box = &boxes[index];
        printf("%u score=%.6f [(%.1f,%.1f),(%.1f,%.1f),(%.1f,%.1f),(%.1f,%.1f)]\n",
               index, box->score, box->x1, box->y1, box->x2, box->y2,
               box->x3, box->y3, box->x4, box->y4);
    }
    return_code = 0;

cleanup:
    free(boxes);
    lw_detector_free(detector);
    lw_example_ppm_image_free(&image);
    return return_code;
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
    if (wide_argv == NULL || argc <= 0) return 2;
    utf8_argv = (char**)calloc((size_t)argc, sizeof(*utf8_argv));
    if (utf8_argv == NULL) {
        LocalFree(wide_argv);
        return 2;
    }
    for (index = 0; index < argc; ++index) {
        int bytes = WideCharToMultiByte(
            CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1,
            NULL, 0, NULL, NULL);
        if (bytes <= 0) goto cleanup;
        utf8_argv[index] = (char*)malloc((size_t)bytes);
        if (utf8_argv[index] == NULL ||
            WideCharToMultiByte(
                CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1,
                utf8_argv[index], bytes, NULL, NULL) <= 0) goto cleanup;
    }
    result = demo_main(argc, utf8_argv);
cleanup:
    for (index = 0; index < argc; ++index) free(utf8_argv[index]);
    free(utf8_argv);
    LocalFree(wide_argv);
    return result;
}
#else
int main(int argc, char** argv) {
    return demo_main(argc, argv);
}
#endif
