#include "lw_infer.h"

#include <stddef.h>
#include <stdio.h>

#define U(value) ((unsigned long long)(value))
#define PRINT_STRUCT(name, type) \
    printf("\"" name "\":{\"size\":%llu,\"struct_size_offset\":%llu}", \
           U(sizeof(type)), U(offsetof(type, struct_size)))

int main(void) {
    printf("{\"structures\":{");
    PRINT_STRUCT("lw_error", lw_error);
    printf(",");
    PRINT_STRUCT("lw_recognizer_options", lw_recognizer_options);
    printf(",");
    PRINT_STRUCT("lw_recognizer_info", lw_recognizer_info);
    printf(",");
    PRINT_STRUCT("lw_recognition_result", lw_recognition_result);
    printf(",");
    PRINT_STRUCT("lw_classifier_options", lw_classifier_options);
    printf(",");
    PRINT_STRUCT("lw_classifier_info", lw_classifier_info);
    printf(",");
    PRINT_STRUCT("lw_classification_result", lw_classification_result);
    printf(",");
    PRINT_STRUCT("lw_detector_options", lw_detector_options);
    printf(",");
    PRINT_STRUCT("lw_detector_info", lw_detector_info);
    printf(",");
    printf("\"lw_detection_box\":{\"size\":%llu,\"struct_size_offset\":null}", U(sizeof(lw_detection_box)));
    printf(",");
    PRINT_STRUCT("lw_detection_result", lw_detection_result);
    printf(",");
    PRINT_STRUCT("lw_ocr_options", lw_ocr_options);
    printf(",");
    PRINT_STRUCT("lw_ocr_info", lw_ocr_info);
    printf(",");
    printf("\"lw_ocr_line\":{\"size\":%llu,\"struct_size_offset\":null}", U(sizeof(lw_ocr_line)));
    printf(",");
    PRINT_STRUCT("lw_ocr_result", lw_ocr_result);
    printf("},\"enums\":{\"LW_STATUS_OK\":%d,\"LW_STATUS_INVALID_ARGUMENT\":%d,\"LW_STATUS_IO_ERROR\":%d,\"LW_STATUS_OUT_OF_MEMORY\":%d,\"LW_STATUS_INVALID_FORMAT\":%d,\"LW_STATUS_UNSUPPORTED_VERSION\":%d,\"LW_STATUS_OUT_OF_BOUNDS\":%d,\"LW_STATUS_CHECKSUM_MISMATCH\":%d,\"LW_STATUS_UNSUPPORTED\":%d,\"LW_STATUS_INVALID_SHAPE\":%d,\"LW_STATUS_MEMORY_LIMIT\":%d,\"LW_READING_ORDER_HORIZONTAL_LTR\":%d,\"LW_READING_ORDER_VERTICAL_RTL\":%d,\"LW_READING_ORDER_VERTICAL_LTR\":%d}}\n",
           LW_STATUS_OK, LW_STATUS_INVALID_ARGUMENT, LW_STATUS_IO_ERROR, LW_STATUS_OUT_OF_MEMORY,
           LW_STATUS_INVALID_FORMAT, LW_STATUS_UNSUPPORTED_VERSION, LW_STATUS_OUT_OF_BOUNDS,
           LW_STATUS_CHECKSUM_MISMATCH, LW_STATUS_UNSUPPORTED, LW_STATUS_INVALID_SHAPE,
           LW_STATUS_MEMORY_LIMIT, LW_READING_ORDER_HORIZONTAL_LTR, LW_READING_ORDER_VERTICAL_RTL,
           LW_READING_ORDER_VERTICAL_LTR);
    return 0;
}