#include "lw_infer.h"

#include <stddef.h>
#include <stdint.h>

_Static_assert(sizeof(lw_error) == 264u, "lw_error ABI changed");
_Static_assert(sizeof(lw_model_options) == 16u, "lw_model_options ABI changed");
_Static_assert(sizeof(lw_model_info) == 56u, "lw_model_info ABI changed");
_Static_assert(sizeof(lw_tensor_desc) == 48u, "lw_tensor_desc ABI changed");
_Static_assert(sizeof(lw_session_options) == 24u, "lw_session_options ABI changed");
_Static_assert(sizeof(lw_session_info) == 24u, "lw_session_info ABI changed");
_Static_assert(sizeof(lw_recognizer_options) == 48u, "lw_recognizer_options ABI changed");
_Static_assert(sizeof(lw_recognizer_info) == 40u, "lw_recognizer_info ABI changed");
_Static_assert(sizeof(lw_recognition_result) == 32u, "lw_recognition_result ABI changed");
_Static_assert(sizeof(lw_classifier_options) == 40u, "lw_classifier_options ABI changed");
_Static_assert(sizeof(lw_classifier_info) == 24u, "lw_classifier_info ABI changed");
_Static_assert(sizeof(lw_classification_result) == 24u, "lw_classification_result ABI changed");
_Static_assert(sizeof(lw_detector_options) == 64u, "lw_detector_options ABI changed");
_Static_assert(sizeof(lw_detector_info) == 40u, "lw_detector_info ABI changed");
_Static_assert(sizeof(lw_detection_box) == 40u, "lw_detection_box ABI changed");
_Static_assert(sizeof(lw_detection_result) == 32u, "lw_detection_result ABI changed");
_Static_assert(sizeof(lw_ocr_options) == 176u, "lw_ocr_options ABI changed");
_Static_assert(sizeof(lw_ocr_info) == 40u, "lw_ocr_info ABI changed");
_Static_assert(sizeof(lw_ocr_line) == 80u, "lw_ocr_line ABI changed");
_Static_assert(sizeof(lw_ocr_result) == 40u, "lw_ocr_result ABI changed");
_Static_assert(LW_STATUS_OK == 0, "success status must remain zero");
_Static_assert(LW_STATUS_UNSUPPORTED == 8, "status numbering changed");
_Static_assert(LW_STATUS_MEMORY_LIMIT == 10, "status numbering changed");
_Static_assert(LW_READING_ORDER_HORIZONTAL_LTR == 0, "reading order numbering changed");
_Static_assert(LW_READING_ORDER_VERTICAL_RTL == 1, "reading order numbering changed");
_Static_assert(LW_READING_ORDER_VERTICAL_LTR == 2, "reading order numbering changed");

int main(void) {
    lw_model_options options;
    lw_error error;
    lw_tensor_desc tensor;
    lw_session_options session_options;
    lw_session_info session_info;
    lw_recognizer_options recognizer_options;
    lw_recognizer_info recognizer_info;
    lw_recognition_result recognition_result;
    lw_classifier_options classifier_options;
    lw_classifier_info classifier_info;
    lw_classification_result classification_result;
    lw_detector_options detector_options;
    lw_detector_info detector_info;
    lw_detection_result detection_result;
    lw_ocr_options ocr_options;
    lw_ocr_info ocr_info;
    lw_ocr_result ocr_result;
    uint32_t reading_order = 99u;
    lw_model_options_init(&options);
    lw_error_init(&error);
    lw_tensor_desc_init(&tensor);
    lw_session_options_init(&session_options);
    lw_session_info_init(&session_info);
    lw_recognizer_options_init(&recognizer_options);
    lw_recognizer_info_init(&recognizer_info);
    lw_recognition_result_init(&recognition_result);
    lw_classifier_options_init(&classifier_options);
    lw_classifier_info_init(&classifier_info);
    lw_classification_result_init(&classification_result);
    lw_detector_options_init(&detector_options);
    lw_detector_info_init(&detector_info);
    lw_detection_result_init(&detection_result);
    lw_ocr_options_init(&ocr_options);
    lw_ocr_info_init(&ocr_info);
    lw_ocr_result_init(&ocr_result);
    if (lw_detector_set_reading_order(NULL, LW_READING_ORDER_VERTICAL_RTL, &error) !=
            LW_STATUS_INVALID_ARGUMENT ||
        lw_detector_get_reading_order(NULL, &reading_order, &error) !=
            LW_STATUS_INVALID_ARGUMENT ||
        lw_ocr_set_reading_order(NULL, LW_READING_ORDER_VERTICAL_RTL, &error) !=
            LW_STATUS_INVALID_ARGUMENT ||
        lw_ocr_get_reading_order(NULL, &reading_order, &error) != LW_STATUS_INVALID_ARGUMENT)
        return 1;
    return options.struct_size == sizeof(options) && error.struct_size == sizeof(error) &&
                   tensor.struct_size == sizeof(tensor) &&
                   session_options.struct_size == sizeof(session_options) &&
                   session_info.struct_size == sizeof(session_info) &&
                   recognizer_options.struct_size == sizeof(recognizer_options) &&
                   recognizer_options.target_width == 320u &&
                   recognizer_info.struct_size == sizeof(recognizer_info) &&
                   recognition_result.struct_size == sizeof(recognition_result) &&
                   classifier_options.struct_size == sizeof(classifier_options) &&
                   classifier_info.struct_size == sizeof(classifier_info) &&
                   classification_result.struct_size == sizeof(classification_result) &&
                   detector_options.struct_size == sizeof(detector_options) &&
                   detector_options.limit_side_length == 960u &&
                   detector_options.max_candidates == 1000u &&
                   detector_info.struct_size == sizeof(detector_info) &&
                   detection_result.struct_size == sizeof(detection_result) &&
                   ocr_options.struct_size == sizeof(ocr_options) &&
                   ocr_options.use_direction_classification == 1u &&
                   ocr_options.detector.struct_size == sizeof(ocr_options.detector) &&
                   ocr_options.classifier.struct_size == sizeof(ocr_options.classifier) &&
                   ocr_options.recognizer.struct_size == sizeof(ocr_options.recognizer) &&
                   ocr_info.struct_size == sizeof(ocr_info) &&
                   ocr_result.struct_size == sizeof(ocr_result)
               ? 0
               : 1;
}
