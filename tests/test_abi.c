#include "lw_infer.h"

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
_Static_assert(LW_STATUS_OK == 0, "success status must remain zero");
_Static_assert(LW_STATUS_UNSUPPORTED == 8, "status numbering changed");
_Static_assert(LW_STATUS_MEMORY_LIMIT == 10, "status numbering changed");

int main(void) {
    lw_model_options options;
    lw_error error;
    lw_tensor_desc tensor;
    lw_session_options session_options;
    lw_session_info session_info;
    lw_recognizer_options recognizer_options;
    lw_recognizer_info recognizer_info;
    lw_recognition_result recognition_result;
    lw_model_options_init(&options);
    lw_error_init(&error);
    lw_tensor_desc_init(&tensor);
    lw_session_options_init(&session_options);
    lw_session_info_init(&session_info);
    lw_recognizer_options_init(&recognizer_options);
    lw_recognizer_info_init(&recognizer_info);
    lw_recognition_result_init(&recognition_result);
    return options.struct_size == sizeof(options) && error.struct_size == sizeof(error) &&
                   tensor.struct_size == sizeof(tensor) &&
                   session_options.struct_size == sizeof(session_options) &&
                   session_info.struct_size == sizeof(session_info) &&
                   recognizer_options.struct_size == sizeof(recognizer_options) &&
                   recognizer_options.target_width == 320u &&
                   recognizer_info.struct_size == sizeof(recognizer_info) &&
                   recognition_result.struct_size == sizeof(recognition_result)
               ? 0
               : 1;
}
