#include "lw_infer.h"

#include <stdint.h>

_Static_assert(sizeof(lw_error) == 264u, "lw_error ABI changed");
_Static_assert(sizeof(lw_model_options) == 16u, "lw_model_options ABI changed");
_Static_assert(sizeof(lw_model_info) == 56u, "lw_model_info ABI changed");
_Static_assert(LW_STATUS_OK == 0, "success status must remain zero");
_Static_assert(LW_STATUS_UNSUPPORTED == 8, "status numbering changed");

int main(void) {
    lw_model_options options;
    lw_error error;
    lw_model_options_init(&options);
    lw_error_init(&error);
    return options.struct_size == sizeof(options) && error.struct_size == sizeof(error) ? 0 : 1;
}
