#ifndef LW_DB_GEOMETRY_INTERNAL_H
#define LW_DB_GEOMETRY_INTERNAL_H

#include "det_internal.h"

#include <stddef.h>

struct det_point {
    float x;
    float y;
};

/* Keep allocation overflow checks meaningful on both 32-bit and 64-bit
 * targets. Passing the count through uint64_t avoids comparisons that GCC can
 * prove are always false when the original count is only uint32_t. */
static inline int lw_db_allocation_fits(uint64_t count, size_t element_size) {
    return element_size != 0u && count <= (uint64_t)(SIZE_MAX / element_size);
}

/* Fit one contour with at least three points in workspace->points. Append the
 * box only on acceptance. Hull allocation failures propagate to the map scan,
 * while polygon scratch failure rejects the candidate. */
lw_status lw_db_fit_candidate(
    uint32_t point_count, const float* prediction, uint32_t map_width, uint32_t map_height,
    float box_threshold, float unclip_ratio, uint32_t source_width, uint32_t source_height,
    lw_db_postprocess_workspace* workspace, lw_pipeline_component_profile* profile,
    uint32_t* result_count);

#endif
