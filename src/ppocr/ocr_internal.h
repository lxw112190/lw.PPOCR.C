#ifndef LW_OCR_INTERNAL_H
#define LW_OCR_INTERNAL_H

/* Private controls and diagnostics for benchmarks and regression tests. They
 * intentionally stay outside the installed C ABI until the policy is stable. */

#include "cpu_topology.h"
#include "lw_infer.h"

#include <stdint.h>

void lw_ocr_set_det_intra_op_thread_count(lw_ocr* ocr, uint32_t thread_count);
uint32_t lw_ocr_get_det_intra_op_thread_count(const lw_ocr* ocr);
uint32_t lw_ocr_get_det_intra_op_thread_cap(const lw_ocr* ocr);
void lw_ocr_get_cpu_topology(const lw_ocr* ocr, lw_cpu_topology* topology);

#endif
