from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools import compare_opencv_dnn


def response_line(text: str, offset: float = 0.0):
    return {
        "text": text,
        "x1": 1.0 + offset,
        "y1": 2.0 + offset,
        "x2": 11.0 + offset,
        "y2": 2.0 + offset,
        "x3": 11.0 + offset,
        "y3": 7.0 + offset,
        "x4": 1.0 + offset,
        "y4": 7.0 + offset,
    }


class CompareOpenCvDnnTest(unittest.TestCase):
    def test_ppm_to_bmp_preserves_dimensions_and_bgr_pixels(self) -> None:
        bmp = compare_opencv_dnn.ppm_to_bmp(
            b"P6\n2 1\n255\n" + bytes((10, 20, 30, 40, 50, 60))
        )
        self.assertEqual(bmp[:2], b"BM")
        self.assertEqual(struct.unpack_from("<ii", bmp, 18), (2, 1))
        self.assertEqual(bmp[54:60], bytes((30, 20, 10, 60, 50, 40)))

    def test_cpu_list_accepts_ranges_and_rejects_empty_values(self) -> None:
        self.assertEqual(compare_opencv_dnn.parse_cpu_list("0-2,4,2"), [0, 1, 2, 4])
        with self.assertRaises(ValueError):
            compare_opencv_dnn.parse_cpu_list("")
        with self.assertRaises(ValueError):
            compare_opencv_dnn.parse_cpu_list("3-1")

    def test_text_normalization_handles_width_and_spacing(self) -> None:
        left = "【品牌】：代加工方式/OEM ODM"
        right = "【品牌】:代加工方式/OEMODM"
        self.assertEqual(
            compare_opencv_dnn.normalize_text(left),
            compare_opencv_dnn.normalize_text(right),
        )

    def test_result_comparison_separates_exact_and_normalized_text(self) -> None:
        c_lines = compare_opencv_dnn.normalize_result(
            {"result": [response_line("每瓶22元，1000瓶起订）")]}
        )
        opencv_lines = compare_opencv_dnn.normalize_result(
            {"result": [response_line("每瓶22元,1000瓶起订)", 2.5)]}
        )
        comparison = compare_opencv_dnn.compare_results(c_lines, opencv_lines, 5.0)
        self.assertFalse(comparison["exact_text_match"])
        self.assertTrue(comparison["normalized_text_match"])
        self.assertTrue(comparison["bounds_within_tolerance"])
        self.assertEqual(comparison["bounds_max_absolute_delta_px"], 2.5)

    def test_result_comparison_rejects_count_and_coordinate_mismatch(self) -> None:
        c_lines = compare_opencv_dnn.normalize_result(
            {"result": [response_line("a"), response_line("b")]}
        )
        opencv_lines = compare_opencv_dnn.normalize_result(
            {"result": [response_line("a", 8.0)]}
        )
        comparison = compare_opencv_dnn.compare_results(c_lines, opencv_lines, 5.0)
        self.assertFalse(comparison["line_counts_match"])
        self.assertFalse(comparison["normalized_text_match"])
        self.assertFalse(comparison["bounds_within_tolerance"])

    def test_result_comparison_ignores_quadrilateral_starting_point(self) -> None:
        c_line = response_line("same")
        opencv_line = {
            "text": "same",
            "x1": c_line["x3"],
            "y1": c_line["y3"],
            "x2": c_line["x4"],
            "y2": c_line["y4"],
            "x3": c_line["x1"],
            "y3": c_line["y1"],
            "x4": c_line["x2"],
            "y4": c_line["y2"],
        }
        comparison = compare_opencv_dnn.compare_results(
            compare_opencv_dnn.normalize_result({"result": [c_line]}),
            compare_opencv_dnn.normalize_result({"result": [opencv_line]}),
            0.0,
        )
        self.assertTrue(comparison["bounds_within_tolerance"])

    def test_timing_fields_are_normalized(self) -> None:
        self.assertEqual(
            compare_opencv_dnn.server_total_ms(
                {"timing_ms": {"server_total": 12.5}}, "c"
            ),
            12.5,
        )
        self.assertEqual(
            compare_opencv_dnn.server_total_ms(
                {"timing": {"server_total_ms": 14.5}}, "opencv"
            ),
            14.5,
        )

    def test_opencv_config_contract_rejects_changed_threshold(self) -> None:
        config = {
            "enable_classifier": True,
            "limit_side_len": 960,
            "det_db_threshold": 0.3,
            "det_db_box_threshold": 0.6,
            "det_db_unclip_ratio": 1.6,
            "det_use_dilation": False,
            "cls_threshold": 0.9,
            "cls_batch_size": 8,
            "rec_batch_size": 8,
            "rec_concurrency": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "http-service.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertTrue(
                compare_opencv_dnn.validate_opencv_config(path)[
                    "matches_comparison_contract"
                ]
            )
            config["det_db_threshold"] = 0.4
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertFalse(
                compare_opencv_dnn.validate_opencv_config(path)[
                    "matches_comparison_contract"
                ]
            )

    def test_latency_summary_uses_interpolated_percentiles(self) -> None:
        summary = compare_opencv_dnn.summarize([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(summary["mean"], 2.5)
        self.assertEqual(summary["p50"], 2.5)
        self.assertEqual(summary["min"], 1.0)
        self.assertEqual(summary["max"], 4.0)


if __name__ == "__main__":
    unittest.main()
