from __future__ import annotations

import argparse
import json
import subprocess
import unittest


EXPECTED_OPERATORS = {
    "Conv",
    "Add",
    "Mul",
    "Div",
    "Erf",
    "HardSigmoid",
    "BatchNormalization",
    "ReduceMean",
    "Relu",
    "AveragePool",
    "Squeeze",
    "Transpose",
    "Unsqueeze",
    "MatMul",
    "Softmax",
    "Reshape",
    "Concat",
    "ConvTranspose",
    "MaxPool",
    "Resize",
    "Sigmoid",
}


class FullOcrProfileTest(unittest.TestCase):
    def run_profile(
        self,
        workers: int,
        target_width: int | None = None,
        det_intra_op_threads: int | None = None,
    ) -> dict:
        command = [
            ARGUMENTS.driver,
            ARGUMENTS.det_model,
            ARGUMENTS.cls_model,
            ARGUMENTS.rec_model,
            ARGUMENTS.dictionary,
            ARGUMENTS.image,
            "1",
            str(workers),
        ]
        if target_width is not None or det_intra_op_threads is not None:
            command.append(str(target_width if target_width is not None else 960))
        if det_intra_op_threads is not None:
            command.append(str(det_intra_op_threads))
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return json.loads(completed.stdout)

    def test_profile_covers_full_pipeline_and_parallel_workers(self) -> None:
        reports = [self.run_profile(1), self.run_profile(4)]
        for report, workers in zip(reports, (1, 4)):
            with self.subTest(workers=workers):
                self.assertEqual(report["schema_version"], 1)
                self.assertEqual(report["iterations"], 1)
                self.assertEqual(report["workers"], workers)
                self.assertEqual(report["rec_target_width"], 960)
                self.assertEqual(report["lines"], 16)

                parallel = report["parallel"]
                self.assertGreaterEqual(parallel["logical_processors"], 1)
                self.assertGreaterEqual(parallel["physical_cores"], 1)
                self.assertGreaterEqual(parallel["available_processors"], 1)
                self.assertGreaterEqual(parallel["smt_width"], 1)
                self.assertGreaterEqual(parallel["det_intra_cap"], 1)
                self.assertLessEqual(parallel["det_intra_cap"], ARGUMENTS.thread_budget)
                if ARGUMENTS.thread_budget == 16:
                    self.assertEqual(
                        parallel["det_intra_cap"], min(16, parallel["available_processors"])
                    )
                self.assertGreaterEqual(parallel["det_intra_actual"], 1)
                self.assertLessEqual(
                    parallel["det_intra_actual"], parallel["det_intra_cap"]
                )
                self.assertEqual(len(parallel["det_conv_thread_histogram"]), 16)
                self.assertEqual(
                    parallel["serial_conv_invocations"],
                    parallel["det_conv_thread_histogram"][0],
                )
                self.assertEqual(
                    parallel["parallel_conv_invocations"],
                    sum(parallel["det_conv_thread_histogram"][1:]),
                )
                self.assertEqual(
                    len(parallel["det_conv_transpose_thread_histogram"]), 16
                )
                self.assertEqual(
                    parallel["serial_conv_transpose_invocations"],
                    parallel["det_conv_transpose_thread_histogram"][0],
                )
                self.assertEqual(
                    parallel["parallel_conv_transpose_invocations"],
                    sum(parallel["det_conv_transpose_thread_histogram"][1:]),
                )
                self.assertGreater(
                    sum(parallel["det_conv_transpose_thread_histogram"]), 0
                )

                wall = report["wall_nanoseconds"]
                for stage in (
                    "total",
                    "det_preprocess",
                    "det_graph",
                    "det_postprocess",
                    "crop",
                    "line_workers",
                    "line_worker_critical",
                ):
                    self.assertGreater(wall[stage], 0, stage)
                # These sections can legitimately complete within one clock tick.
                self.assertGreaterEqual(wall["line_dispatch_overhead"], 0)
                self.assertGreaterEqual(wall["crop_setup"], 0)
                self.assertGreaterEqual(wall["det_unclip"], 0)
                self.assertGreaterEqual(wall["output"], 0)
                self.assertGreaterEqual(
                    wall["line_workers"], wall["line_worker_critical"]
                )
                self.assertGreaterEqual(wall["total"], wall["det_graph"])
                self.assertGreaterEqual(wall["total"], wall["line_workers"])

                line_work = report["line_work_nanoseconds"]
                session_cache = report["rec_session_cache"]
                coverage = report["rec_backend_coverage"]
                self.assertEqual(
                    coverage["compiled_lines"] + coverage["canonical_lines"],
                    report["lines"],
                )
                if ARGUMENTS.expect_compiled_rec:
                    self.assertEqual(coverage["compiled_lines"], report["lines"])
                    self.assertEqual(coverage["canonical_lines"], 0)
                    self.assertEqual(session_cache["reconfigurations"], 0)
                self.assertEqual(
                    session_cache["hits"] + session_cache["misses"],
                    report["lines"],
                )
                self.assertEqual(
                    session_cache["misses"],
                    session_cache["reconfigurations"],
                )
                if ARGUMENTS.expect_resident:
                    self.assertEqual(session_cache["misses"], 0)
                    self.assertEqual(session_cache["reconfigurations"], 0)
                for stage in (
                    "cls_preprocess",
                    "cls_graph",
                    "rec_preprocess",
                    "rec_graph",
                    "rec_postprocess",
                ):
                    self.assertGreater(line_work[stage], 0, stage)
                self.assertGreaterEqual(line_work["cls_postprocess"], 0)

                implementation_paths = report["implementation_paths"]
                operators = report["operators"]
                self.assertEqual(len(operators), 21)
                observed_operators = {
                    item["name"] for item in operators if item["invocations"] > 0
                }
                if coverage["canonical_lines"] > 0:
                    self.assertEqual(observed_operators, EXPECTED_OPERATORS)
                else:
                    self.assertTrue(observed_operators <= EXPECTED_OPERATORS)
                # Lines that reach adaptive width 960 run through the compiled
                # x64 REC backend and execute outside the per-operator profile
                # counters. Derive the backend row count from the greedy CTC
                # counter instead of hard-coding it: total invocations are the
                # DET nodes plus CLS nodes per line plus REC nodes per
                # canonical line.
                canonical_rows = (
                    implementation_paths["recognizer"]["ctc_greedy"]
                    // report["iterations"]
                )
                self.assertEqual(canonical_rows, coverage["canonical_lines"])
                # A compiled x64 CLS backend executes outside the per-operator
                # profile counters just like compiled REC lines do; detect it
                # from the classifier implementation-path counters (all zero
                # when every line took the compiled backend).
                cls_impl = implementation_paths["classifier"]
                cls_canonical_invocations = sum(
                    value for value in cls_impl.values() if isinstance(value, int)
                )
                cls_canonical_lines = (
                    report["lines"] if cls_canonical_invocations > 0 else 0
                )
                self.assertEqual(
                    sum(item["invocations"] for item in operators),
                    242 + cls_canonical_lines * 106 + canonical_rows * 159,
                )
                self.assertGreater(report["graph_work_nanoseconds"], 0)
                self.assertAlmostEqual(
                    sum(item["percentage"] for item in operators), 100.0, places=3
                )

                layout = report["layout"]
                self.assertGreater(layout["candidate_nodes"], 0)
                # The promoted NHWC DET backend selects its effective NHWC
                # nodes; the rest stay on the canonical fallback.
                self.assertGreater(layout["selected_nodes"], 0)
                self.assertEqual(
                    layout["fallback_nodes"],
                    layout["candidate_nodes"] - layout["selected_nodes"],
                )
                self.assertGreaterEqual(layout["transform_nanoseconds"], 0)
                self.assertGreaterEqual(layout["transform_invocations"], 0)
                self.assertGreaterEqual(layout["transform_bytes"], 0)
                det_backend_active = layout["selected_nodes"] > 0

                for component_name in ("detector", "classifier", "recognizer"):
                    binding = implementation_paths[component_name]["prepared_binding"]
                    # The compiled DET backend bypasses the executor's prepared
                    # bindings entirely, so its counters stay zero even when
                    # prepared execution is enabled for the other components.
                    component_backend = component_name == "detector" and det_backend_active
                    if component_name == "recognizer" and canonical_rows == 0:
                        component_backend = True
                    if ARGUMENTS.expect_prepared and not component_backend:
                        self.assertGreater(binding["lookups"], 0)
                        self.assertEqual(
                            binding["hits"] + binding["fallbacks"], binding["lookups"]
                        )
                    else:
                        self.assertEqual(binding["lookups"], 0)
                        self.assertEqual(binding["hits"], 0)
                        self.assertEqual(binding["fallbacks"], 0)
                prepared_execution = report["prepared_execution"]
                if ARGUMENTS.expect_prepared:
                    self.assertGreater(prepared_execution["prepared_nodes"], 0)
                    self.assertEqual(
                        prepared_execution["prepared_nodes"],
                        prepared_execution["conv1x1"] + prepared_execution["conv3x3"],
                    )
                    self.assertGreater(prepared_execution["generic_nodes"], 0)
                else:
                    self.assertEqual(prepared_execution["prepared_nodes"], 0)
                    self.assertEqual(prepared_execution["conv1x1"], 0)
                    self.assertEqual(prepared_execution["conv3x3"], 0)
                for component_name, component_key in (
                    ("detector", "det_invocations"),
                    ("classifier", "cls_invocations"),
                    ("recognizer", "rec_invocations"),
                ):
                    paths = implementation_paths[component_name]
                    conv = next(
                        item for item in operators if item["name"] == "Conv"
                    )[component_key]
                    if component_name == "detector" and det_backend_active:
                        # The compiled DET backend runs outside the executor's
                        # kernel-path counters, so those stay zero while the
                        # operator profile still records every Conv.
                        self.assertEqual(paths["packed_conv1x1"], 0)
                        self.assertEqual(paths["packed_conv3x3_stride2"], 0)
                        self.assertEqual(paths["unpacked_conv"], 0)
                        self.assertGreater(conv, 0)
                    else:
                        self.assertEqual(
                            paths["packed_conv1x1"]
                            + paths["packed_conv3x3_stride2"]
                            + paths["unpacked_conv"],
                            conv,
                        )
                    matmul = next(
                        item for item in operators if item["name"] == "MatMul"
                    )[component_key]
                    self.assertEqual(
                        paths["packed_matmul"] + paths["unpacked_matmul"],
                        matmul,
                    )
                # Lines that reach adaptive width 960 run the compiled x64 REC
                # backend, whose CTC projection bypasses these counters; the
                # canonical-path counters cover only the remaining lines.
                self.assertEqual(
                    implementation_paths["recognizer"]["ctc_greedy"],
                    canonical_rows * report["iterations"],
                )
                self.assertEqual(
                    implementation_paths["recognizer"]["ctc_packed_projection"]
                    + implementation_paths["recognizer"]["ctc_generic_projection"],
                    canonical_rows * report["iterations"],
                )

                rec_nodes = report["rec_nodes"]
                self.assertEqual(len(rec_nodes), 159 if canonical_rows > 0 else 0)
                self.assertEqual(
                    len({item["node"] for item in rec_nodes}), len(rec_nodes)
                )
                self.assertEqual(
                    sum(item["invocations"] for item in rec_nodes),
                    sum(item["rec_invocations"] for item in operators),
                )
                convolution_shape_count = 0
                for item in rec_nodes:
                    self.assertIn(
                        item["operation"],
                        EXPECTED_OPERATORS,
                    )
                    self.assertEqual(len(item["by_width"]), 8)
                    self.assertEqual(
                        sum(width["invocations"] for width in item["by_width"]),
                        item["invocations"],
                    )
                    self.assertEqual(
                        [width["max_width"] for width in item["by_width"]],
                        [192, 256, 320, 480, 640, 800, 960, None],
                    )
                    if item["operation"] in ("Conv", "ConvTranspose"):
                        convolution_shape_count += 1
                        for field in ("input", "weights", "output"):
                            self.assertIn(field, item)
                            self.assertEqual(len(item[field]), 4)
                            self.assertTrue(
                                all(dimension > 0 for dimension in item[field])
                            )
                        self.assertIn("group", item)
                        self.assertGreater(item["group"], 0)
                        for field, length in (
                            ("kernel", 2),
                            ("strides", 2),
                            ("dilations", 2),
                            ("pads", 4),
                        ):
                            self.assertIn(field, item)
                            self.assertEqual(len(item[field]), length)
                            self.assertTrue(all(value >= 0 for value in item[field]))
                    else:
                        for field in (
                            "input",
                            "weights",
                            "output",
                            "group",
                            "kernel",
                            "strides",
                            "dilations",
                            "pads",
                        ):
                            self.assertNotIn(field, item)
                if canonical_rows > 0:
                    self.assertGreater(convolution_shape_count, 0)
                rec_width = report["rec_width"]
                self.assertEqual(rec_width["samples"], report["lines"])
                self.assertGreater(rec_width["resized_width_sum"], 0)
                self.assertGreaterEqual(
                    rec_width["target_width_sum"], rec_width["resized_width_sum"]
                )
                self.assertGreater(rec_width["mean_resized_width"], 0.0)
                self.assertLessEqual(rec_width["mean_resized_width"], 960.0)
                self.assertGreaterEqual(rec_width["mean_padding_ratio"], 0.0)
                self.assertLessEqual(rec_width["mean_padding_ratio"], 1.0)
                histogram = rec_width["histogram"]
                self.assertEqual(
                    [item["max_width"] for item in histogram],
                    [192, 256, 320, 480, 640, 800, 960, None],
                )
                self.assertEqual(
                    sum(item["count"] for item in histogram), rec_width["samples"]
                )

                conv = next(item for item in operators if item["name"] == "Conv")
                self.assertEqual(report["conv_invocations"], conv["invocations"])
                self.assertEqual(
                    sum(item["invocations"] for item in report["conv_classes"]),
                    conv["invocations"],
                )
                self.assertEqual(
                    {item["name"] for item in report["conv_classes"]},
                    {
                        "Conv1x1",
                        "Conv3x3",
                        "Depthwise3x3",
                        "Stride2Conv3x3",
                        "OtherConv",
                    },
                )

                det_nodes = report["det_convolution_nodes"]
                self.assertGreater(len(det_nodes), 0)
                self.assertEqual(
                    sum(item["invocations"] for item in det_nodes),
                    next(
                        item["det_invocations"]
                        for item in operators
                        if item["name"] == "Conv"
                    )
                    + next(
                        item["det_invocations"]
                        for item in operators
                        if item["name"] == "ConvTranspose"
                    ),
                )
                self.assertEqual(
                    {item["operation"] for item in det_nodes},
                    {"Conv", "ConvTranspose"},
                )
                self.assertEqual(
                    len({item["node"] for item in det_nodes}), len(det_nodes)
                )
                for item in det_nodes:
                    self.assertEqual(len(item["input"]), 4)
                    self.assertEqual(len(item["weights"]), 4)
                    self.assertEqual(len(item["output"]), 4)
                    self.assertTrue(all(dimension > 0 for dimension in item["input"]))
                    self.assertTrue(all(dimension > 0 for dimension in item["weights"]))
                    self.assertTrue(all(dimension > 0 for dimension in item["output"]))

        self.assertEqual(
            [item["invocations"] for item in reports[0]["operators"]],
            [item["invocations"] for item in reports[1]["operators"]],
        )
        self.assertEqual(reports[0]["output_checksum"], reports[1]["output_checksum"])
        self.assertEqual(reports[0]["rec_width"], reports[1]["rec_width"])

    def test_det_intra_op_is_byte_identical(self) -> None:
        reports = [
            self.run_profile(1, 320, det_threads)
            for det_threads in (1, 2, 4, 8)
        ]
        reference = reports[0]
        for report, det_threads in zip(reports, (1, 2, 4, 8)):
            with self.subTest(det_threads=det_threads):
                parallel = report["parallel"]
                self.assertEqual(parallel["det_intra_cap"], det_threads)
                self.assertLessEqual(parallel["det_intra_actual"], det_threads)
                self.assertEqual(report["output_checksum"], reference["output_checksum"])
                self.assertEqual(report["lines"], reference["lines"])
                self.assertEqual(report["rec_width"], reference["rec_width"])
                self.assertEqual(
                    [item["invocations"] for item in report["operators"]],
                    [item["invocations"] for item in reference["operators"]],
                )
                transpose_histogram = parallel[
                    "det_conv_transpose_thread_histogram"
                ]
                if parallel["det_intra_actual"] > 1:
                    self.assertGreater(sum(transpose_histogram[1:]), 0)

        serial_histogram = reports[0]["parallel"]["det_conv_thread_histogram"]
        self.assertGreater(serial_histogram[0], 0)
        self.assertEqual(sum(serial_histogram[1:]), 0)
        serial_transpose_histogram = reports[0]["parallel"][
            "det_conv_transpose_thread_histogram"
        ]
        self.assertGreater(serial_transpose_histogram[0], 0)
        self.assertEqual(sum(serial_transpose_histogram[1:]), 0)

    def test_profile_accepts_long_text_target_width(self) -> None:
        reports = [self.run_profile(1, 960), self.run_profile(4, 960)]
        for report in reports:
            self.assertEqual(report["rec_target_width"], 960)
            rec_width = report["rec_width"]
            self.assertEqual(rec_width["target_width_sum"], 9920)
            self.assertLess(
                rec_width["target_width_sum"], 960 * report["lines"]
            )
            self.assertGreaterEqual(
                rec_width["target_width_sum"], rec_width["resized_width_sum"]
            )
            self.assertLessEqual(rec_width["mean_resized_width"], 960.0)
            self.assertEqual(
                [item["max_width"] for item in rec_width["histogram"]],
                [192, 256, 320, 480, 640, 800, 960, None],
            )
        self.assertEqual(reports[0]["output_checksum"], reports[1]["output_checksum"])
        self.assertEqual(reports[0]["rec_width"], reports[1]["rec_width"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    parser.add_argument("--det-model", required=True)
    parser.add_argument("--cls-model", required=True)
    parser.add_argument("--rec-model", required=True)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--thread-budget", type=int, choices=(8, 16), default=8)
    parser.add_argument("--expect-resident", action="store_true")
    parser.add_argument("--expect-compiled-rec", action="store_true")
    parser.add_argument("--expect-prepared", action="store_true")
    return parser.parse_args()


ARGUMENTS = parse_args()

if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
