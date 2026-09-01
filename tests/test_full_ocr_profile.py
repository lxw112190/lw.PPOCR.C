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
            command.append(str(target_width if target_width is not None else 320))
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
                self.assertEqual(report["rec_target_width"], 320)
                self.assertEqual(report["lines"], 16)

                parallel = report["parallel"]
                self.assertGreaterEqual(parallel["logical_processors"], 1)
                self.assertGreaterEqual(parallel["physical_cores"], 1)
                self.assertGreaterEqual(parallel["available_processors"], 1)
                self.assertGreaterEqual(parallel["smt_width"], 1)
                self.assertGreaterEqual(parallel["det_intra_cap"], 1)
                self.assertLessEqual(parallel["det_intra_cap"], 8)
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
                self.assertGreaterEqual(wall["output"], 0)
                self.assertGreaterEqual(
                    wall["line_workers"], wall["line_worker_critical"]
                )
                self.assertGreaterEqual(wall["total"], wall["det_graph"])
                self.assertGreaterEqual(wall["total"], wall["line_workers"])

                line_work = report["line_work_nanoseconds"]
                for stage in (
                    "cls_preprocess",
                    "cls_graph",
                    "rec_preprocess",
                    "rec_graph",
                    "rec_postprocess",
                ):
                    self.assertGreater(line_work[stage], 0, stage)
                self.assertGreaterEqual(line_work["cls_postprocess"], 0)

                operators = report["operators"]
                self.assertEqual(len(operators), 21)
                self.assertEqual(
                    {item["name"] for item in operators if item["invocations"] > 0},
                    EXPECTED_OPERATORS,
                )
                self.assertEqual(
                    sum(item["invocations"] for item in operators),
                    242 + 16 * (106 + 159),
                )
                self.assertGreater(report["graph_work_nanoseconds"], 0)
                self.assertAlmostEqual(
                    sum(item["percentage"] for item in operators), 100.0, places=3
                )

                rec_width = report["rec_width"]
                self.assertEqual(rec_width["samples"], report["lines"])
                self.assertGreater(rec_width["resized_width_sum"], 0)
                self.assertGreaterEqual(
                    rec_width["target_width_sum"], rec_width["resized_width_sum"]
                )
                self.assertGreater(rec_width["mean_resized_width"], 0.0)
                self.assertLessEqual(rec_width["mean_resized_width"], 320.0)
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

        serial_histogram = reports[0]["parallel"]["det_conv_thread_histogram"]
        self.assertGreater(serial_histogram[0], 0)
        self.assertEqual(sum(serial_histogram[1:]), 0)

    def test_profile_accepts_long_text_target_width(self) -> None:
        reports = [self.run_profile(1, 960), self.run_profile(4, 960)]
        for report in reports:
            self.assertEqual(report["rec_target_width"], 960)
            rec_width = report["rec_width"]
            self.assertEqual(rec_width["target_width_sum"], 9280)
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
    return parser.parse_args()


ARGUMENTS = parse_args()

if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
