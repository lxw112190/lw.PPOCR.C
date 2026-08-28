from __future__ import annotations

import argparse
import json
import subprocess
import unittest


class RecProfileTest(unittest.TestCase):
    def test_profile_covers_every_rec_node(self) -> None:
        completed = subprocess.run(
            [ARGUMENTS.driver, ARGUMENTS.model, "320", "2"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["width"], 320)
        self.assertEqual(report["iterations"], 2)
        operators = report["operators"]
        self.assertEqual(len(operators), 15)
        self.assertEqual(sum(item["invocations"] for item in operators), 159 * 2)
        self.assertGreater(sum(item["nanoseconds"] for item in operators), 0)
        self.assertEqual(
            {item["name"] for item in operators if item["invocations"] > 0},
            {
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
            },
        )
        conv_nodes = report["conv_nodes"]
        self.assertEqual(len(conv_nodes), 37)
        self.assertEqual(sum(item["invocations"] for item in conv_nodes), 37 * 2)
        self.assertTrue(all(item["nanoseconds"] > 0 for item in conv_nodes))
        self.assertTrue(all(len(item["input"]) == 4 for item in conv_nodes))
        self.assertTrue(all(len(item["weights"]) == 4 for item in conv_nodes))
        self.assertTrue(all(len(item["output"]) == 4 for item in conv_nodes))
        binary_nodes = report["binary_nodes"]
        self.assertEqual(len(binary_nodes), 87)
        self.assertEqual(sum(item["invocations"] for item in binary_nodes), 87 * 2)
        self.assertTrue(all(item["nanoseconds"] > 0 for item in binary_nodes))
        self.assertEqual(
            {item["operation"] for item in binary_nodes}, {"Add", "Mul", "Div"}
        )
        self.assertTrue(all(item["output"] for item in binary_nodes))
        self.assertTrue(
            all(isinstance(item["left_constant"], bool) and
                isinstance(item["right_constant"], bool)
                for item in binary_nodes)
        )
        matmul_nodes = report["matmul_nodes"]
        self.assertEqual(len(matmul_nodes), 2)
        self.assertEqual(sum(item["invocations"] for item in matmul_nodes), 2 * 2)
        self.assertTrue(all(item["nanoseconds"] > 0 for item in matmul_nodes))
        for item in matmul_nodes:
            self.assertGreater(item["batch_count"], 0)
            self.assertGreater(item["rows"], 0)
            self.assertGreater(item["inner_dimension"], 0)
            self.assertGreater(item["columns"], 0)
            self.assertEqual(item["input"][-2], item["rows"])
            self.assertEqual(item["input"][-1], item["inner_dimension"])
            self.assertEqual(item["weights"][0], item["inner_dimension"])
            self.assertEqual(item["weights"][1], item["columns"])
            self.assertEqual(item["output"][-2:], [item["rows"], item["columns"]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


ARGUMENTS = parse_args()

if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
