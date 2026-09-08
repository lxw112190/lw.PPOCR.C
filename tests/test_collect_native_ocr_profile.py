from __future__ import annotations

import unittest

from tools.collect_native_ocr_profile import build_parser


class CollectNativeOcrProfileTests(unittest.TestCase):
    @staticmethod
    def parser_args(*extra: str) -> list[str]:
        args = [
            "--profile-driver", "profile",
            "--benchmark-driver", "benchmark",
            "--det", "det",
            "--cls", "cls",
            "--rec", "rec",
            "--dictionary", "dictionary",
            "--image", "image",
            "--output", "out",
        ]
        args.extend(extra)
        return args

    def test_summary_name_preserves_arm64_default(self) -> None:
        args = build_parser().parse_args(self.parser_args())
        self.assertEqual(args.summary_name, "arm64-profile-summary.json")

    def test_summary_name_can_identify_other_architectures(self) -> None:
        args = build_parser().parse_args(
            self.parser_args("--summary-name", "x64-profile-summary.json")
        )
        self.assertEqual(args.summary_name, "x64-profile-summary.json")


if __name__ == "__main__":
    unittest.main()
