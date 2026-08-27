import pathlib
import unittest

from converter.analyze_conv_shapes import (
    _conv_channels_and_kind,
    analyze_conv_distribution,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ConvShapeAnalysisTests(unittest.TestCase):
    def test_grouped_weight_shape_restores_full_input_channels(self) -> None:
        self.assertEqual(
            _conv_channels_and_kind([96, 16, 1, 1], 4),
            (64, 96, 1, 1, "1x1"),
        )

    def test_depthwise_shape_is_classified_from_full_input_channels(self) -> None:
        self.assertEqual(
            _conv_channels_and_kind([128, 1, 3, 3], 128),
            (128, 128, 3, 3, "dw"),
        )

    def test_depthwise_multiplier_is_still_depthwise(self) -> None:
        self.assertEqual(
            _conv_channels_and_kind([64, 1, 5, 5], 32),
            (32, 64, 5, 5, "dw"),
        )

    def test_bundled_det_model_reports_full_depthwise_channels(self) -> None:
        distribution = analyze_conv_distribution(
            "det",
            ROOT / "models" / "ppocrv6-tiny" / "det.onnx",
            [1, 3, 640, 640],
        )
        depthwise_rows = [row for row in distribution["rows"] if row["kind"] == "dw"]
        self.assertEqual(len(depthwise_rows), 17)
        self.assertTrue(depthwise_rows)
        self.assertTrue(all(row["cin"] == row["g"] for row in depthwise_rows))
        self.assertTrue(all(row["cin"] != 1 for row in depthwise_rows))


if __name__ == "__main__":
    unittest.main()
