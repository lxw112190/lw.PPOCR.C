from __future__ import annotations

import struct
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from converter.lwm_v0 import CHECKSUM_OFFSET, HEADER_SIZE, convert_cls_model, fnv1a64


ROOT = Path(__file__).resolve().parents[1]
CLS_MODEL = ROOT / "models" / "ppocrv6-tiny" / "cls.onnx"


class ConvertClsTests(unittest.TestCase):
    def test_conversion_is_deterministic_and_has_expected_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.lwm"
            second = Path(directory) / "second.lwm"
            first_info = convert_cls_model(CLS_MODEL, first)
            second_info = convert_cls_model(CLS_MODEL, second)
            first_bytes = first.read_bytes()
            second_bytes = second.read_bytes()

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first_info, second_info)
        self.assertEqual(first_info.tensor_count, 173)
        self.assertEqual(first_info.node_count, 106)
        self.assertEqual(first_info.input_count, 1)
        self.assertEqual(first_info.output_count, 1)
        self.assertEqual(first_info.weight_size, 965488)
        self.assertEqual(first_info.file_size, len(first_bytes))

        header = struct.unpack_from("<4sHH6I13Q3Q", first_bytes)
        self.assertEqual(header[0], b"LWM0")
        self.assertEqual(header[3], HEADER_SIZE)
        node_count = header[6]
        node_offset = header[12]
        op_counts = Counter(
            struct.unpack_from("<H", first_bytes, node_offset + index * 72)[0]
            for index in range(node_count)
        )
        self.assertEqual(sum(op_counts.values()), first_info.node_count)
        self.assertEqual(op_counts[1], 32)  # Conv; all 27 eligible BN nodes are folded.
        self.assertEqual(op_counts[7], 0)
        self.assertEqual(op_counts[8], 3)  # GlobalAveragePool -> ReduceMean.
        self.assertEqual(op_counts[16], 1)  # Static Reshape.

        checksum_copy = bytearray(first_bytes)
        checksum = struct.unpack_from("<Q", checksum_copy, CHECKSUM_OFFSET)[0]
        struct.pack_into("<Q", checksum_copy, CHECKSUM_OFFSET, 0)
        self.assertEqual(fnv1a64(checksum_copy), checksum)

    def test_different_model_is_rejected_before_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "cls.onnx"
            output_path = Path(directory) / "cls.lwm"
            input_path.write_bytes(CLS_MODEL.read_bytes() + b"\0")
            with self.assertRaisesRegex(ValueError, "only supports the bundled"):
                convert_cls_model(input_path, output_path)
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
