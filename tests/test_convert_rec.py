from __future__ import annotations

import struct
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from converter.lwm_v0 import CHECKSUM_OFFSET, HEADER_SIZE, convert_rec_model, fnv1a64


ROOT = Path(__file__).resolve().parents[1]
REC_MODEL = ROOT / "models" / "ppocrv6-tiny" / "rec.onnx"


class ConvertRecTests(unittest.TestCase):
    def test_conversion_is_deterministic_and_has_expected_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.lwm"
            second = Path(directory) / "second.lwm"
            first_info = convert_rec_model(REC_MODEL, first)
            second_info = convert_rec_model(REC_MODEL, second)
            first_bytes = first.read_bytes()
            second_bytes = second.read_bytes()

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first_info, second_info)
        self.assertEqual(first_info.tensor_count, 274)
        self.assertEqual(first_info.node_count, 159)
        self.assertEqual(first_info.input_count, 1)
        self.assertEqual(first_info.output_count, 1)
        self.assertEqual(first_info.file_size, len(first_bytes))

        header = struct.unpack_from("<4sHH6I13Q3Q", first_bytes)
        self.assertEqual(header[0], b"LWM0")
        self.assertEqual(header[3], HEADER_SIZE)
        node_count = header[6]
        node_offset = header[12]
        op_counts = Counter(struct.unpack_from("<H", first_bytes, node_offset + index * 72)[0] for index in range(node_count))
        self.assertEqual(sum(op_counts.values()), 159)
        self.assertEqual(op_counts[1], 37)  # Conv
        # Two source pairs are safely folded; two unshareable BN nodes remain.
        self.assertEqual(op_counts[7], 2)

        checksum_copy = bytearray(first_bytes)
        checksum = struct.unpack_from("<Q", checksum_copy, CHECKSUM_OFFSET)[0]
        struct.pack_into("<Q", checksum_copy, CHECKSUM_OFFSET, 0)
        self.assertEqual(fnv1a64(checksum_copy), checksum)

    def test_different_model_is_rejected_before_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "rec.onnx"
            output_path = Path(directory) / "rec.lwm"
            input_path.write_bytes(REC_MODEL.read_bytes() + b"\0")
            with self.assertRaisesRegex(ValueError, "only supports the bundled"):
                convert_rec_model(input_path, output_path)
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
