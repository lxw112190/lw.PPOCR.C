from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


HORIZONTAL_BOX = np.asarray(
    [[3.2, 2.4], [24.6, 1.2], [26.1, 11.8], [2.1, 13.4]], dtype=np.float32
)
VERTICAL_BOX = np.asarray(
    [[8.0, 1.0], [14.0, 2.0], [12.0, 16.0], [6.0, 15.0]], dtype=np.float32
)

EXPECTED_TEXT = [
    "纯臻营养护发素",
    "产品信息/参数",
    "(45元/每公斤，100公斤起订）",
    "每瓶22元，1000瓶起订)",
    "【品牌】：代加工方式/OEMODM",
    "【品名】：纯臻营养护发素",
    "【产品编号】：YM-X-3011",
    "ODM OEM",
    "【净含量】:220ml",
    "【适用人群】：适合所有肤质",
    "【主要成分】：鲸蜡硬脂醇、燕麦β-葡聚",
    "糖、椰油酰胺丙基甜菜碱、泛醌",
    "(成品包材)",
    "【主要功能】：可紧致头发磷层，从而达到",
    "即时持久改善头发光泽的效果，给干燥的头",
    "发足够的滋养",
]


def make_source(width: int, height: int, stride: int) -> tuple[np.ndarray, bytes]:
    image = np.empty((height, width, 3), dtype=np.uint8)
    raw = bytearray([0xD3] * (stride * height))
    for y in range(height):
        for x in range(width):
            pixel = np.asarray(
                [(x * 17 + y * 7 + 11) % 256,
                 (x * 3 + y * 29 + 37) % 256,
                 (x * 23 + y * 5 + 71) % 256],
                dtype=np.uint8,
            )
            image[y, x] = pixel
            raw[y * stride + x * 3 : y * stride + x * 3 + 3] = pixel.tobytes()
    return image, bytes(raw)


def crop_reference(source: np.ndarray, points: np.ndarray) -> np.ndarray:
    width = max(1, int(np.floor(np.linalg.norm(points[0] - points[1]) + 0.5)))
    height = max(1, int(np.floor(np.linalg.norm(points[0] - points[3]) + 0.5)))
    destination = np.asarray(
        [[0.0, 0.0], [float(width), 0.0],
         [float(width), float(height)], [0.0, float(height)]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, destination)
    crop = cv2.warpPerspective(
        source, transform, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    if height >= width * 1.5:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


class OcrPipelineReferenceTest(unittest.TestCase):
    driver: Path
    detector: Path
    classifier: Path
    recognizer: Path
    dictionary: Path
    sample: Path

    def run_driver(self, args: list[str], timeout: int = 900) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.driver), *args], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout
        )

    def test_crop_matches_opencv_reference(self) -> None:
        width, height, stride = 32, 18, 100
        image, raw = make_source(width, height, stride)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bgr"
            source.write_bytes(raw)
            for vertical, points in ((0, HORIZONTAL_BOX), (1, VERTICAL_BOX)):
                with self.subTest(vertical=vertical):
                    output = root / f"crop-{vertical}.bgr"
                    completed = self.run_driver([
                        "crop", str(source), str(width), str(height), str(stride),
                        str(output), str(vertical)
                    ])
                    self.assertEqual(
                        completed.returncode, 0,
                        completed.stdout + completed.stderr,
                    )
                    expected = crop_reference(image, points)
                    actual = np.fromfile(output, dtype=np.uint8).reshape(expected.shape)
                    difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
                    self.assertLessEqual(int(difference.max()), 6)
                    self.assertLess(float(difference.mean()), 0.8)

    def test_public_full_ocr_matches_real_image_golden(self) -> None:
        rgb = np.asarray(Image.open(self.sample).convert("RGB"), dtype=np.uint8)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.bgr"
            source.write_bytes(bgr.tobytes())
            completed = self.run_driver([
                "pipeline", str(self.detector), str(self.classifier),
                str(self.recognizer), str(self.dictionary), str(source),
                str(bgr.shape[1]), str(bgr.shape[0]), str(bgr.shape[1] * 3), "1"
            ])
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        header = re.search(
            r"lines=(\d+) detected=(\d+) text_bytes=(\d+) "
            r"resized=(\d+)x(\d+) cls=(\d+)", completed.stdout
        )
        self.assertIsNotNone(header, completed.stdout)
        assert header is not None
        self.assertEqual(tuple(int(header.group(i)) for i in (1, 2)), (16, 16))
        self.assertGreater(int(header.group(3)), 0)
        self.assertEqual(tuple(int(header.group(i)) for i in (4, 5, 6)), (320, 320, 1))
        lines = re.findall(
            r"line=(\d+) det=([^ ]+) rec=([^ ]+) cls_label=(\d+) "
            r"cls=([^ ]+) rotation=(\d+) point=([^ ]+) text=([^\r\n]*)",
            completed.stdout,
        )
        self.assertEqual(len(lines), len(EXPECTED_TEXT), completed.stdout)
        for index, values in enumerate(lines):
            self.assertEqual(int(values[0]), index)
            self.assertGreaterEqual(float(values[1]), 0.6)
            self.assertGreaterEqual(float(values[2]), 0.85)
            self.assertGreaterEqual(float(values[4]), 0.9)
            self.assertEqual(values[7], EXPECTED_TEXT[index])
            if index == 7:
                self.assertEqual((int(values[3]), int(values[5])), (1, 180))
            else:
                self.assertEqual(int(values[5]), 0)

    def test_public_full_ocr_without_direction_classifier(self) -> None:
        rgb = np.asarray(Image.open(self.sample).convert("RGB"), dtype=np.uint8)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sample.bgr"
            source.write_bytes(bgr.tobytes())
            completed = self.run_driver([
                "pipeline", str(self.detector), str(self.classifier),
                str(self.recognizer), str(self.dictionary), str(source),
                str(bgr.shape[1]), str(bgr.shape[0]), str(bgr.shape[1] * 3), "0"
            ])
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        header = re.search(
            r"lines=(\d+) detected=(\d+) text_bytes=(\d+) "
            r"resized=(\d+)x(\d+) cls=(\d+)", completed.stdout
        )
        self.assertIsNotNone(header, completed.stdout)
        assert header is not None
        self.assertEqual(tuple(int(header.group(i)) for i in (1, 2)), (16, 16))
        self.assertEqual(int(header.group(6)), 0)
        metadata = re.findall(
            r"cls_label=(\d+) cls=([^ ]+) rotation=(\d+)", completed.stdout
        )
        self.assertEqual(len(metadata), 16, completed.stdout)
        for label, score, rotation in metadata:
            self.assertEqual((int(label), float(score), int(rotation)), (0, 0.0, 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    OcrPipelineReferenceTest.driver = arguments.driver
    OcrPipelineReferenceTest.detector = arguments.detector
    OcrPipelineReferenceTest.classifier = arguments.classifier
    OcrPipelineReferenceTest.recognizer = arguments.recognizer
    OcrPipelineReferenceTest.dictionary = arguments.dictionary
    OcrPipelineReferenceTest.sample = arguments.sample
    unittest.main(argv=[__file__], verbosity=2)
