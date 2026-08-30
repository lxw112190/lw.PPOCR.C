from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


HEADER_PATTERN = re.compile(
    r"lines=(\d+) detected=(\d+) text_bytes=(\d+) "
    r"resized=(\d+)x(\d+) cls=(\d+)"
)
LINE_PATTERN = re.compile(
    r"line=(\d+) det=([^ ]+) rec=([^ ]+) cls_label=(\d+) "
    r"cls=([^ ]+) rotation=(\d+) box=([^ ]+) text=([^\r\n]*)"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest_resize(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = rgb.shape[:2]
    x_indices = np.arange(width, dtype=np.uint64) * source_width // width
    y_indices = np.arange(height, dtype=np.uint64) * source_height // height
    return np.ascontiguousarray(rgb[y_indices[:, None], x_indices[None, :], :])


def transform_image(source: np.ndarray, transform: dict[str, object]) -> np.ndarray:
    kind = transform["type"]
    if kind == "identity":
        return np.ascontiguousarray(source)
    if kind == "nearest_resize":
        return nearest_resize(source, int(transform["width"]), int(transform["height"]))
    if kind == "canvas":
        width = int(transform["width"])
        height = int(transform["height"])
        offset_x, offset_y = (int(value) for value in transform["offset"])
        color = np.asarray(transform["color"], dtype=np.uint8)
        result = np.empty((height, width, 3), dtype=np.uint8)
        result[:, :] = color
        source_height, source_width = source.shape[:2]
        result[offset_y : offset_y + source_height, offset_x : offset_x + source_width] = source
        return result
    if kind == "rotate_90_counterclockwise":
        return np.ascontiguousarray(np.rot90(source, 1))
    if kind == "blank":
        width = int(transform["width"])
        height = int(transform["height"])
        color = np.asarray(transform["color"], dtype=np.uint8)
        result = np.empty((height, width, 3), dtype=np.uint8)
        result[:, :] = color
        return result
    raise AssertionError(f"unsupported corpus transform: {kind!r}")


def polygon_area(box: list[float]) -> float:
    points = np.asarray(box, dtype=np.float64).reshape(4, 2)
    return 0.5 * abs(
        float(
            np.dot(points[:, 0], np.roll(points[:, 1], -1))
            - np.dot(points[:, 1], np.roll(points[:, 0], -1))
        )
    )


class OcrGoldenCorpusTest(unittest.TestCase):
    driver: Path
    detector: Path
    classifier: Path
    recognizer: Path
    dictionary: Path
    detector_source: Path
    classifier_source: Path
    recognizer_source: Path
    sample: Path
    corpus: Path

    def test_versioned_full_ocr_corpus(self) -> None:
        manifest = json.loads(self.corpus.read_text(encoding="utf-8"))
        self.assertEqual(manifest.get("schema_version"), 1)
        self.assertEqual(manifest.get("detector_limit_side_length"), 320)
        self.assertEqual(sha256(self.sample), manifest.get("source_sha256"))
        self.assertEqual(
            {
                "detector": sha256(self.detector_source),
                "classifier": sha256(self.classifier_source),
                "recognizer": sha256(self.recognizer_source),
                "dictionary": sha256(self.dictionary),
            },
            manifest.get("model_sha256"),
        )

        text_profiles = manifest["text_profiles"]
        classification_profiles = manifest["classification_profiles"]
        box_profiles = manifest["box_profiles"]
        cases = manifest["cases"]
        self.assertGreaterEqual(len(cases), 7)
        self.assertEqual(len({case["name"] for case in cases}), len(cases))
        source = np.asarray(Image.open(self.sample).convert("RGB"), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            for case in cases:
                with self.subTest(case=case["name"]):
                    rgb = transform_image(source, case["transform"])
                    height, width = rgb.shape[:2]
                    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                    image_path = temporary / f"{case['name']}.bgr"
                    image_path.write_bytes(bgr.tobytes(order="C"))
                    use_classifier = int(bool(case["use_classifier"]))
                    completed = subprocess.run(
                        [
                            str(self.driver),
                            "golden",
                            str(self.detector),
                            str(self.classifier),
                            str(self.recognizer),
                            str(self.dictionary),
                            str(image_path),
                            str(width),
                            str(height),
                            str(width * 3),
                            str(use_classifier),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=300,
                    )
                    self.assertEqual(
                        completed.returncode, 0, completed.stdout + completed.stderr
                    )
                    header = HEADER_PATTERN.search(completed.stdout)
                    self.assertIsNotNone(header, completed.stdout)
                    assert header is not None
                    expected_text = text_profiles[case["text_profile"]]
                    self.assertEqual(int(header.group(1)), len(expected_text))
                    self.assertEqual(int(header.group(2)), int(case["detected_count"]))
                    self.assertEqual(
                        [int(header.group(4)), int(header.group(5))], case["detector_resize"]
                    )
                    self.assertEqual(int(header.group(6)), use_classifier)
                    if expected_text:
                        self.assertGreater(int(header.group(3)), len(expected_text))
                    else:
                        self.assertEqual(int(header.group(3)), 0)

                    parsed_lines = LINE_PATTERN.findall(completed.stdout)
                    self.assertEqual(len(parsed_lines), len(expected_text), completed.stdout)
                    profile = classification_profiles[case["classification_profile"]]
                    expected_labels = profile["labels"]
                    expected_rotations = profile["rotations"]
                    self.assertEqual(len(expected_labels), len(expected_text))
                    self.assertEqual(len(expected_rotations), len(expected_text))
                    box_profile = case.get("box_profile")
                    expected_boxes = (
                        box_profiles[box_profile] if box_profile is not None else None
                    )
                    if expected_boxes is not None:
                        self.assertEqual(len(expected_boxes), len(expected_text))
                    for index, values in enumerate(parsed_lines):
                        self.assertEqual(int(values[0]), index)
                        detection_score = float(values[1])
                        recognition_score = float(values[2])
                        classification_label = int(values[3])
                        classification_score = float(values[4])
                        rotation = int(values[5])
                        box = [float(value) for value in values[6].split(",")]
                        self.assertEqual(values[7], expected_text[index])
                        self.assertGreaterEqual(detection_score, float(case["min_det_score"]))
                        self.assertGreaterEqual(recognition_score, float(case["min_rec_score"]))
                        self.assertLessEqual(detection_score, 1.0)
                        self.assertLessEqual(recognition_score, 1.0)
                        self.assertEqual(classification_label, expected_labels[index])
                        self.assertEqual(rotation, expected_rotations[index])
                        self.assertEqual(len(box), 8)
                        for x, y in zip(box[0::2], box[1::2]):
                            self.assertTrue(-0.5 <= x <= width + 0.5, box)
                            self.assertTrue(-0.5 <= y <= height + 0.5, box)
                        self.assertGreater(polygon_area(box), 1.0)
                        if use_classifier:
                            self.assertGreaterEqual(
                                classification_score, float(case["min_cls_score"])
                            )
                            self.assertLessEqual(classification_score, 1.0)
                        else:
                            self.assertEqual(classification_score, 0.0)
                        if expected_boxes is not None:
                            difference = np.abs(
                                np.asarray(box) - np.asarray(expected_boxes[index])
                            )
                            self.assertLessEqual(
                                float(difference.max()), float(case["box_tolerance"])
                            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--recognizer", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--detector-source", type=Path, required=True)
    parser.add_argument("--classifier-source", type=Path, required=True)
    parser.add_argument("--recognizer-source", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    OcrGoldenCorpusTest.driver = arguments.driver
    OcrGoldenCorpusTest.detector = arguments.detector
    OcrGoldenCorpusTest.classifier = arguments.classifier
    OcrGoldenCorpusTest.recognizer = arguments.recognizer
    OcrGoldenCorpusTest.dictionary = arguments.dictionary
    OcrGoldenCorpusTest.detector_source = arguments.detector_source
    OcrGoldenCorpusTest.classifier_source = arguments.classifier_source
    OcrGoldenCorpusTest.recognizer_source = arguments.recognizer_source
    OcrGoldenCorpusTest.sample = arguments.sample
    OcrGoldenCorpusTest.corpus = arguments.corpus
    unittest.main(argv=[__file__], verbosity=2)
