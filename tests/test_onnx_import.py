"""Native ONNX import: official assets, ORT numerics and bounded rejection.

ONNX/ORT are test dependencies only; the C runtime does not link either.
Fixtures are generated in a temporary directory and never enter Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def run(command: list[str], *, success: bool = True) -> str:
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=300)
    if success != (result.returncode == 0):
        raise AssertionError(f"{command}: exit={result.returncode}\n"
                             f"{result.stdout}\n{result.stderr}")
    return result.stdout


def fixture(nodes=None, initializers=None):
    graph = helper.make_graph(
        nodes or [helper.make_node("Relu", ["input"], ["output"])],
        "bounded-native-import",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 48, "width"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 48, "width"])],
        initializer=initializers or [],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    # The public model loader has a minimum container size of 160 bytes.
    model.doc_string = "native parser fixture " * 10
    return model


def rejection_tests(inspect: Path, directory: Path) -> int:
    count = 0

    def check(name: str, payload: bytes, accepted: bool = False):
        nonlocal count
        path = directory / (name + ".onnx")
        path.write_bytes(payload)
        run([str(inspect), str(path)], success=accepted)
        count += 1

    base = fixture()
    check("valid", base.SerializeToString(), True)
    for size in (0, 1, 159, 160, len(base.SerializeToString()) - 1):
        check(f"truncated-{size}", base.SerializeToString()[:size])
    check("varint-overflow", b"\x80" * 10 + b"\x00" * 200)
    check("wire-group", b"\x0b" + b"\x00" * 200)
    check("length-overflow", b"\x3a" + b"\xff" * 9 + b"\x01" + b"\x00" * 200)
    custom = fixture()
    custom.graph.node[0].domain = "custom.domain"
    check("custom-domain", custom.SerializeToString())
    unsupported = fixture([helper.make_node("UnknownOp", ["input"], ["output"])])
    check("unsupported-op", unsupported.SerializeToString())
    for operator in ("Conv", "BatchNormalization", "Mul", "Resize", "Reshape"):
        model = fixture([helper.make_node(operator, ["input"], ["output"])])
        check("arity-" + operator, model.SerializeToString())
    invalid = fixture()
    invalid.graph.node[0].attribute.append(helper.make_attribute("unknown", 1))
    check("unknown-attribute", invalid.SerializeToString())
    external = fixture(initializers=[
        helper.make_tensor("external", TensorProto.FLOAT, [1], [0.0])
    ])
    external.graph.initializer[0].ClearField("float_data")
    external.graph.initializer[0].data_location = TensorProto.EXTERNAL
    external.graph.initializer[0].external_data.add(key="location", value="../outside.bin")
    check("external-data", external.SerializeToString())
    for dtype, values, accepted in (
        (TensorProto.UINT8, [0, 255], True),
        (TensorProto.INT32, [-2147483648, 2147483647], True),
    ):
        tensor = helper.make_tensor("unused", dtype, [2], values)
        # Exercise packed typed payloads, not raw_data.
        check("typed-" + str(dtype), fixture(initializers=[tensor]).SerializeToString(), accepted)
    overflow = helper.make_tensor("unused", TensorProto.UINT8, [1], [0])
    overflow.int32_data[0] = 256
    check("uint8-overflow", fixture(initializers=[overflow]).SerializeToString())
    bad_float = helper.make_tensor("unused", TensorProto.FLOAT, [1], [float("nan")])
    check("nonfinite", fixture(initializers=[bad_float]).SerializeToString())
    model = fixture()
    model.opset_import[0].version = 999
    check("unknown-opset", model.SerializeToString())
    model = fixture()
    model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 1
    check("non-rgb-input", model.SerializeToString())
    model = fixture()
    model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 4
    check("wrong-output-shape", model.SerializeToString())
    model = fixture()
    model.graph.output[0].type.tensor_type.elem_type = TensorProto.INT64
    check("wrong-output-type", model.SerializeToString())
    duplicate = fixture()
    duplicate.graph.node.add().CopyFrom(duplicate.graph.node[0])
    check("duplicate-producer", duplicate.SerializeToString())
    model = fixture()
    for _ in range(4096):
        model.graph.node.add().CopyFrom(model.graph.node[0])
    check("node-budget", model.SerializeToString())
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", type=Path, required=True)
    parser.add_argument("--fine-rec-widths", action="store_true")
    parser.add_argument("--rec-driver", type=Path, required=True)
    parser.add_argument("--ocr-driver", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = {"schema_version": 1, "models": {}, "rec_reference": []}
    with tempfile.TemporaryDirectory(prefix="lw-onnx-") as temporary:
        directory = Path(temporary)
        report["bounded_cases"] = rejection_tests(args.inspect, directory)
        cls = args.root / "models/ppocrv6-tiny/cls.onnx"
        report["models"]["cls"] = json.loads(run([str(args.inspect), str(cls)]))
        for variant in ("tiny", "small", "medium"):
            model_dir = args.root / "models" / ("ppocrv6-" + variant)
            for component in ("det", "rec"):
                model = model_dir / (component + ".onnx")
                report["models"][variant + "-" + component] = json.loads(
                    run([str(args.inspect), str(model)]))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            reference = ort.InferenceSession(str(model_dir / "rec.onnx"), options,
                                            providers=["CPUExecutionProvider"])
            for width in (17, 192, 960):
                n = 3 * 48 * width
                values = ((((np.arange(n, dtype=np.int64) * 17) % 257) - 128)
                          .astype(np.float32) / np.float32(127)).reshape(1, 3, 48, width)
                expected = reference.run(None, {reference.get_inputs()[0].name: values})[0]
                output = directory / f"{variant}-{width}.f32"
                run([str(args.rec_driver), str(model_dir / "rec.onnx"), str(width), str(output)])
                actual = np.fromfile(output, dtype="<f4").reshape(expected.shape)
                # ORT folds BN and may use FMA; our importer preserves raw BN.
                # Bound both relative error and the absolute probability error.
                np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=2e-5)
                assert float(np.abs(actual - expected).max()) <= 1e-3
                report["rec_reference"].append({
                    "variant": variant, "width": width, "shape": list(actual.shape),
                    "max_abs": float(np.abs(actual - expected).max()),
                })
                print(f"ONNX {variant} width={width} max_abs="
                      f"{report['rec_reference'][-1]['max_abs']:.8g}", flush=True)
            dictionary = (model_dir / "ppocr_keys.txt" if variant == "tiny" else
                          args.root / "models/ppocrv6-shared/PP-OCRv6_small_rec_dict.txt")
            text = run([str(args.ocr_driver), str(model_dir / "det.onnx"), str(cls),
                        str(model_dir / "rec.onnx"), str(dictionary), str(args.sample)])
            assert "lines=16" in text or "lines: 16" in text, text
            texts = re.findall(r"^\d+ text=(.*?) rec=", text, re.MULTILINE)
            digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()
            prefix = "native-fine-" if args.fine_rec_widths and variant == "small" else "web-"
            golden = json.loads((args.root / "ci" / (prefix + "ppocrv6-" + variant + ".json"))
                                .read_text(encoding="utf-8"))
            assert len(texts) == golden["expected_line_count"], text
            assert digest == golden["expected_text_sha256"], (variant, digest, texts)
            report["models"][variant + "-ocr"] = {
                "text_sha256": digest,
                "line_count": 16,
            }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
