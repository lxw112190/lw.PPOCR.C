import argparse
import subprocess

import numpy as np
from onnx import helper
from onnx.reference import ReferenceEvaluator


def parse_output(text: str) -> dict[str, np.ndarray]:
    results: dict[str, np.ndarray] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            raise AssertionError(f"invalid driver output line: {line!r}")
        name = fields[0]
        count = int(fields[1])
        values = np.asarray([float(value) for value in fields[2:]], dtype=np.float32)
        if values.size != count:
            raise AssertionError(
                f"{name}: declared {count} values but emitted {values.size}"
            )
        results[name] = values
    return results


def fill_values(
    count: int, multiplier: int, modulus: int, offset: int, divisor: float
) -> np.ndarray:
    return np.asarray(
        [(((index * multiplier) % modulus) - offset) / divisor for index in range(count)],
        dtype=np.float32,
    )


def conv2d_reference(
    input_values: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray | None,
    strides: tuple[int, int],
    dilations: tuple[int, int],
    pads: tuple[int, int, int, int],
    groups: int,
) -> np.ndarray:
    input_names = ["input", "weights"]
    feeds = {"input": input_values, "weights": weights}
    if bias is not None:
        input_names.append("bias")
        feeds["bias"] = bias
    node = helper.make_node(
        "Conv",
        input_names,
        ["output"],
        kernel_shape=list(weights.shape[2:]),
        strides=list(strides),
        dilations=list(dilations),
        pads=list(pads),
        group=groups,
    )
    return ReferenceEvaluator(node, opsets={"": 11}).run(None, feeds)[0]


def expected_results() -> dict[str, np.ndarray]:
    normal_input = fill_values(80, 5, 19, 9, 4.0).reshape(2, 2, 4, 5)
    normal_weights = fill_values(54, 7, 17, 8, 6.0).reshape(3, 2, 3, 3)
    normal_bias = np.asarray([0.25, -0.5, 1.0], dtype=np.float32)
    grouped_input = fill_values(64, 3, 23, 11, 5.0).reshape(1, 4, 4, 4)
    grouped_weights = fill_values(108, 11, 29, 14, 7.0).reshape(6, 2, 3, 3)
    depthwise_input = fill_values(60, 13, 31, 15, 8.0).reshape(1, 3, 4, 5)
    depthwise_weights = fill_values(18, 5, 13, 6, 5.0).reshape(3, 1, 3, 2)
    asymmetric_input = fill_values(6, 3, 11, 5, 4.0).reshape(1, 1, 2, 3)
    asymmetric_weights = fill_values(4, 5, 13, 6, 3.0).reshape(1, 1, 2, 2)
    pointwise_input = fill_values(80, 7, 19, 9, 5.0).reshape(2, 4, 2, 5)
    pointwise_weights = fill_values(12, 11, 23, 11, 6.0).reshape(6, 2, 1, 1)
    pointwise_bias = np.asarray(
        [0.25, -0.5, 1.0, -1.25, 0.75, 0.5], dtype=np.float32
    )
    batch_norm_input = fill_values(24, 7, 21, 10, 4.0).reshape(2, 3, 2, 2)
    scale = np.asarray([1.5, -0.75, 0.25], dtype=np.float32).reshape(1, 3, 1, 1)
    bias = np.asarray([0.1, 0.5, -1.0], dtype=np.float32).reshape(1, 3, 1, 1)
    mean = np.asarray([-0.25, 1.0, 0.5], dtype=np.float32).reshape(1, 3, 1, 1)
    variance = np.asarray([0.5, 2.0, 0.25], dtype=np.float32).reshape(1, 3, 1, 1)
    batch_norm_node = helper.make_node(
        "BatchNormalization",
        ["input", "scale", "bias", "mean", "variance"],
        ["output"],
        epsilon=1.0e-5,
        momentum=0.9,
    )
    batch_norm = ReferenceEvaluator(
        batch_norm_node, opsets={"": 11}
    ).run(
        None,
        {
            "input": batch_norm_input,
            "scale": scale.ravel(),
            "bias": bias.ravel(),
            "mean": mean.ravel(),
            "variance": variance.ravel(),
        },
    )[0]
    return {
        "conv": conv2d_reference(
            normal_input, normal_weights, normal_bias,
            (2, 2), (1, 1), (1, 1, 1, 1), 1,
        ).ravel(),
        "grouped_conv": conv2d_reference(
            grouped_input, grouped_weights, None,
            (1, 1), (1, 1), (1, 1, 1, 1), 2,
        ).ravel(),
        "depthwise_conv": conv2d_reference(
            depthwise_input, depthwise_weights, None,
            (1, 1), (1, 2), (1, 1, 1, 1), 3,
        ).ravel(),
        "asymmetric_conv": conv2d_reference(
            asymmetric_input, asymmetric_weights, None,
            (1, 2), (2, 1), (2, 1, 1, 2), 1,
        ).ravel(),
        "grouped_pointwise_conv": conv2d_reference(
            pointwise_input, pointwise_weights, pointwise_bias,
            (1, 1), (1, 1), (0, 0, 0, 0), 2,
        ).ravel(),
        "batch_norm": batch_norm.ravel(),
        "batch_norm_in_place": batch_norm.ravel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    args = parser.parse_args()
    completed = subprocess.run(
        [args.driver], check=True, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    actual = parse_output(completed.stdout)
    expected = expected_results()
    if actual.keys() != expected.keys():
        raise AssertionError(
            f"result names differ: actual={sorted(actual)}, expected={sorted(expected)}"
        )
    for name, expected_values in expected.items():
        np.testing.assert_allclose(
            actual[name], expected_values, rtol=2.0e-5, atol=3.0e-6,
            err_msg=name,
        )
    print(f"validated {len(expected)} Conv/BN results against NumPy")


if __name__ == "__main__":
    main()
