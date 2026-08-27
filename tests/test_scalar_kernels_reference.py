import argparse
import math
import subprocess

import numpy as np


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


def expected_results() -> dict[str, np.ndarray]:
    left = np.asarray(
        [(((index * 7) % 19) - 9) / 5.0 for index in range(24)],
        dtype=np.float32,
    ).reshape(2, 3, 4)
    add_right = np.asarray([0.25, -1.0, 2.0], dtype=np.float32).reshape(3, 1)
    mul_right = np.asarray([-2.0, 0.5, 3.0], dtype=np.float32).reshape(3, 1)
    div_right = np.asarray([0.5, -2.0, 4.0], dtype=np.float32).reshape(3, 1)
    activation_input = np.asarray(
        [-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0],
        dtype=np.float32,
    )
    softmax_input = np.asarray(
        [
            1000.0, -1000.0, 0.0, 50.0,
            1001.0, -999.0, 2.0, 48.0,
            999.0, -1002.0, -1.0, 51.0,
            -500.0, 500.0, 10.0, -10.0,
            -501.0, 501.0, 12.0, -9.0,
            -499.0, 499.0, 8.0, -11.0,
        ],
        dtype=np.float32,
    ).reshape(2, 3, 4)
    shifted = softmax_input - np.max(softmax_input, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    softmax = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    softmax_contiguous_input = softmax_input.reshape(2, 12)
    contiguous_shifted = softmax_contiguous_input - np.max(
        softmax_contiguous_input, axis=1, keepdims=True
    )
    contiguous_exponentials = np.exp(contiguous_shifted)
    softmax_contiguous = contiguous_exponentials / np.sum(
        contiguous_exponentials, axis=1, keepdims=True
    )
    flat_left = np.asarray(
        [-4.0, -2.5, -1.0, -0.25, 0.0, 0.5, 1.25, 2.0, 3.5, 5.0],
        dtype=np.float32,
    )
    flat_right = np.asarray(
        [0.5, -2.0, 4.0, 0.25, -0.75, 2.5, -1.25, 8.0, 1.75, -4.0],
        dtype=np.float32,
    )
    flat_scalar = np.float32(1.25)
    trailing_left = np.asarray(
        [(((index * 11) % 23) - 11) / 4.0 for index in range(20)],
        dtype=np.float32,
    ).reshape(2, 10)
    general_right = np.asarray(
        [0.5, -1.0, 1.5, -2.0, 2.5, -3.0, 3.5, -4.0],
        dtype=np.float32,
    ).reshape(2, 1, 4)
    return {
        "add": (left + add_right).ravel(),
        "mul": (left * mul_right).ravel(),
        "div": (left / div_right).ravel(),
        "trailing_add": (trailing_left + flat_right).ravel(),
        "general_add": (left + general_right).ravel(),
        "flat_add": flat_left + flat_right,
        "right_scalar_add": flat_left + flat_scalar,
        "flat_mul": flat_left * flat_right,
        "right_scalar_mul": flat_left * flat_scalar,
        "flat_div": flat_left / flat_right,
        "right_scalar_div": flat_left / flat_scalar,
        "relu": np.maximum(activation_input, np.float32(0.0)),
        "erf": np.asarray(
            [math.erf(float(value)) for value in activation_input], dtype=np.float32
        ),
        "hard_sigmoid": np.clip(
            np.float32(0.2) * activation_input + np.float32(0.5), 0.0, 1.0
        ),
        "sigmoid": np.asarray(
            [1.0 / (1.0 + math.exp(-float(value))) for value in activation_input],
            dtype=np.float32,
        ),
        "softmax": softmax.ravel(),
        "softmax_in_place": softmax.ravel(),
        "softmax_contiguous_axis": softmax_contiguous.ravel(),
        "softmax_contiguous_axis_in_place": softmax_contiguous.ravel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    args = parser.parse_args()
    completed = subprocess.run(
        [args.driver], check=True, capture_output=True, text=True, encoding="utf-8"
    )
    actual = parse_output(completed.stdout)
    expected = expected_results()
    if actual.keys() != expected.keys():
        raise AssertionError(
            f"result names differ: actual={sorted(actual)}, expected={sorted(expected)}"
        )
    for name, expected_values in expected.items():
        np.testing.assert_allclose(
            actual[name], expected_values, rtol=1.0e-5, atol=2.0e-6,
            err_msg=name,
        )
    print(f"validated {len(expected)} scalar kernel results against NumPy")


if __name__ == "__main__":
    main()
