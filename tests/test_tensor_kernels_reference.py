import argparse
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


def average_pool_reference(input_values: np.ndarray, include_pad: bool) -> np.ndarray:
    output = np.empty((1, 2, 3, 3), dtype=np.float32)
    kernel = (2, 3)
    strides = (1, 2)
    pads = (1, 1)
    for batch in range(output.shape[0]):
        for channel in range(output.shape[1]):
            for output_y in range(output.shape[2]):
                input_y_start = output_y * strides[0] - pads[0]
                for output_x in range(output.shape[3]):
                    input_x_start = output_x * strides[1] - pads[1]
                    values: list[np.float32] = []
                    for kernel_y in range(kernel[0]):
                        input_y = input_y_start + kernel_y
                        for kernel_x in range(kernel[1]):
                            input_x = input_x_start + kernel_x
                            if (
                                0 <= input_y < input_values.shape[2]
                                and 0 <= input_x < input_values.shape[3]
                            ):
                                values.append(input_values[batch, channel, input_y, input_x])
                            elif include_pad:
                                values.append(np.float32(0.0))
                    output[batch, channel, output_y, output_x] = np.mean(
                        np.asarray(values, dtype=np.float32), dtype=np.float32
                    )
    return output


def max_pool_reference(input_values: np.ndarray) -> np.ndarray:
    output = np.empty((1, 2, 3, 3), dtype=np.float32)
    for channel in range(2):
        for output_y in range(3):
            for output_x in range(3):
                values = []
                for kernel_y in range(2):
                    input_y = output_y + kernel_y - 1
                    for kernel_x in range(3):
                        input_x = output_x * 2 + kernel_x - 1
                        if 0 <= input_y < 3 and 0 <= input_x < 5:
                            values.append(input_values[0, channel, input_y, input_x])
                output[0, channel, output_y, output_x] = np.max(values)
    return output


def expected_results() -> dict[str, np.ndarray]:
    tensor_input = np.asarray(
        [(((index * 5) % 17) - 8) / 3.0 for index in range(30)],
        dtype=np.float32,
    )
    transpose_input = tensor_input[:24].reshape(2, 3, 4)
    reshape_input = tensor_input[:6].reshape(2, 1, 3, 1)
    reduce_input = tensor_input[:24].reshape(2, 3, 4)
    pool_input = tensor_input.reshape(1, 2, 3, 5)
    resize_multi_input = tensor_input[:24].reshape(2, 2, 2, 3)
    matmul_input = np.asarray(
        [(((index * 3) % 13) - 6) / 4.0 for index in range(24)],
        dtype=np.float32,
    ).reshape(2, 3, 4)
    matmul_weights = np.asarray(
        [(((index * 7) % 11) - 5) / 5.0 for index in range(20)],
        dtype=np.float32,
    ).reshape(4, 5)
    squeezed = np.squeeze(reshape_input, axis=(1, 3))
    return {
        "transpose": np.transpose(transpose_input, (0, 2, 1)).ravel(),
        "squeeze": squeezed.ravel(),
        "squeeze_all": np.squeeze(reshape_input).ravel(),
        "unsqueeze": np.expand_dims(squeezed, axis=(0, 3)).ravel(),
        "reshape": reshape_input.reshape(2, 3).ravel(),
        "reduce_mean": np.mean(
            reduce_input, axis=(1, 2), keepdims=True, dtype=np.float32
        ).ravel(),
        "reduce_noop": reduce_input.ravel(),
        "average_pool": average_pool_reference(pool_input, False).ravel(),
        "average_pool_include_pad": average_pool_reference(pool_input, True).ravel(),
        "max_pool": max_pool_reference(pool_input).ravel(),
        "concat": np.concatenate(
            (
                np.asarray([-3, -2, -1, 1, 2, 3], dtype=np.float32).reshape(1, 2, 3),
                np.asarray([4, 5, 6], dtype=np.float32).reshape(1, 1, 3),
            ),
            axis=1,
        ).ravel(),
        "resize_nearest": np.repeat(
            np.repeat(
                np.asarray([-3, -2, -1, 1, 2, 3], dtype=np.float32).reshape(1, 1, 2, 3),
                2,
                axis=2,
            ),
            2,
            axis=3,
        ).ravel(),
        "resize_nearest_nchw": np.repeat(
            np.repeat(resize_multi_input, 2, axis=2), 3, axis=3
        ).ravel(),
        "resize_nearest_fractional": np.repeat(
            np.asarray([-3, -2, -1, -3, -2, -1, 1, 2, 3], dtype=np.float32)
            .reshape(1, 1, 3, 3),
            2,
            axis=3,
        ).ravel(),
        "matmul": np.matmul(matmul_input, matmul_weights).ravel(),
        "matmul_dispatched": np.matmul(matmul_input, matmul_weights).ravel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", required=True)
    args = parser.parse_args()
    completed = subprocess.run(
        [args.driver],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
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
    print(f"validated {len(expected)} tensor kernel results against NumPy")


if __name__ == "__main__":
    main()
