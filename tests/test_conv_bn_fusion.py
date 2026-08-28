from __future__ import annotations

import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from converter.lwm_v0 import _fold_conv_batch_normalization


class ConvBatchNormalizationFusionTests(unittest.TestCase):
    def test_folds_inference_pair_and_prunes_source_constants(self) -> None:
        """A folded Conv must produce the standard inference-mode BN equation."""
        weights = np.array([[[[2.0]]], [[[4.0]]]], dtype=np.float32)
        gamma = np.array([3.0, 5.0], dtype=np.float32)
        beta = np.array([7.0, 11.0], dtype=np.float32)
        mean = np.array([13.0, 17.0], dtype=np.float32)
        variance = np.array([4.0, 9.0], dtype=np.float32)
        graph = helper.make_graph(
            [
                helper.make_node("Conv", ["image", "weights"], ["conv"], name="conv"),
                helper.make_node(
                    "BatchNormalization",
                    ["conv", "gamma", "beta", "mean", "variance"],
                    ["result"],
                    name="normalization",
                    epsilon=0.0,
                ),
            ],
            "conv_bn",
            [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 1, 1])],
            [helper.make_tensor_value_info("result", TensorProto.FLOAT, [1, 2, 1, 1])],
            [
                numpy_helper.from_array(weights, name="weights"),
                numpy_helper.from_array(gamma, name="gamma"),
                numpy_helper.from_array(beta, name="beta"),
                numpy_helper.from_array(mean, name="mean"),
                numpy_helper.from_array(variance, name="variance"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

        folded = _fold_conv_batch_normalization(model)

        self.assertEqual([node.op_type for node in folded.graph.node], ["Conv"])
        conv = folded.graph.node[0]
        self.assertEqual(list(conv.output), ["result"])
        self.assertEqual(len(conv.input), 3)
        constants = {
            item.name: numpy_helper.to_array(item) for item in folded.graph.initializer
        }
        self.assertEqual(len(constants), 2)
        scale = gamma / np.sqrt(variance)
        np.testing.assert_allclose(constants[conv.input[1]], weights * scale[:, None, None, None])
        np.testing.assert_allclose(constants[conv.input[2]], -mean * scale + beta)

    def test_does_not_fold_when_conv_value_has_another_consumer(self) -> None:
        """A shared Conv result cannot be rewired without changing another path."""
        graph = helper.make_graph(
            [
                helper.make_node("Conv", ["image", "weights"], ["conv"]),
                helper.make_node(
                    "BatchNormalization",
                    ["conv", "gamma", "beta", "mean", "variance"],
                    ["normalized"],
                ),
                helper.make_node("Relu", ["conv"], ["other"]),
            ],
            "shared_conv",
            [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 1, 1])],
            [helper.make_tensor_value_info("normalized", TensorProto.FLOAT, [1, 1, 1, 1])],
            [
                numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="weights"),
                numpy_helper.from_array(np.ones(1, dtype=np.float32), name="gamma"),
                numpy_helper.from_array(np.zeros(1, dtype=np.float32), name="beta"),
                numpy_helper.from_array(np.zeros(1, dtype=np.float32), name="mean"),
                numpy_helper.from_array(np.ones(1, dtype=np.float32), name="variance"),
            ],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

        unchanged = _fold_conv_batch_normalization(model)

        self.assertEqual(
            [node.op_type for node in unchanged.graph.node],
            ["Conv", "BatchNormalization", "Relu"],
        )


if __name__ == "__main__":
    unittest.main()
