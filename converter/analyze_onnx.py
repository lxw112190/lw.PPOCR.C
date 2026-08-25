#!/usr/bin/env python3
"""Inspect the exact ONNX surface used by the bundled PP-OCR models.

This is a development-time tool. Nothing in the future deployment runtime may
import ONNX, NumPy, protobuf, or Python.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import pathlib
import sys
from typing import Any, Iterable

import onnx
from onnx import AttributeProto, ModelProto, TensorProto, helper, shape_inference


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MODELS = {
    "det": REPO_ROOT / "models" / "ppocrv6-tiny" / "det.onnx",
    "cls": REPO_ROOT / "models" / "ppocrv6-tiny" / "cls.onnx",
    "rec": REPO_ROOT / "models" / "ppocrv6-tiny" / "rec.onnx",
}
DEFAULT_REPRESENTATIVE_SHAPES = {
    "det": [1, 3, 640, 640],
    "cls": [1, 3, 80, 160],
    "rec": [1, 3, 48, 320],
}

SHAPE_SOURCE_OPS = {"Shape", "Size"}
SHAPE_TRANSFORM_OPS = {
    "Add", "Cast", "Ceil", "Concat", "ConstantOfShape", "Div", "Floor",
    "Gather", "GatherElements", "Identity", "Max", "Min", "Mul", "Range",
    "Reshape", "Slice", "Squeeze", "Sub", "Unsqueeze",
}
HEAVY_OPS = {"Conv", "ConvTranspose", "Gemm", "MatMul"}


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dtype_name(dtype: int) -> str:
    try:
        return TensorProto.DataType.Name(dtype).lower()
    except ValueError:
        return f"unknown({dtype})"


def _dimension(dimension: onnx.TensorShapeProto.Dimension) -> int | str | None:
    if dimension.HasField("dim_value"):
        return int(dimension.dim_value)
    if dimension.HasField("dim_param"):
        return dimension.dim_param
    return None


def _value_info(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor = value.type.tensor_type
    if not value.type.HasField("tensor_type"):
        return {"name": value.name, "dtype": "non-tensor", "shape": []}
    return {
        "name": value.name,
        "dtype": _dtype_name(tensor.elem_type),
        "shape": [_dimension(dimension) for dimension in tensor.shape.dim],
    }


def _shape_map(model: ModelProto) -> dict[str, list[int | str | None]]:
    result: dict[str, list[int | str | None]] = {}
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for value in values:
        if value.type.HasField("tensor_type"):
            result[value.name] = _value_info(value)["shape"]
    for tensor in model.graph.initializer:
        result[tensor.name] = [int(dimension) for dimension in tensor.dims]
    return result


def _bind_input_symbols(
    original: ModelProto,
    shapes: dict[str, list[int | str | None]],
    representative_shape: list[int],
) -> dict[str, list[int | str | None]]:
    """Bind exact input dimension symbols for representative FLOP analysis.

    ONNX shape inference may preserve pre-existing symbolic ValueInfo entries
    even after the graph input is specialized. Exact symbol substitution is
    safe here; compound expressions remain diagnostic strings rather than
    being evaluated by this tool.
    """
    initializer_names = {tensor.name for tensor in original.graph.initializer}
    graph_inputs = [value for value in original.graph.input if value.name not in initializer_names]
    if len(graph_inputs) != 1:
        return shapes
    original_shape = _value_info(graph_inputs[0])["shape"]
    bindings = {
        dimension: value
        for dimension, value in zip(original_shape, representative_shape)
        if isinstance(dimension, str)
    }
    return {
        name: [bindings.get(dimension, dimension) for dimension in shape]
        for name, shape in shapes.items()
    }


def _infer_shapes(model: ModelProto) -> tuple[ModelProto, str | None]:
    try:
        try:
            return shape_inference.infer_shapes(
                model, strict_mode=False, data_prop=True
            ), None
        except TypeError:
            return shape_inference.infer_shapes(model), None
    except Exception as error:  # The diagnostic must survive incomplete models.
        return model, f"{type(error).__name__}: {error}"


def _apply_representative_shape(model: ModelProto, shape: list[int]) -> ModelProto:
    cloned = copy.deepcopy(model)
    initializer_names = {tensor.name for tensor in cloned.graph.initializer}
    graph_inputs = [value for value in cloned.graph.input if value.name not in initializer_names]
    if len(graph_inputs) != 1:
        raise ValueError(
            f"representative shape requires exactly one graph input, found {len(graph_inputs)}"
        )
    dimensions = graph_inputs[0].type.tensor_type.shape.dim
    if len(dimensions) != len(shape):
        raise ValueError(
            f"representative shape rank {len(shape)} does not match input rank {len(dimensions)}"
        )
    for dimension, value in zip(dimensions, shape):
        dimension.ClearField("dim_param")
        dimension.dim_value = value
    return cloned


def _tensor_storage_bytes(tensor: TensorProto) -> int:
    if tensor.raw_data:
        return len(tensor.raw_data)
    if tensor.data_type == TensorProto.STRING:
        return sum(len(value) for value in tensor.string_data)
    element_sizes = {
        TensorProto.FLOAT: 4,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.UINT16: 2,
        TensorProto.INT16: 2,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
        TensorProto.BOOL: 1,
        TensorProto.FLOAT16: 2,
        TensorProto.DOUBLE: 8,
        TensorProto.UINT32: 4,
        TensorProto.UINT64: 8,
        TensorProto.COMPLEX64: 8,
        TensorProto.COMPLEX128: 16,
        TensorProto.BFLOAT16: 2,
    }
    size = element_sizes.get(tensor.data_type, 0)
    count = math.prod(int(dimension) for dimension in tensor.dims) if tensor.dims else 1
    return size * count


def _summarize_attribute(attribute: AttributeProto) -> Any:
    if attribute.type == AttributeProto.TENSOR:
        tensor = attribute.t
        return {
            "dtype": _dtype_name(tensor.data_type),
            "shape": [int(dimension) for dimension in tensor.dims],
            "bytes": _tensor_storage_bytes(tensor),
        }
    if attribute.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS):
        return "<subgraph>"
    value = helper.get_attribute_value(attribute)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="backslashreplace")
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        rendered = [
            item.decode("utf-8", errors="backslashreplace")
            if isinstance(item, bytes) else item
            for item in value[:16]
        ]
        if len(value) > 16:
            rendered.append(f"... ({len(value) - 16} more)")
        return rendered
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _operator_attributes(model: ModelProto) -> dict[str, dict[str, list[Any]]]:
    collected: dict[str, dict[str, dict[str, Any]]] = {}
    for node in model.graph.node:
        operator = collected.setdefault(node.op_type, {})
        for attribute in node.attribute:
            value = _summarize_attribute(attribute)
            key = json.dumps(value, ensure_ascii=False, sort_keys=True)
            operator.setdefault(attribute.name, {})[key] = value
    result: dict[str, dict[str, list[Any]]] = {}
    for op_type in sorted(collected):
        result[op_type] = {}
        for name in sorted(collected[op_type]):
            values = collected[op_type][name]
            result[op_type][name] = [values[key] for key in sorted(values)[:16]]
    return result


def _constant_outputs(model: ModelProto) -> set[str]:
    outputs: set[str] = set()
    for node in model.graph.node:
        if node.op_type == "Constant":
            outputs.update(output for output in node.output if output)
    return outputs


def _classify_foldable_nodes(model: ModelProto) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    initializers = {tensor.name for tensor in model.graph.initializer}
    constant_values = set(initializers) | _constant_outputs(model)
    constant_nodes: list[dict[str, Any]] = []
    shape_values: set[str] = set()
    shape_nodes: list[dict[str, Any]] = []

    changed = True
    while changed:
        changed = False
        for index, node in enumerate(model.graph.node):
            inputs = [name for name in node.input if name]
            outputs = [name for name in node.output if name]
            record = {"index": index, "name": node.name, "op": node.op_type}

            if node.op_type != "Constant" and inputs and all(
                name in constant_values for name in inputs
            ) and not all(name in constant_values for name in outputs):
                constant_values.update(outputs)
                constant_nodes.append(record)
                changed = True

            is_shape = node.op_type in SHAPE_SOURCE_OPS
            if node.op_type in SHAPE_TRANSFORM_OPS and inputs:
                is_shape = all(
                    name in shape_values or name in constant_values for name in inputs
                ) and any(name in shape_values for name in inputs)
            if is_shape and not all(name in shape_values for name in outputs):
                shape_values.update(outputs)
                shape_nodes.append(record)
                changed = True

    def unique(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        by_index = {record["index"]: record for record in records}
        return [by_index[index] for index in sorted(by_index)]

    return unique(constant_nodes), unique(shape_nodes)


def _fusion_candidates(model: ModelProto) -> dict[str, int]:
    producer: dict[str, int] = {}
    consumers: collections.Counter[str] = collections.Counter()
    for index, node in enumerate(model.graph.node):
        for output in node.output:
            if output:
                producer[output] = index
        consumers.update(name for name in node.input if name)

    conv_batch_norm = 0
    for node in model.graph.node:
        if node.op_type != "BatchNormalization" or not node.input:
            continue
        parent_index = producer.get(node.input[0])
        if parent_index is None:
            continue
        parent = model.graph.node[parent_index]
        if parent.op_type == "Conv" and consumers[node.input[0]] == 1:
            conv_batch_norm += 1
    return {"conv_batch_normalization": conv_batch_norm}


def _element_count(shape: list[int | str | None] | None) -> int | None:
    if shape is None or any(not isinstance(value, int) or value < 0 for value in shape):
        return None
    return math.prod(shape)


def _estimate_flops(
    model: ModelProto, shapes: dict[str, list[int | str | None]]
) -> dict[str, Any]:
    initializer_shapes = {
        tensor.name: [int(dimension) for dimension in tensor.dims]
        for tensor in model.graph.initializer
    }
    by_operator: collections.Counter[str] = collections.Counter()
    top_nodes: list[dict[str, Any]] = []
    unestimated: collections.Counter[str] = collections.Counter()

    for index, node in enumerate(model.graph.node):
        if node.op_type not in HEAVY_OPS:
            continue
        flops: int | None = None
        output_shape = shapes.get(node.output[0]) if node.output else None
        output_elements = _element_count(output_shape)

        if node.op_type in {"Conv", "ConvTranspose"} and len(node.input) >= 2:
            weight_shape = initializer_shapes.get(node.input[1], shapes.get(node.input[1]))
            if output_elements is not None and weight_shape and len(weight_shape) >= 3:
                kernel_work = _element_count(weight_shape[1:])
                if kernel_work is not None:
                    flops = output_elements * kernel_work * 2
        elif node.op_type == "MatMul" and len(node.input) >= 2:
            left_shape = shapes.get(node.input[0])
            if output_elements is not None and left_shape and isinstance(left_shape[-1], int):
                flops = output_elements * int(left_shape[-1]) * 2
        elif node.op_type == "Gemm" and len(node.input) >= 2:
            left_shape = shapes.get(node.input[0])
            right_shape = initializer_shapes.get(node.input[1], shapes.get(node.input[1]))
            if left_shape and right_shape and len(left_shape) >= 2 and len(right_shape) >= 2:
                if all(isinstance(value, int) for value in (left_shape[-2], left_shape[-1], right_shape[-1])):
                    flops = int(left_shape[-2]) * int(left_shape[-1]) * int(right_shape[-1]) * 2

        if flops is None:
            unestimated[node.op_type] += 1
            continue
        by_operator[node.op_type] += flops
        top_nodes.append({
            "index": index,
            "name": node.name,
            "op": node.op_type,
            "flops": flops,
        })

    top_nodes.sort(key=lambda item: (-item["flops"], item["index"]))
    return {
        "total": sum(by_operator.values()),
        "by_operator": dict(sorted(by_operator.items())),
        "top_nodes": top_nodes[:12],
        "unestimated_heavy_nodes": dict(sorted(unestimated.items())),
        "note": "Approximate multiply-add count as two FLOPs; preprocessing and elementwise ops are excluded.",
    }


def _format_shape(shape: list[int | str | None]) -> str:
    return "[" + ", ".join("?" if value is None else str(value) for value in shape) + "]"


def analyze_model(
    label: str,
    path: pathlib.Path,
    representative_shape: list[int] | None = None,
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    model = onnx.load(path, load_external_data=True)
    onnx.checker.check_model(model)
    inferred, inference_error = _infer_shapes(model)
    shapes = _shape_map(inferred)

    representative_outputs: list[dict[str, Any]] = []
    flop_model = inferred
    if representative_shape:
        representative_model = _apply_representative_shape(model, representative_shape)
        flop_model, representative_error = _infer_shapes(representative_model)
        if representative_error:
            inference_error = inference_error or representative_error
        representative_outputs = [_value_info(value) for value in flop_model.graph.output]
    flop_shapes = _shape_map(flop_model)
    if representative_shape:
        flop_shapes = _bind_input_symbols(model, flop_shapes, representative_shape)
        for output in representative_outputs:
            output["shape"] = [
                flop_shapes.get(output["name"], output["shape"])[index]
                for index in range(len(output["shape"]))
            ]

    initializer_dtype_counts = collections.Counter(
        _dtype_name(tensor.data_type) for tensor in model.graph.initializer
    )
    initializers = [
        {
            "name": tensor.name,
            "dtype": _dtype_name(tensor.data_type),
            "shape": [int(dimension) for dimension in tensor.dims],
            "bytes": _tensor_storage_bytes(tensor),
        }
        for tensor in model.graph.initializer
    ]
    initializers.sort(key=lambda item: (-item["bytes"], item["name"]))

    operator_counts = collections.Counter(node.op_type for node in model.graph.node)
    constant_nodes = [
        {"index": index, "name": node.name, "op": node.op_type}
        for index, node in enumerate(model.graph.node)
        if node.op_type == "Constant"
    ]
    constant_fold_nodes, shape_only_nodes = _classify_foldable_nodes(model)

    dynamic_values = [
        {"name": name, "shape": shape}
        for name, shape in sorted(shapes.items())
        if any(not isinstance(value, int) for value in shape)
    ]
    dynamic_node_counts: collections.Counter[str] = collections.Counter()
    dynamic_names = {value["name"] for value in dynamic_values}
    for node in model.graph.node:
        if any(name in dynamic_names for name in node.output):
            dynamic_node_counts[node.op_type] += 1

    initializer_names = {tensor.name for tensor in model.graph.initializer}
    graph_inputs = [value for value in inferred.graph.input if value.name not in initializer_names]
    relative_path: str
    try:
        relative_path = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        relative_path = str(path)

    return {
        "label": label,
        "path": relative_path,
        "sha256": _sha256(path),
        "file_bytes": path.stat().st_size,
        "ir_version": model.ir_version,
        "producer": {"name": model.producer_name, "version": model.producer_version},
        "opsets": [
            {"domain": item.domain or "ai.onnx", "version": item.version}
            for item in model.opset_import
        ],
        "inputs": [_value_info(value) for value in graph_inputs],
        "outputs": [_value_info(value) for value in inferred.graph.output],
        "representative_input_shape": representative_shape,
        "representative_outputs": representative_outputs,
        "shape_inference_error": inference_error,
        "node_count": len(model.graph.node),
        "operator_counts": dict(sorted(operator_counts.items())),
        "operator_attributes": _operator_attributes(model),
        "initializer_count": len(initializers),
        "initializer_bytes": sum(item["bytes"] for item in initializers),
        "initializer_dtype_counts": dict(sorted(initializer_dtype_counts.items())),
        "largest_initializers": initializers[:12],
        "constant_nodes": constant_nodes,
        "constant_fold_candidates": constant_fold_nodes,
        "shape_only_nodes": shape_only_nodes,
        "identity_nodes": operator_counts.get("Identity", 0),
        "fusion_candidates": _fusion_candidates(model),
        "dynamic_values": dynamic_values,
        "dynamic_node_counts": dict(sorted(dynamic_node_counts.items())),
        "flops": _estimate_flops(flop_model, flop_shapes),
    }


def build_report(models: list[dict[str, Any]]) -> dict[str, Any]:
    operator_models: dict[str, list[str]] = collections.defaultdict(list)
    for model in models:
        for operator in model["operator_counts"]:
            operator_models[operator].append(model["label"])
    union = {
        operator: sorted(labels)
        for operator, labels in sorted(operator_models.items())
    }
    return {
        "analysis_schema_version": 1,
        "tool": "converter/analyze_onnx.py",
        "onnx_version": onnx.__version__,
        "models": models,
        "operator_union": union,
        "operator_union_count": len(union),
    }


def _format_flops(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f} GFLOPs"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f} MFLOPs"
    if value >= 1_000:
        return f"{value / 1_000:.3f} KFLOPs"
    return f"{value} FLOPs"


def render_markdown(report: dict[str, Any]) -> str:
    models = report["models"]
    by_label = {model["label"]: model for model in models}
    labels = [model["label"] for model in models]
    lines = [
        "# PP-OCRv6 Tiny ONNX Operator Analysis (V0)",
        "",
        "> Generated by `converter/analyze_onnx.py`. Do not edit operator counts by hand.",
        "",
        "## Scope and decision boundary",
        "",
        "This report describes only the exact bundled PP-OCRv6 tiny DET, CLS, and REC models.",
        "It is not a promise to support arbitrary ONNX graphs. The v0.1 runtime supports the exact REC and fixed-batch CLS graphs,",
        "FP32, CPU, scalar, and single-threaded. ONNX is a converter dependency only.",
        "",
        "## Model inventory",
        "",
        "| Model | SHA-256 | Opset | Nodes | Initializers | Weight bytes | Input | Output | Representative FLOPs |",
        "|---|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for model in models:
        opsets = ", ".join(
            f"{item['domain']}:{item['version']}" for item in model["opsets"]
        )
        inputs = "; ".join(
            f"`{item['name']}` {_format_shape(item['shape'])}" for item in model["inputs"]
        )
        outputs = "; ".join(
            f"`{item['name']}` {_format_shape(item['shape'])}" for item in model["outputs"]
        )
        lines.append(
            f"| {model['label'].upper()} | `{model['sha256']}` | {opsets} | "
            f"{model['node_count']} | {model['initializer_count']} | "
            f"{model['initializer_bytes']:,} | {inputs} | {outputs} | "
            f"{_format_flops(model['flops']['total'])} |"
        )

    lines.extend([
        "",
        "Representative shapes are DET `[1,3,640,640]`, CLS `[1,3,80,160]`, and",
        "REC `[1,3,48,320]`. FLOPs count multiply-add as two operations and exclude",
        "pre/post-processing and elementwise costs, so they are estimates rather than benchmark data.",
        "",
        f"## Operator union ({report['operator_union_count']} types)",
        "",
        "| Operator | " + " | ".join(label.upper() for label in labels) + " | Required by |",
        "|---|" + "---:|" * len(labels) + "---|",
    ])
    for operator, required_by in report["operator_union"].items():
        counts = [str(by_label[label]["operator_counts"].get(operator, 0)) for label in labels]
        lines.append(
            f"| {operator} | " + " | ".join(counts) + " | "
            + ", ".join(label.upper() for label in required_by) + " |"
        )

    lines.extend(["", "## Per-model findings", ""])
    for model in models:
        lines.extend([
            f"### {model['label'].upper()}",
            "",
            "Operators: " + ", ".join(
                f"`{operator}` × {count}"
                for operator, count in model["operator_counts"].items()
            ) + ".",
            "",
            f"- Dynamic tensors after ONNX shape inference: {len(model['dynamic_values'])}.",
            f"- Shape/metadata nodes: {len(model['shape_only_nodes'])}.",
            f"- Constant-fold candidates: {len(model['constant_fold_candidates'])}.",
            f"- Identity nodes removable by the converter: {model['identity_nodes']}.",
            "- Conv + BatchNormalization fusion candidates: "
            f"{model['fusion_candidates']['conv_batch_normalization']}.",
            f"- Estimated heavy-operator work: {_format_flops(model['flops']['total'])}.",
        ])
        if model["flops"]["by_operator"]:
            lines.append("- FLOPs by heavy operator: " + ", ".join(
                f"{operator} {_format_flops(value)}"
                for operator, value in model["flops"]["by_operator"].items()
            ) + ".")
        if model["shape_inference_error"]:
            lines.append(f"- Shape inference warning: `{model['shape_inference_error']}`.")
        lines.append("")

    rec = by_label.get("rec")
    lines.extend([
        "## REC-first runtime surface",
        "",
    ])
    if rec:
        removable = {"Identity"}
        if rec["fusion_candidates"]["conv_batch_normalization"] == rec["operator_counts"].get("BatchNormalization", 0):
            removable.add("BatchNormalization")
        candidate_ops = [operator for operator in rec["operator_counts"] if operator not in removable]
        lines.extend([
            "The exact REC graph contains " + ", ".join(
                f"`{operator}`" for operator in rec["operator_counts"]
            ) + ".",
            "",
            "The REC converter removes verified Identity aliases. The remaining executable",
            "operator requirement derived from this exact model is:",
            "",
            ", ".join(f"`{operator}`" for operator in candidate_ops) + ".",
            "",
            "This analysis-derived set is not a frozen LWM contract. Runtime implementation",
            "status and independent reference-test coverage are tracked in `scalar-kernels.md`.",
            "",
            "## REC dynamic-width propagation",
            "",
            f"REC input: `{rec['inputs'][0]['name']}` {_format_shape(rec['inputs'][0]['shape'])}.",
            "",
            f"REC output: `{rec['outputs'][0]['name']}` {_format_shape(rec['outputs'][0]['shape'])}.",
            "",
            "The batch, channel, and height constraints remain fixed for the MVP while input width",
            "is symbolic. ONNX shape inference propagates width through convolution/pooling and into",
            "the sequence dimension. Dynamic output-producing nodes by operator are: " + ", ".join(
                f"`{operator}` × {count}"
                for operator, count in rec["dynamic_node_counts"].items()
            ) + ". The converter must preserve only this PP-OCR-specific width expression; it must",
            "not introduce a general symbolic-shape engine.",
            "",
        ])

    lines.extend([
        "## Shape and converter-removable work",
        "",
        "- `Identity` is structurally removable after verifying graph outputs and aliases.",
        "- `Shape`-derived integer subgraphs are metadata candidates and should be evaluated offline",
        "  whenever all required dimensions are fixed or expressible by the supported REC width rule.",
        "- Initializer-only subgraphs are constant-fold candidates.",
        "- Conv + BatchNormalization is a low-risk fusion candidate but requires FP32 equivalence tests.",
        "- Data-path `Transpose`, `Squeeze`, `Unsqueeze`, `Reshape`, and arithmetic operations remain",
        "  runtime work unless analysis proves that a specific node is metadata-only or foldable.",
        "",
        "## Main compute conclusion",
        "",
        "The representative FLOP estimate is dominated by convolution and matrix multiplication.",
        "The first performance work should therefore follow correctness-tested scalar Conv and MatMul;",
        "metadata operators should be minimized in the converter. No kernel is justified solely because",
        "it exists in ONNX—the exact model/operator matrix above is the requirement source.",
        "",
        "## Runtime handoff requirements",
        "",
        "1. Lock conversion to the exact verified REC model identity and opset.",
        "2. Remove only structurally verified aliases and retain every required data-path node.",
        "3. Require an independent NumPy or ONNX reference test for every LWM operator.",
        "4. Compare complete-graph output before exposing a public inference call; enforce this with the private executor reference test.",
        "",
    ])
    return "\n".join(lines)


def _parse_model(value: str) -> tuple[str, pathlib.Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("model must use non-empty LABEL=PATH")
    return label.lower(), pathlib.Path(path)


def _parse_shape(value: str) -> tuple[str, list[int]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input shape must use LABEL=N,C,H,W")
    label, dimensions = value.split("=", 1)
    try:
        shape = [int(item) for item in dimensions.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("input shape dimensions must be integers") from error
    if not label or not shape or any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError("input shape dimensions must be positive")
    return label.lower(), shape


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", action="append", type=_parse_model,
        help="model to analyze as LABEL=PATH; defaults to bundled det/cls/rec",
    )
    parser.add_argument(
        "--input-shape", action="append", type=_parse_shape,
        help="representative input shape as LABEL=N,C,H,W",
    )
    parser.add_argument("--json-output", type=pathlib.Path)
    parser.add_argument("--markdown-output", type=pathlib.Path)
    args = parser.parse_args(argv)

    model_items = args.model or list(DEFAULT_MODELS.items())
    labels = [label for label, _ in model_items]
    if len(labels) != len(set(labels)):
        parser.error("model labels must be unique")

    shapes = dict(DEFAULT_REPRESENTATIVE_SHAPES)
    if args.input_shape:
        shapes.update(args.input_shape)

    analyses = [
        analyze_model(label, path, shapes.get(label))
        for label, path in model_items
    ]
    report = build_report(analyses)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            render_markdown(report), encoding="utf-8", newline="\n"
        )

    summary = {
        "models": {
            model["label"]: {
                "nodes": model["node_count"],
                "operator_types": len(model["operator_counts"]),
                "dynamic_values": len(model["dynamic_values"]),
                "estimated_flops": model["flops"]["total"],
            }
            for model in analyses
        },
        "operator_union_count": report["operator_union_count"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
