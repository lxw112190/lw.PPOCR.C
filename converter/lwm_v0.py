"""Deterministic writer for the experimental LWM v0.1 file format."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import struct
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import numpy_helper

MAGIC = b"LWM0"
FORMAT_MAJOR = 0
FORMAT_MINOR = 1
HEADER_SIZE = 160
TENSOR_SIZE = 80
NODE_SIZE = 72
CHECKSUM_OFFSET = 128
MAX_DIMS = 8
MAX_NODE_INPUTS = 8
MAX_NODE_OUTPUTS = 4
UINT64_MAX = (1 << 64) - 1
SUPPORTED_REC_SHA256 = "9ef676d6ed3c88256a2d92c640c44f25b0c40947e111b14b8be8f594091563e6"
SUPPORTED_CLS_SHA256 = "dd8b2b61983d76ab230a58da9e0e0e84956b71c3877f2ce6e438fe22d74d2cf2"
SUPPORTED_DET_SHA256 = "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8"

HEADER_FLAG_NO_MEMORY_PLAN = 1 << 0
TENSOR_FLAG_CONSTANT = 1 << 0
TENSOR_FLAG_INPUT = 1 << 1
TENSOR_FLAG_OUTPUT = 1 << 2

DTYPE_F32 = 1
DTYPE_I32 = 2
DTYPE_I64 = 3
DTYPE_U8 = 4

OP_IDS = {
    "Conv": 1,
    "Add": 2,
    "Mul": 3,
    "Div": 4,
    "Erf": 5,
    "HardSigmoid": 6,
    "BatchNormalization": 7,
    "ReduceMean": 8,
    "Relu": 9,
    "AveragePool": 10,
    "Squeeze": 11,
    "Transpose": 12,
    "Unsqueeze": 13,
    "MatMul": 14,
    "Softmax": 15,
    "Reshape": 16,
    "Concat": 17,
    "ConvTranspose": 18,
    "MaxPool": 19,
    "Resize": 20,
    "Sigmoid": 21,
}

PARAM_SIZES = {
    OP_IDS["Conv"]: 64,
    OP_IDS["HardSigmoid"]: 16,
    OP_IDS["BatchNormalization"]: 24,
    OP_IDS["ReduceMean"]: 48,
    OP_IDS["AveragePool"]: 64,
    OP_IDS["Squeeze"]: 40,
    OP_IDS["Transpose"]: 40,
    OP_IDS["Unsqueeze"]: 40,
    OP_IDS["Softmax"]: 16,
    OP_IDS["Concat"]: 16,
    OP_IDS["ConvTranspose"]: 64,
    OP_IDS["MaxPool"]: 64,
    OP_IDS["Resize"]: 32,
}


def align8(value: int) -> int:
    return (value + 7) & ~7


def fnv1a64(data: bytes) -> int:
    value = 14695981039346656037
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & UINT64_MAX
    return value


def _attribute_map(node: onnx.NodeProto) -> dict[str, object]:
    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


def _int_list(attrs: dict[str, object], name: str, default: Iterable[int]) -> list[int]:
    value = attrs.get(name, list(default))
    return [int(item) for item in value]  # type: ignore[arg-type]


def _require_count(values: list[int], count: int, description: str) -> list[int]:
    if len(values) != count:
        raise ValueError(f"{description} must contain {count} values, got {len(values)}")
    return values


def encode_params(node: onnx.NodeProto) -> bytes:
    attrs = _attribute_map(node)
    if node.op_type in ("Conv", "ConvTranspose"):
        kernel = _require_count(_int_list(attrs, "kernel_shape", []), 2, "Conv kernel_shape")
        strides = _require_count(_int_list(attrs, "strides", [1, 1]), 2, "Conv strides")
        dilations = _require_count(_int_list(attrs, "dilations", [1, 1]), 2, "Conv dilations")
        pads = _require_count(_int_list(attrs, "pads", [0, 0, 0, 0]), 4, "Conv pads")
        if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
            raise ValueError("Conv auto_pad other than NOTSET is unsupported")
        return struct.pack(
            "<HHI2i2i2i4i4I",
            1,
            2,
            int(attrs.get("group", 1)),
            *kernel,
            *strides,
            *dilations,
            *pads,
            0,
            0,
            0,
            0,
        )
    if node.op_type == "BatchNormalization":
        return struct.pack(
            "<HHffI2I",
            1,
            0,
            float(attrs.get("epsilon", 1.0e-5)),
            float(attrs.get("momentum", 0.9)),
            int(attrs.get("training_mode", 0)),
            0,
            0,
        )
    if node.op_type == "ReduceMean":
        axes = _int_list(attrs, "axes", [])
        if len(axes) > MAX_DIMS:
            raise ValueError("ReduceMean has too many axes")
        return struct.pack(
            "<HHII8iI",
            1,
            len(axes),
            int(attrs.get("keepdims", 1)),
            int(attrs.get("noop_with_empty_axes", 0)),
            *(axes + [0] * (MAX_DIMS - len(axes))),
            0,
        )
    if node.op_type == "AveragePool":
        kernel = _require_count(_int_list(attrs, "kernel_shape", []), 2, "AveragePool kernel_shape")
        strides = _require_count(_int_list(attrs, "strides", [1, 1]), 2, "AveragePool strides")
        pads = _require_count(_int_list(attrs, "pads", [0, 0, 0, 0]), 4, "AveragePool pads")
        if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
            raise ValueError("AveragePool auto_pad other than NOTSET is unsupported")
        return struct.pack(
            "<HHI2i2i4iII4I",
            1,
            0,
            2,
            *kernel,
            *strides,
            *pads,
            int(attrs.get("ceil_mode", 0)),
            int(attrs.get("count_include_pad", 0)),
            0,
            0,
            0,
            0,
        )
    if node.op_type == "MaxPool":
        kernel = _require_count(_int_list(attrs, "kernel_shape", []), 2, "MaxPool kernel_shape")
        strides = _require_count(_int_list(attrs, "strides", [1, 1]), 2, "MaxPool strides")
        pads = _require_count(_int_list(attrs, "pads", [0, 0, 0, 0]), 4, "MaxPool pads")
        dilations = _require_count(_int_list(attrs, "dilations", [1, 1]), 2, "MaxPool dilations")
        if dilations != [1, 1]:
            raise ValueError("MaxPool dilation other than one is unsupported")
        if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET"):
            raise ValueError("MaxPool auto_pad other than NOTSET is unsupported")
        return struct.pack(
            "<HHI2i2i4iII4I",
            1,
            0,
            2,
            *kernel,
            *strides,
            *pads,
            int(attrs.get("ceil_mode", 0)),
            0,
            0,
            0,
            0,
            0,
        )
    if node.op_type == "HardSigmoid":
        return struct.pack(
            "<HHffI",
            1,
            0,
            float(attrs.get("alpha", 0.2)),
            float(attrs.get("beta", 0.5)),
            0,
        )
    if node.op_type in ("Squeeze", "Unsqueeze", "Transpose"):
        name = "perm" if node.op_type == "Transpose" else "axes"
        values = _int_list(attrs, name, [])
        if len(values) > MAX_DIMS:
            raise ValueError(f"{node.op_type} has too many {name} values")
        return struct.pack(
            "<HH8iI",
            1,
            len(values),
            *(values + [0] * (MAX_DIMS - len(values))),
            0,
        )
    if node.op_type == "Softmax":
        return struct.pack("<HHiII", 1, 0, int(attrs.get("axis", 1)), 0, 0)
    if node.op_type == "Concat":
        return struct.pack("<HHiII", 1, 0, int(attrs["axis"]), 0, 0)
    if node.op_type == "Resize":
        scales = [float(value) for value in attrs.get("scales", [])]  # type: ignore[arg-type]
        if len(scales) != 4:
            raise ValueError("Resize scales must contain four values")
        return struct.pack("<HH4f3I", 1, 4, *scales, 0, 0, 0)
    if attrs:
        raise ValueError(f"unexpected attributes on {node.op_type}: {sorted(attrs)}")
    return b""


@dataclasses.dataclass
class TensorRecord:
    name: str
    dtype: int
    dimensions: list[int]
    flags: int
    data: bytes = b""
    data_relative_offset: int = 0


@dataclasses.dataclass
class NodeRecord:
    op: int
    inputs: list[int]
    outputs: list[int]
    params: bytes
    param_relative_offset: int = 0


@dataclasses.dataclass(frozen=True)
class ConversionInfo:
    tensor_count: int
    node_count: int
    input_count: int
    output_count: int
    weight_size: int
    file_size: int
    checksum: int


def _shape(value: onnx.ValueInfoProto) -> list[int]:
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise ValueError(f"tensor {value.name!r} has no shape")
    dims: list[int] = []
    for dim in tensor_type.shape.dim:
        dims.append(int(dim.dim_value) if dim.HasField("dim_value") and dim.dim_value > 0 else -1)
    if len(dims) > MAX_DIMS:
        raise ValueError(f"tensor {value.name!r} rank exceeds {MAX_DIMS}")
    return dims


def _dtype_from_onnx(dtype: int) -> int:
    mapping = {
        onnx.TensorProto.FLOAT: DTYPE_F32,
        onnx.TensorProto.INT32: DTYPE_I32,
        onnx.TensorProto.INT64: DTYPE_I64,
        onnx.TensorProto.UINT8: DTYPE_U8,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported ONNX dtype {dtype}") from exc


def _initializer_bytes(initializer: onnx.TensorProto) -> tuple[int, bytes]:
    dtype = _dtype_from_onnx(initializer.data_type)
    numpy_dtypes = {
        DTYPE_F32: np.dtype("<f4"),
        DTYPE_I32: np.dtype("<i4"),
        DTYPE_I64: np.dtype("<i8"),
        DTYPE_U8: np.dtype("u1"),
    }
    array = np.asarray(numpy_helper.to_array(initializer), dtype=numpy_dtypes[dtype], order="C")
    return dtype, array.tobytes(order="C")


def _resolve_alias(name: str, aliases: dict[str, str]) -> str:
    visited: set[str] = set()
    while name in aliases:
        if name in visited:
            raise ValueError(f"Identity alias cycle involving {name!r}")
        visited.add(name)
        name = aliases[name]
    return name


def _build_records(
    model: onnx.ModelProto,
    inferred_model: onnx.ModelProto | None = None,
) -> tuple[list[TensorRecord], list[NodeRecord], list[int], list[int]]:
    graph = model.graph
    aliases = {
        node.output[0]: node.input[0]
        for node in graph.node
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1
    }
    if sum(1 for node in graph.node if node.op_type == "Identity") != len(aliases):
        raise ValueError("only one-input, one-output Identity nodes are supported")

    inferred = inferred_model or onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=False
    )
    values = {
        value.name: value
        for value in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
    }
    initializers = {item.name: item for item in graph.initializer}
    input_names = [item.name for item in graph.input if item.name not in initializers]
    output_names = [_resolve_alias(item.name, aliases) for item in graph.output]

    ordered_names: list[str] = []
    seen: set[str] = set()
    for name in input_names:
        if name not in seen:
            ordered_names.append(name)
            seen.add(name)
    for item in graph.initializer:
        if item.name not in seen:
            ordered_names.append(item.name)
            seen.add(item.name)
    for node in graph.node:
        if node.op_type == "Identity":
            continue
        if node.op_type not in OP_IDS:
            raise ValueError(f"unsupported operator {node.op_type!r}")
        if len(node.input) > MAX_NODE_INPUTS or len(node.output) > MAX_NODE_OUTPUTS:
            raise ValueError(f"{node.op_type} exceeds LWM node arity limits")
        for name in node.output:
            if name not in seen:
                ordered_names.append(name)
                seen.add(name)

    tensors: list[TensorRecord] = []
    input_set = set(input_names)
    output_set = set(output_names)
    for name in ordered_names:
        flags = (TENSOR_FLAG_INPUT if name in input_set else 0) | (TENSOR_FLAG_OUTPUT if name in output_set else 0)
        if name in initializers:
            initializer = initializers[name]
            dtype, data = _initializer_bytes(initializer)
            tensors.append(TensorRecord(name, dtype, [int(dim) for dim in initializer.dims], flags | TENSOR_FLAG_CONSTANT, data))
        else:
            if name not in values:
                raise ValueError(f"shape inference produced no type information for {name!r}")
            value = values[name]
            tensors.append(TensorRecord(name, _dtype_from_onnx(value.type.tensor_type.elem_type), _shape(value), flags))

    indexes = {tensor.name: index for index, tensor in enumerate(tensors)}
    nodes: list[NodeRecord] = []
    for node in graph.node:
        if node.op_type == "Identity":
            continue
        inputs = [indexes[_resolve_alias(name, aliases)] for name in node.input]
        outputs = [indexes[name] for name in node.output]
        params = encode_params(node)
        op = OP_IDS[node.op_type]
        expected_size = PARAM_SIZES.get(op, 0)
        if len(params) != expected_size:
            raise AssertionError(f"internal parameter size mismatch for {node.op_type}")
        nodes.append(NodeRecord(op, inputs, outputs, params))

    return tensors, nodes, [indexes[name] for name in input_names], [indexes[name] for name in output_names]


def _write_model(
    model: onnx.ModelProto,
    output_path: Path,
    inferred_model: onnx.ModelProto | None = None,
) -> ConversionInfo:
    tensors, nodes, graph_inputs, graph_outputs = _build_records(model, inferred_model)
    return _write_records_layout(
        tensors, nodes, graph_inputs, graph_outputs, output_path
    )


def convert_rec_model(input_path: Path, output_path: Path) -> ConversionInfo:
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if digest != SUPPORTED_REC_SHA256:
        raise ValueError(
            "REC converter only supports the bundled PP-OCRv6 tiny REC model; "
            f"expected SHA-256 {SUPPORTED_REC_SHA256}, got {digest}"
        )
    model = onnx.load(str(input_path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    default_opset = next((item.version for item in model.opset_import if item.domain in ("", "ai.onnx")), None)
    if default_opset != 11:
        raise ValueError(f"REC converter requires ONNX opset 11, got {default_opset}")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("REC converter requires exactly one graph input and one graph output")

    return _write_model(model, output_path)


def _prepare_cls_model(
    model: onnx.ModelProto,
) -> tuple[onnx.ModelProto, onnx.ModelProto]:
    fixed = copy.deepcopy(model)
    batch_dimension = fixed.graph.input[0].type.tensor_type.shape.dim[0]
    batch_dimension.ClearField("dim_param")
    batch_dimension.dim_value = 1
    inferred = onnx.shape_inference.infer_shapes(
        fixed, strict_mode=True, data_prop=False
    )
    for value in (
        *inferred.graph.input,
        *inferred.graph.value_info,
        *inferred.graph.output,
    ):
        shape = value.type.tensor_type.shape
        for axis, dimension in enumerate(shape.dim):
            if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
                if axis != 0:
                    raise ValueError(
                        f"CLS tensor {value.name!r} has a non-batch dynamic dimension"
                    )
                dimension.ClearField("dim_param")
                dimension.dim_value = 1

    converted = copy.deepcopy(fixed)
    rewritten_nodes: list[onnx.NodeProto] = []
    removed_metadata_ops = {"Shape", "Slice", "Concat"}
    for node in converted.graph.node:
        if node.op_type in removed_metadata_ops:
            continue
        if node.op_type == "GlobalAveragePool":
            replacement = onnx.helper.make_node(
                "ReduceMean",
                list(node.input),
                list(node.output),
                name=node.name,
                axes=[2, 3],
                keepdims=1,
            )
            rewritten_nodes.append(replacement)
            continue
        replacement = copy.deepcopy(node)
        if replacement.op_type == "Reshape":
            if len(replacement.input) != 2:
                raise ValueError("CLS Reshape must have data and shape inputs")
            del replacement.input[1:]
        rewritten_nodes.append(replacement)
    del converted.graph.node[:]
    converted.graph.node.extend(rewritten_nodes)

    used_initializers = {name for node in converted.graph.node for name in node.input}
    retained_initializers = [
        item for item in converted.graph.initializer if item.name in used_initializers
    ]
    del converted.graph.initializer[:]
    converted.graph.initializer.extend(retained_initializers)
    return converted, inferred


def convert_cls_model(input_path: Path, output_path: Path) -> ConversionInfo:
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if digest != SUPPORTED_CLS_SHA256:
        raise ValueError(
            "CLS converter only supports the bundled PP-OCRv6 tiny CLS model; "
            f"expected SHA-256 {SUPPORTED_CLS_SHA256}, got {digest}"
        )
    model = onnx.load(str(input_path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    default_opset = next(
        (item.version for item in model.opset_import if item.domain in ("", "ai.onnx")),
        None,
    )
    if default_opset != 7:
        raise ValueError(f"CLS converter requires ONNX opset 7, got {default_opset}")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("CLS converter requires exactly one graph input and one graph output")
    converted, inferred = _prepare_cls_model(model)
    return _write_model(converted, output_path, inferred)


def _normalize_same_upper(node: onnx.NodeProto) -> onnx.NodeProto:
    attrs = _attribute_map(node)
    auto_pad = attrs.get("auto_pad", b"NOTSET")
    if auto_pad in (b"NOTSET", "NOTSET"):
        return copy.deepcopy(node)
    if auto_pad not in (b"SAME_UPPER", "SAME_UPPER"):
        raise ValueError(f"{node.op_type} auto_pad mode is unsupported")
    kernel = _require_count(_int_list(attrs, "kernel_shape", []), 2, "kernel_shape")
    strides = _require_count(_int_list(attrs, "strides", [1, 1]), 2, "strides")
    dilations = _require_count(_int_list(attrs, "dilations", [1, 1]), 2, "dilations")
    if strides != [1, 1]:
        raise ValueError("dynamic SAME_UPPER with non-unit stride is unsupported")
    total_height = (kernel[0] - 1) * dilations[0]
    total_width = (kernel[1] - 1) * dilations[1]
    normalized = copy.deepcopy(node)
    del normalized.attribute[:]
    normalized_attrs = {
        name: value for name, value in attrs.items() if name != "auto_pad"
    }
    normalized_attrs["pads"] = [
        total_height // 2,
        total_width // 2,
        total_height - total_height // 2,
        total_width - total_width // 2,
    ]
    for name in sorted(normalized_attrs):
        normalized.attribute.append(
            onnx.helper.make_attribute(name, normalized_attrs[name])
        )
    return normalized


def _prepare_det_model(
    model: onnx.ModelProto,
) -> tuple[onnx.ModelProto, onnx.ModelProto]:
    inferred = onnx.shape_inference.infer_shapes(
        model, strict_mode=True, data_prop=False
    )
    converted = copy.deepcopy(model)
    initializers = {
        item.name: numpy_helper.to_array(item) for item in model.graph.initializer
    }
    rewritten_nodes: list[onnx.NodeProto] = []
    for node in converted.graph.node:
        if node.op_type == "GlobalAveragePool":
            rewritten_nodes.append(
                onnx.helper.make_node(
                    "ReduceMean",
                    list(node.input),
                    list(node.output),
                    name=node.name,
                    axes=[2, 3],
                    keepdims=1,
                )
            )
            continue
        if node.op_type == "Resize":
            attrs = _attribute_map(node)
            if (
                len(node.input) != 3
                or node.input[1] not in initializers
                or node.input[2] not in initializers
                or initializers[node.input[1]].size != 0
                or attrs.get("mode") != b"nearest"
                or attrs.get("coordinate_transformation_mode") != b"asymmetric"
                or attrs.get("nearest_mode") != b"floor"
            ):
                raise ValueError("DET Resize is outside the supported nearest-neighbor pattern")
            scales = np.asarray(initializers[node.input[2]], dtype=np.float32).reshape(-1)
            if scales.size != 4 or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
                raise ValueError("DET Resize scales are invalid")
            rewritten_nodes.append(
                onnx.helper.make_node(
                    "Resize",
                    [node.input[0]],
                    list(node.output),
                    name=node.name,
                    scales=[float(value) for value in scales],
                )
            )
            continue
        if node.op_type in ("Conv", "MaxPool"):
            rewritten_nodes.append(_normalize_same_upper(node))
            continue
        rewritten_nodes.append(copy.deepcopy(node))
    del converted.graph.node[:]
    converted.graph.node.extend(rewritten_nodes)
    used_initializers = {name for node in converted.graph.node for name in node.input}
    retained_initializers = [
        item for item in converted.graph.initializer if item.name in used_initializers
    ]
    del converted.graph.initializer[:]
    converted.graph.initializer.extend(retained_initializers)
    return converted, inferred


def convert_det_model(input_path: Path, output_path: Path) -> ConversionInfo:
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if digest != SUPPORTED_DET_SHA256:
        raise ValueError(
            "DET converter only supports the bundled PP-OCRv6 tiny DET model; "
            f"expected SHA-256 {SUPPORTED_DET_SHA256}, got {digest}"
        )
    model = onnx.load(str(input_path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    default_opset = next(
        (item.version for item in model.opset_import if item.domain in ("", "ai.onnx")),
        None,
    )
    if default_opset != 14:
        raise ValueError(f"DET converter requires ONNX opset 14, got {default_opset}")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("DET converter requires exactly one graph input and one graph output")
    converted, inferred = _prepare_det_model(model)
    return _write_model(converted, output_path, inferred)


def _write_records_layout(
    tensors: list[TensorRecord],
    nodes: list[NodeRecord],
    graph_inputs: list[int],
    graph_outputs: list[int],
    output_path: Path,
) -> ConversionInfo:
    input_offset = HEADER_SIZE
    output_offset = align8(input_offset + 4 * len(graph_inputs))
    tensor_offset = align8(output_offset + 4 * len(graph_outputs))
    node_offset = align8(tensor_offset + TENSOR_SIZE * len(tensors))
    param_offset = align8(node_offset + NODE_SIZE * len(nodes))

    param_blob = bytearray()
    for node in nodes:
        if node.params:
            node.param_relative_offset = len(param_blob)
            param_blob.extend(node.params)
            param_blob.extend(b"\0" * (align8(len(param_blob)) - len(param_blob)))
    param_size = len(param_blob)
    string_offset = align8(param_offset + param_size)
    string_size = 0
    weight_offset = string_offset

    weight_blob = bytearray()
    for tensor in tensors:
        if tensor.flags & TENSOR_FLAG_CONSTANT:
            tensor.data_relative_offset = len(weight_blob)
            weight_blob.extend(tensor.data)
            weight_blob.extend(b"\0" * (align8(len(weight_blob)) - len(weight_blob)))
    weight_size = len(weight_blob)
    file_size = weight_offset + weight_size

    header = struct.pack(
        "<4sHH6I13Q3Q",
        MAGIC,
        FORMAT_MAJOR,
        FORMAT_MINOR,
        HEADER_SIZE,
        HEADER_FLAG_NO_MEMORY_PLAN,
        len(tensors),
        len(nodes),
        len(graph_inputs),
        len(graph_outputs),
        input_offset,
        output_offset,
        tensor_offset,
        node_offset,
        param_offset,
        param_size,
        string_offset,
        string_size,
        weight_offset,
        weight_size,
        file_size,
        0,
        0,
        0,
        0,
        0,
    )
    assert len(header) == HEADER_SIZE

    output = bytearray(header)
    output.extend(struct.pack(f"<{len(graph_inputs)}I", *graph_inputs))
    output.extend(b"\0" * (output_offset - len(output)))
    output.extend(struct.pack(f"<{len(graph_outputs)}I", *graph_outputs))
    output.extend(b"\0" * (tensor_offset - len(output)))

    for tensor in tensors:
        dimensions = tensor.dimensions + [0] * (MAX_DIMS - len(tensor.dimensions))
        data_offset = weight_offset + tensor.data_relative_offset if tensor.data else 0
        output.extend(
            struct.pack(
                "<II8iII4Q",
                tensor.dtype,
                len(tensor.dimensions),
                *dimensions,
                tensor.flags,
                0,
                data_offset,
                len(tensor.data),
                UINT64_MAX,
                0,
            )
        )
    for node in nodes:
        inputs = node.inputs + [0] * (MAX_NODE_INPUTS - len(node.inputs))
        outputs = node.outputs + [0] * (MAX_NODE_OUTPUTS - len(node.outputs))
        absolute_param = param_offset + node.param_relative_offset if node.params else 0
        output.extend(
            struct.pack(
                "<4H8I4IQII",
                node.op,
                len(node.inputs),
                len(node.outputs),
                0,
                *inputs,
                *outputs,
                absolute_param,
                len(node.params),
                0,
            )
        )
    output.extend(b"\0" * (param_offset - len(output)))
    output.extend(param_blob)
    output.extend(b"\0" * (weight_offset - len(output)))
    output.extend(weight_blob)
    if len(output) != file_size:
        raise AssertionError("internal LWM layout size mismatch")

    checksum = fnv1a64(output)
    struct.pack_into("<Q", output, CHECKSUM_OFFSET, checksum)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)
    return ConversionInfo(len(tensors), len(nodes), len(graph_inputs), len(graph_outputs), weight_size, file_size, checksum)
