#!/usr/bin/env python3
"""Analyze PP-OCR REC graph patterns for offline AOT specialization.

This development-time tool deliberately stops at reporting. It does not emit
LWM, change the runtime ABI, or make a kernel-selection decision. The report
uses ONNX node indexes; those indexes are not LWM node indexes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any

import onnx
from onnx import helper


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from converter import analyze_onnx as A
DEFAULT_MODEL = REPO_ROOT / "models" / "ppocrv6-tiny" / "rec.onnx"
DEFAULT_INPUT_SHAPE = [1, 3, 48, 320]
SCHEMA_VERSION = 1
HEAVY_OPS = {"Conv", "MatMul", "Gemm"}


def _shape(value: list[Any] | None) -> list[int | str | None] | None:
    if value is None:
        return None
    return [int(item) if isinstance(item, int) else item for item in value]


def _attrs(node: onnx.NodeProto) -> dict[str, Any]:
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _flops_conv(output_shape: list[Any] | None, weight_shape: list[Any] | None) -> int | None:
    if not output_shape or not weight_shape:
        return None
    if any(not isinstance(item, int) for item in output_shape + weight_shape):
        return None
    output_elements = 1
    for item in output_shape:
        output_elements *= item
    kernel_elements = 1
    for item in weight_shape[1:]:
        kernel_elements *= item
    return output_elements * kernel_elements * 2


def _flops_matmul(output_shape: list[Any] | None, left_shape: list[Any] | None) -> int | None:
    if not output_shape or not left_shape or any(not isinstance(item, int) for item in output_shape):
        return None
    if not isinstance(left_shape[-1], int):
        return None
    output_elements = 1
    for item in output_shape:
        output_elements *= item
    return output_elements * left_shape[-1] * 2


def _family_key(node: dict[str, Any]) -> tuple[Any, ...] | None:
    if node["op"] != "Conv":
        return None
    return (
        node["kind"],
        node["input_channels"],
        node["output_channels"],
        tuple(node["kernel"]),
        tuple(node["strides"]),
        tuple(node["dilations"]),
        node["groups"],
        tuple(node["output_shape"] or []),
    )


def _family_label(node: dict[str, Any]) -> str:
    return (
        f"{node['kind']} {node['input_channels']}->{node['output_channels']} "
        f"k={node['kernel']} s={node['strides']} g={node['groups']} "
        f"out={node['output_shape']}"
    )


def analyze_rec_patterns(
    model_path: pathlib.Path,
    representative_shape: list[int],
) -> dict[str, Any]:
    model = onnx.load(model_path, load_external_data=True)
    representative = A._apply_representative_shape(model, representative_shape)
    inferred, _ = A._infer_shapes(representative)
    shape_map = A._bind_input_symbols(model, A._shape_map(inferred), representative_shape)
    initializers = {
        tensor.name: [int(dimension) for dimension in tensor.dims]
        for tensor in model.graph.initializer
    }

    nodes: list[dict[str, Any]] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type not in HEAVY_OPS:
            continue
        output_shape = shape_map.get(node.output[0]) if node.output else None
        input_shape = shape_map.get(node.input[0]) if node.input else None
        row: dict[str, Any] = {
            "index": index,
            "name": node.name or f"{node.op_type}.{index}",
            "op": node.op_type,
            "input_shape": _shape(input_shape),
            "output_shape": _shape(output_shape),
        }
        if node.op_type == "Conv":
            weight_shape = initializers.get(node.input[1]) if len(node.input) > 1 else None
            attributes = _attrs(node)
            strides = [int(value) for value in attributes.get("strides", [1, 1])]
            dilations = [int(value) for value in attributes.get("dilations", [1, 1])]
            kernel = [int(value) for value in (weight_shape[2:4] if weight_shape else [0, 0])]
            groups = int(attributes.get("group", 1))
            input_channels = int(weight_shape[1] * groups) if weight_shape else None
            output_channels = int(weight_shape[0]) if weight_shape else None
            if groups == input_channels and weight_shape and weight_shape[1] == 1:
                kind = "depthwise"
            elif kernel == [1, 1]:
                kind = "pointwise"
            else:
                kind = "conv"
            row.update(
                {
                    "kind": kind,
                    "input_channels": input_channels,
                    "output_channels": output_channels,
                    "kernel": kernel,
                    "strides": strides,
                    "dilations": dilations,
                    "groups": groups,
                    "weight_shape": _shape(weight_shape),
                    "flops": _flops_conv(output_shape, weight_shape),
                }
            )
        else:
            right_shape = initializers.get(node.input[1]) if len(node.input) > 1 else shape_map.get(node.input[1])
            row.update(
                {
                    "kind": node.op_type.lower(),
                    "right_shape": _shape(right_shape),
                    "flops": _flops_matmul(output_shape, input_shape),
                }
            )
        nodes.append(row)

    families: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        key = _family_key(node)
        if key is not None:
            families[key].append(node)
    family_rows = []
    for members in families.values():
        if len(members) < 2:
            continue
        family_rows.append(
            {
                "kind": members[0]["kind"],
                "label": _family_label(members[0]),
                "count": len(members),
                "node_indexes": [member["index"] for member in members],
                "total_flops": sum(member["flops"] or 0 for member in members),
                "flops_per_node": members[0]["flops"],
            }
        )
    family_rows.sort(key=lambda item: (-item["total_flops"], item["label"]))

    matmuls = [node for node in nodes if node["op"] == "MatMul"]
    terminal_matmul = matmuls[-1] if matmuls else None
    candidates: list[dict[str, Any]] = []
    if terminal_matmul is not None:
        candidates.append(
            {
                "pattern": "terminal_matmul",
                "node_indexes": [terminal_matmul["index"]],
                "reason": "terminal recognition projection; benchmark packed activation or epilogue variants",
                "flops": terminal_matmul["flops"],
            }
        )
    for family in family_rows:
        if family["kind"] == "pointwise" and family["count"] >= 3:
            candidates.append(
                {
                    "pattern": "repeated_pointwise_family",
                    "node_indexes": family["node_indexes"],
                    "reason": "repeated 1x1 Conv shape suitable for shared AOT layout or block fusion",
                    "flops": family["total_flops"],
                }
            )
    candidates.sort(key=lambda item: (-(item["flops"] or 0), item["pattern"], item["node_indexes"]))

    return {
        "schema_version": SCHEMA_VERSION,
        "model": str(model_path),
        "model_sha256": A._sha256(model_path),
        "representative_input_shape": representative_shape,
        "onnx_node_count": len(model.graph.node),
        "heavy_nodes": nodes,
        "repeated_families": family_rows,
        "aot_candidates": candidates,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# REC AOT pattern analysis",
        "",
        f"Model: `{report['model']}`",
        f"Representative input: `{report['representative_input_shape']}`",
        f"ONNX nodes: {report['onnx_node_count']}",
        "",
        "## AOT candidates",
        "",
        "| Pattern | ONNX nodes | FLOPs | Reason |",
        "|---|---:|---:|---|",
    ]
    for candidate in report["aot_candidates"]:
        lines.append(
            f"| {candidate['pattern']} | `{candidate['node_indexes']}` | "
            f"{candidate['flops'] or 0:,} | {candidate['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Repeated Conv families",
            "",
            "| Kind | Count | Nodes | FLOPs | Shape |",
            "|---|---:|---|---:|---|",
        ]
    )
    for family in report["repeated_families"]:
        lines.append(
            f"| {family['kind']} | {family['count']} | `{family['node_indexes']}` | "
            f"{family['total_flops'] or 0:,} | {family['label']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Node indexes are ONNX indexes, not LWM indexes.",
            "- This report is analysis-only; it does not enable a new kernel or fusion.",
            "- Every candidate requires an isolated A/B benchmark plus OCR checksum, Golden corpus, and RSS gates.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-shape", type=A._parse_shape, default=DEFAULT_INPUT_SHAPE)
    parser.add_argument("--json-output", type=pathlib.Path)
    parser.add_argument("--markdown-output", type=pathlib.Path)
    args = parser.parse_args(argv)
    report = analyze_rec_patterns(args.model, args.input_shape)
    markdown = render_markdown(report)
    if args.json_output:
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.write_text(markdown + "\n", encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())