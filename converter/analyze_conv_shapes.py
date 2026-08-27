#!/usr/bin/env python3
"""Dump the exact Conv/MatMul/Gemm shape distribution of the bundled PP-OCR models.

Development-time tool. Reuses analyze_onnx.py shape inference and FLOP rules so
the numbers always match the main analysis report. Output is a per-model table
of every heavy node with (Cin, Cout, H, W, kernel, stride, pad, group, FLOPs)
plus a category summary that answers: which kernel class to optimize first.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze_onnx as A  # noqa: E402


def _prod(values: list[int]) -> int:
    return math.prod(values) if values else 1


def _flops_conv(out_shape: list[int], w_shape: list[int]) -> int | None:
    out_elem = A._element_count(out_shape)
    kernel_work = A._element_count(w_shape[1:])
    if out_elem is None or kernel_work is None:
        return None
    return out_elem * kernel_work * 2


def _flops_matmul(out_shape: list[int], left_shape: list[int]) -> int | None:
    out_elem = A._element_count(out_shape)
    if out_elem is None or not left_shape or not isinstance(left_shape[-1], int):
        return None
    return out_elem * left_shape[-1] * 2


def _conv_channels_and_kind(
    weight_shape: list[int], group: int
) -> tuple[int, int, int, int, str]:
    """Decode ONNX [Cout, Cin/group, KH, KW] without losing group channels."""
    output_channels, input_channels_per_group, kernel_height, kernel_width = weight_shape
    input_channels = input_channels_per_group * group
    if group == input_channels and input_channels_per_group == 1:
        kind = "dw"
    elif kernel_height == 1 and kernel_width == 1:
        kind = "1x1"
    else:
        kind = f"{kernel_height}x{kernel_width}"
    return input_channels, output_channels, kernel_height, kernel_width, kind


def analyze_conv_distribution(
    label: str, path: pathlib.Path, representative_shape: list[int]
) -> dict[str, Any]:
    import onnx

    model = onnx.load(path, load_external_data=True)
    rep_model = A._apply_representative_shape(model, representative_shape)
    flop_model, _ = A._infer_shapes(rep_model)
    flop_shapes = A._bind_input_symbols(model, A._shape_map(flop_model), representative_shape)
    init_shapes = {tensor.name: [int(d) for d in tensor.dims] for tensor in model.graph.initializer}

    rows: list[dict[str, Any]] = []
    unestimated = 0
    for index, node in enumerate(model.graph.node):
        if node.op_type not in ("Conv", "MatMul", "Gemm"):
            continue
        if not node.input or not node.output:
            continue
        out_shape = flop_shapes.get(node.output[0])
        if out_shape is None or any(not isinstance(v, int) for v in out_shape):
            unestimated += 1
            continue

        if node.op_type == "Conv":
            w_shape = init_shapes.get(node.input[1], flop_shapes.get(node.input[1]))
            if w_shape is None or len(w_shape) != 4:
                unestimated += 1
                continue
            attrs = {a.name: A.helper.get_attribute_value(a) for a in node.attribute}
            strides = list(attrs.get("strides", [1, 1]))
            pads = list(attrs.get("pads", [0, 0, 0, 0]))
            group = int(attrs.get("group", 1))
            ic, oc, kh, kw, kind = _conv_channels_and_kind(
                [int(v) for v in w_shape], group
            )
            rows.append({
                "index": index,
                "name": node.name,
                "op": "Conv",
                "kind": kind,
                "in": f"{ic}->{oc}",
                "cin": ic, "cout": oc,
                "h": int(out_shape[2]), "w": int(out_shape[3]),
                "k": f"{kh}x{kw}", "s": strides, "p": pads, "g": group,
                "flops": _flops_conv(out_shape, w_shape),
            })
        else:  # MatMul / Gemm
            left_shape = flop_shapes.get(node.input[0])
            right_shape = init_shapes.get(node.input[1], flop_shapes.get(node.input[1]))
            rows.append({
                "index": index,
                "name": node.name,
                "op": node.op_type,
                "kind": node.op_type.lower(),
                "in": str([int(v) for v in left_shape]) if left_shape else "?",
                "cin": 0, "cout": 0,
                "h": 0, "w": 0,
                "k": str([int(v) for v in right_shape]) if right_shape else "?",
                "s": [], "p": [], "g": 0,
                "flops": _flops_matmul(out_shape, left_shape) if left_shape else None,
            })

    total = sum(r["flops"] or 0 for r in rows)
    for row in rows:
        row["share"] = (row["flops"] or 0) / total if total else 0.0

    by_kind: dict[str, list[int]] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row["flops"] or 0)
    summary = {
        kind: {"count": len(values), "flops": sum(values)}
        for kind, values in sorted(by_kind.items())
    }
    for item in summary.values():
        item["share"] = item["flops"] / total if total else 0.0

    # Unique shape families: collapse by (kind, cin, cout, k, s, group)
    families: dict[tuple, list[int]] = {}
    for row in rows:
        if row["op"] != "Conv":
            continue
        key = (row["kind"], row["cin"], row["cout"], row["k"], tuple(row["s"]), row["g"])
        families.setdefault(key, []).append(row["flops"] or 0)
    family_list = [
        {"kind": k[0], "cin": k[1], "cout": k[2], "k": k[3], "s": list(k[4]), "g": k[5],
         "count": len(v), "flops": sum(v), "share": sum(v) / total if total else 0.0}
        for k, v in sorted(families.items(), key=lambda item: -sum(item[1]))
    ]

    return {
        "label": label,
        "total_flops": total,
        "unestimated": unestimated,
        "rows": rows,
        "summary": summary,
        "families": family_list,
    }


def _fmt(value: int) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.3f}G"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    return f"{value / 1e3:.1f}K"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render(dist: dict[str, Any]) -> str:
    lines = [
        f"# {dist['label'].upper()} heavy-operator distribution",
        "",
        f"Total heavy FLOPs: {_fmt(dist['total_flops'])}; unestimated nodes: {dist['unestimated']}.",
        "",
        "## By kernel category",
        "",
        "| Category | Count | FLOPs | Share |",
        "|---|---:|---:|---:|",
    ]
    for kind, item in dist["summary"].items():
        lines.append(f"| {kind} | {item['count']} | {_fmt(item['flops'])} | {_fmt_pct(item['share'])} |")

    lines.extend(["", "## Shape families (deduplicated)", "", "| Kind | Cin | Cout | K | Stride | Group | Count | FLOPs | Share |", "|---|---:|---:|---|---:|---:|---:|---:|---:|"])
    for family in dist["families"]:
        lines.append(
            f"| {family['kind']} | {family['cin']} | {family['cout']} | {family['k']} | "
            f"{family['s']} | {family['g']} | {family['count']} | {_fmt(family['flops'])} | "
            f"{_fmt_pct(family['share'])} |"
        )

    lines.extend(["", "## Every heavy node", "", "| # | Name | Kind | Cin->Cout | Out HxW | K | S | P | G | FLOPs | Share |", "|---|---:|---|---|---|---:|---|---|---:|---:|---:|"])
    for row in dist["rows"]:
        if row["op"] == "Conv":
            lines.append(
                f"| {row['index']} | `{row['name']}` | {row['kind']} | {row['cin']}->{row['cout']} | "
                f"{row['h']}x{row['w']} | {row['k']} | {row['s']} | {row['p']} | {row['g']} | "
                f"{_fmt(row['flops'] or 0)} | {_fmt_pct(row['share'])} |"
            )
        else:
            lines.append(
                f"| {row['index']} | `{row['name']}` | {row['op']} | {row['in']} | - | {row['k']} | - | - | - | "
                f"{_fmt(row['flops'] or 0)} | {_fmt_pct(row['share'])} |"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=A._parse_model)
    parser.add_argument("--input-shape", action="append", type=A._parse_shape)
    args = parser.parse_args(argv)

    model_items = args.model or list(A.DEFAULT_MODELS.items())
    shapes = dict(A.DEFAULT_REPRESENTATIVE_SHAPES)
    if args.input_shape:
        shapes.update(args.input_shape)

    sections = [
        render(analyze_conv_distribution(label, path, shapes.get(label)))
        for label, path in model_items
    ]
    print("\n\n".join(sections))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
