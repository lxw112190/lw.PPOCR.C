"""Package the standalone browser OCR JavaScript SDK."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def encode(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--det", type=Path, required=True)
    parser.add_argument("--cls", type=Path, required=True)
    parser.add_argument("--rec", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--model-family", default="PP-OCRv6")
    parser.add_argument("--model-variant", default="tiny")
    parser.add_argument("--model-display-name", default="PP-OCRv6 Tiny")
    parser.add_argument("--version", required=True)
    parser.add_argument("--build-flavor", choices=("modern", "legacy"), default="modern")
    parser.add_argument("--wasm-simd128", choices=("true", "false"), default="true")
    parser.add_argument("--pdf-enabled", choices=("true", "false"), default="true")
    parser.add_argument("--min-chrome-version", default="")
    parser.add_argument("--js-target", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runtime = args.runtime.read_text(encoding="utf-8")
    wrapper = args.template.read_text(encoding="utf-8")
    if "LwPpocrModule" not in runtime:
        raise SystemExit("runtime does not contain LwPpocrModule")
    values = {
        "__LW_SDK_VERSION__": json.dumps(args.version),
        "__LW_MODEL_FAMILY__": json.dumps(args.model_family),
        "__LW_MODEL_VARIANT__": json.dumps(args.model_variant),
        "__LW_MODEL_DISPLAY_NAME__": json.dumps(args.model_display_name),
        "__LW_BUILD_FLAVOR__": json.dumps(args.build_flavor),
        "__LW_WASM_SIMD128__": args.wasm_simd128,
        "__LW_PDF_ENABLED__": args.pdf_enabled,
        "__LW_MIN_CHROME_VERSION__": (
            args.min_chrome_version if args.min_chrome_version else "null"
        ),
        "__LW_JS_TARGET__": (
            json.dumps(args.js_target) if args.js_target else "null"
        ),
        "__LW_RUNTIME_JS_JSON__": json.dumps(runtime),
        "__LW_DET_MODEL_BASE64__": encode(args.det),
        "__LW_CLS_MODEL_BASE64__": encode(args.cls),
        "__LW_REC_MODEL_BASE64__": encode(args.rec),
        "__LW_DICTIONARY_BASE64__": encode(args.dictionary),
    }
    for placeholder, value in values.items():
        wrapper = wrapper.replace(placeholder, value)
    if "__LW_" in wrapper:
        raise SystemExit("unresolved SDK placeholder")
    if "LwPpocr" not in wrapper:
        raise SystemExit("SDK wrapper is missing LwPpocr")

    output = runtime + "\n" + wrapper
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8", newline="\n")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
