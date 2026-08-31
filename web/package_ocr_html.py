"""Inline the browser SDK and UI glue into the standalone offline HTML demo."""
from __future__ import annotations

import argparse
import base64
from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package lw-ppocr.js and the demo UI into one offline HTML file."
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--ui", type=Path, required=True)
    parser.add_argument("--sponsor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    html = read_text(args.template)
    replacements = {
        "__LW_SDK_JS__": read_text(args.sdk),
        "__LW_DEMO_UI_JS__": read_text(args.ui),
        "__LW_SPONSOR_IMAGE_BASE64__": base64.b64encode(
            args.sponsor.read_bytes()
        ).decode("ascii"),
    }
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)
    if "__LW_" in html:
        raise SystemExit("unresolved HTML placeholder")
    if "</script>" in replacements["__LW_SDK_JS__"].lower() or "</script>" in replacements[
        "__LW_DEMO_UI_JS__"
    ].lower():
        raise SystemExit("script payload contains a closing script tag")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8", newline="\n")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
