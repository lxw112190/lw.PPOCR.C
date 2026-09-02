"""Inline the browser SDK, optional PDF frontend, and UI into one HTML file."""
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
    parser.add_argument("--pdf-adapter", type=Path)
    parser.add_argument("--pdfjs-core", type=Path)
    parser.add_argument("--pdfjs-worker", type=Path)
    parser.add_argument("--pdfjs-jbig2", type=Path)
    parser.add_argument("--pdfjs-openjpeg", type=Path)
    parser.add_argument("--pdfjs-qcms", type=Path)
    parser.add_argument("--pdfjs-version")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pdf_inputs = [
        args.pdf_adapter,
        args.pdfjs_core,
        args.pdfjs_worker,
        args.pdfjs_jbig2,
        args.pdfjs_openjpeg,
        args.pdfjs_qcms,
        args.pdfjs_version,
    ]
    if args.no_pdf and any(pdf_inputs):
        raise SystemExit("--no-pdf cannot be combined with PDF asset arguments")
    if not args.no_pdf and not all(pdf_inputs):
        raise SystemExit(
            "PDF packaging requires --pdf-adapter, --pdfjs-core, "
            "--pdfjs-worker, --pdfjs-jbig2, --pdfjs-openjpeg, "
            "--pdfjs-qcms, and --pdfjs-version"
        )

    if args.no_pdf:
        pdf_bootstrap = "/* PDF support excluded by LW_WEB_PDF=OFF. */"
    else:
        pdf_bootstrap = read_text(args.pdf_adapter)
        pdf_bootstrap = pdf_bootstrap.replace(
            "__LW_PDFJS_VERSION__", args.pdfjs_version
        )
        pdf_bootstrap = pdf_bootstrap.replace(
            "__LW_PDFJS_CORE_BASE64__",
            base64.b64encode(args.pdfjs_core.read_bytes()).decode("ascii"),
        )
        pdf_bootstrap = pdf_bootstrap.replace(
            "__LW_PDFJS_WORKER_BASE64__",
            base64.b64encode(args.pdfjs_worker.read_bytes()).decode("ascii"),
        )
        pdf_bootstrap = pdf_bootstrap.replace(
            "__LW_PDFJS_JBIG2_BASE64__",
            base64.b64encode(args.pdfjs_jbig2.read_bytes()).decode("ascii"),
        )
        pdf_bootstrap = pdf_bootstrap.replace(
            "__LW_PDFJS_OPENJPEG_BASE64__",
            base64.b64encode(args.pdfjs_openjpeg.read_bytes()).decode("ascii"),
        )
        pdf_bootstrap = pdf_bootstrap.replace(
            "__LW_PDFJS_QCMS_BASE64__",
            base64.b64encode(args.pdfjs_qcms.read_bytes()).decode("ascii"),
        )

    html = read_text(args.template)
    replacements = {
        "__LW_PDF_BOOTSTRAP_JS__": pdf_bootstrap,
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
    for name in ("__LW_PDF_BOOTSTRAP_JS__", "__LW_SDK_JS__", "__LW_DEMO_UI_JS__"):
        if "</script>" in replacements[name].lower():
            raise SystemExit(f"script payload {name} contains a closing script tag")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8", newline="\n")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
