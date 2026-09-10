"""Gate old-browser Web SDK bootstrap and image-decoding fallbacks."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

EXPECTED_FIRST_LINE = "纯臻营养护发素"


def static_contract(root: Path) -> None:
    sdk = (root / "web" / "lw_ppocr_sdk.template.js").read_text(encoding="utf-8")
    ui = (root / "web" / "ocr-demo-ui.js").read_text(encoding="utf-8")
    template = (root / "web" / "ocr-demo.template.html").read_text(encoding="utf-8")
    assert "})(globalThis);" not in sdk
    assert 'typeof globalThis !== "undefined"' in sdk
    assert "markSdkReady" in sdk and "LW_WEB_WASM_INIT_FAILED" in sdk
    assert "waitForImage" in sdk and "willReadFrequently: true}) ||" in sdk
    assert "LwPpocr.create(" not in ui
    assert "getOcrSdk()" in ui
    assert "markWasmReady" in ui
    assert "captureError" in template and "LW_WEB_JS_FAILED" in template


def run_case(html: Path, sample: Path, mode: str) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, args=["--allow-file-access-from-files"]
        )
        page = browser.new_page()
        if mode == "no-globalThis":
            page.add_init_script(
                """() => {
                  try {
                    Object.defineProperty(window, "globalThis", {
                      configurable: true, value: undefined
                    });
                  } catch (_) {}
                }"""
            )
        elif mode == "no-image-decode":
            page.add_init_script(
                """() => {
                  try {
                    Object.defineProperty(window, "createImageBitmap", {
                      configurable: true, value: undefined
                    });
                    Object.defineProperty(HTMLImageElement.prototype, "decode", {
                      configurable: true, value: undefined
                    });
                  } catch (_) {}
                }"""
            )
        page.goto(html.resolve().as_uri(), wait_until="load", timeout=180_000)
        page.wait_for_function(
            "() => window.__lwOcrTest && window.__lwOcrTest.snapshot().ready",
            timeout=180_000,
        )
        boot = page.evaluate(
            "() => window.__lwOcrBootStatus && window.__lwOcrBootStatus.snapshot()"
        )
        assert boot["sdkReady"], boot
        assert boot["wasmReady"], boot
        page.locator("#file").set_input_files(str(sample.resolve()))
        page.wait_for_function(
            "() => !document.querySelector('#run').disabled", timeout=180_000
        )
        result = page.evaluate("() => window.lwPpocrDemo.recognize()")
        assert len(result["lines"]) == 16, result
        assert result["lines"][0]["text"] == EXPECTED_FIRST_LINE, result
        assert page.evaluate(
            "() => window.__lwOcrBootStatus.snapshot().lastError"
        ) is None, boot
        browser.close()
        print(f"{mode}: ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    static_contract(root)
    if not args.html.is_file() or not args.sample.is_file():
        raise SystemExit("standalone HTML or sample image is missing")
    run_case(args.html, args.sample, "no-globalThis")
    run_case(args.html, args.sample, "no-image-decode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())