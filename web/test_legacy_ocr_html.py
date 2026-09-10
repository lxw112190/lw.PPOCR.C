
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

def text_sha256(lines):
    return hashlib.sha256("\n".join(line["text"] for line in lines).encode("utf-8")).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--html",type=Path,required=True)
    parser.add_argument("--sample",type=Path,required=True)
    parser.add_argument("--golden",type=Path,required=True)
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    contract=json.loads(args.golden.read_text(encoding="utf-8"))
    expected_count=int(contract["expected_line_count"])
    expected_hash=contract["expected_text_sha256"]
    started=time.perf_counter()
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=True,args=["--allow-file-access-from-files"])
        page=browser.new_page()
        page.goto(args.html.resolve().as_uri(),wait_until="load",timeout=180000)
        page.wait_for_function("() => window.LwPpocr && window.__lwOcrTest",timeout=180000)
        page.wait_for_function("() => window.__lwOcrTest.snapshot().ready",timeout=180000)
        build=page.evaluate("() => window.LwPpocr.buildInfo")
        if build != {"flavor":"legacy","wasmSimd128":False,"pdf":False,"minChromeVersion":70}:
            raise AssertionError(build)
        if page.evaluate("() => typeof window.LwPdf") not in ("undefined","object"):
            raise AssertionError("unexpected PDF frontend")
        page.locator("#file").set_input_files(str(args.sample.resolve()))
        page.wait_for_function("() => !document.querySelector('#run').disabled",timeout=180000)
        first=page.evaluate("() => window.lwPpocrDemo.recognize()")
        second=page.evaluate("() => window.lwPpocrDemo.recognize()")
        if len(first["lines"]) != expected_count:
            raise AssertionError(first)
        if text_sha256(first["lines"]) != expected_hash:
            raise AssertionError("Legacy OCR text checksum mismatch")
        if len(second["lines"]) != expected_count or text_sha256(second["lines"]) != expected_hash:
            raise AssertionError("Legacy second OCR mismatch")
        status=page.evaluate("() => window.__lwOcrTest.snapshot()")
        browser.close()
    report={"build_flavor":"legacy","model":"PP-OCRv6 Tiny","wasm_simd128":False,
            "pdf":False,"min_chrome":70,"html_bytes":args.html.stat().st_size,
            "sdk_ready":True,"wasm_ready":True,"line_count":len(first["lines"]),
            "text_sha256":text_sha256(first["lines"]),
            "init_ms":None,"ocr_ms":first.get("elapsed_ms"),"heap_bytes":status.get("heapBytes"),
            "wall_ms":round((time.perf_counter()-started)*1000,3)}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
