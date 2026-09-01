"""Verify offline multi-page PDF OCR in the standalone HTML application."""

from __future__ import annotations

import argparse
import json
import struct
import tempfile
from pathlib import Path

from PIL import Image
from playwright.sync_api import ConsoleMessage, Request, sync_playwright


EXPECTED_FIRST_LINE = "纯臻营养护发素"


def create_image_pdf(jpeg_path: Path, output: Path, page_count: int = 2) -> None:
    """Wrap one JPEG losslessly into repeated PDF pages without re-encoding it."""
    jpeg = jpeg_path.read_bytes()
    with Image.open(jpeg_path) as image:
        width, height = image.size
        if image.mode != "RGB":
            raise ValueError("PDF test fixture requires an RGB JPEG")

    # Rendered at 180 DPI, the page produces exactly the original image size.
    width_pt = width * 72.0 / 180.0
    height_pt = height * 72.0 / 180.0
    objects: list[bytes] = []
    page_object_ids = list(range(3, 3 + page_count))
    image_object_id = 3 + page_count
    content_object_ids = list(
        range(image_object_id + 1, image_object_id + 1 + page_count)
    )
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = b" ".join(f"{number} 0 R".encode("ascii") for number in page_object_ids)
    objects.append(
        b"<< /Type /Pages /Count "
        + str(page_count).encode("ascii")
        + b" /Kids [ "
        + kids
        + b" ] >>"
    )
    for page_index, page_object_id in enumerate(page_object_ids):
        del page_object_id
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {width_pt:.6f} {height_pt:.6f}] "
                f"/Resources << /XObject << /Im0 {image_object_id} 0 R >> >> "
                f"/Contents {content_object_ids[page_index]} 0 R >>"
            ).encode("ascii")
        )
    objects.append(
        (
            "<< /Type /XObject /Subtype /Image "
            f"/Width {width} /Height {height} /ColorSpace /DeviceRGB "
            f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(jpeg)} >>\nstream\n"
        ).encode("ascii")
        + jpeg
        + b"\nendstream"
    )
    content = f"q {width_pt:.6f} 0 0 {height_pt:.6f} 0 0 cm /Im0 Do Q\n".encode(
        "ascii"
    )
    for _ in range(page_count):
        objects.append(
            f"<< /Length {len(content)} >>\nstream\n".encode("ascii")
            + content
            + b"endstream"
        )

    data = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, payload in enumerate(objects, start=1):
        offsets.append(len(data))
        data.extend(f"{object_id} 0 obj\n".encode("ascii"))
        data.extend(payload)
        data.extend(b"\nendobj\n")
    xref_offset = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    data.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    output.write_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--browser-executable", type=Path)
    arguments = parser.parse_args()

    html = arguments.html.resolve()
    sample = arguments.sample.resolve()
    if not html.is_file() or not sample.is_file():
        raise SystemExit("standalone HTML or sample image is missing")
    with Image.open(sample) as sample_image:
        sample_width, sample_height = sample_image.size

    browser_messages: list[str] = []
    network_requests: list[str] = []
    with tempfile.TemporaryDirectory(prefix="lw-ppocr-pdf-") as temporary:
        fixture = Path(temporary) / "sample-two-pages.pdf"
        create_image_pdf(sample, fixture)

        with sync_playwright() as playwright:
            launch_options: dict[str, object] = {
                "headless": True,
                "args": ["--allow-file-access-from-files"],
            }
            if arguments.browser_executable:
                launch_options["executable_path"] = str(
                    arguments.browser_executable.resolve()
                )
            browser = playwright.chromium.launch(**launch_options)
            page = browser.new_page(viewport={"width": 1180, "height": 900})

            def capture_console(message: ConsoleMessage) -> None:
                if message.type == "error":
                    browser_messages.append(f"{message.type}: {message.text}")

            def capture_request(request: Request) -> None:
                if not request.url.startswith(("file:", "blob:", "data:")):
                    network_requests.append(request.url)

            page.on("console", capture_console)
            page.on("pageerror", lambda error: browser_messages.append(f"pageerror: {error}"))
            page.on("request", capture_request)
            page.goto(html.as_uri(), wait_until="load", timeout=180_000)
            page.wait_for_function(
                "() => window.__lwOcrTest && window.__lwOcrTest.snapshot().ready",
                timeout=180_000,
            )
            assert page.evaluate("window.LwPdf && window.LwPdf.apiVersion") == 1
            assert page.evaluate("window.LwPdf.pdfjsVersion") == "6.3.289"
            assert "application/pdf" in page.locator("#file").get_attribute("accept")
            assert "application/pdf" not in page.locator("#camera").get_attribute("accept")

            page.locator("#file").set_input_files(str(fixture))
            page.wait_for_function(
                "() => window.__lwOcrTest.snapshot().sourceKind === 'pdf' && "
                "window.__lwOcrTest.snapshot().pdfPageCount === 2 && "
                "!document.querySelector('#run').disabled",
                timeout=180_000,
            )
            snapshot = page.evaluate("window.__lwOcrTest.snapshot()")
            assert snapshot["pdfCurrentPage"] == 1, snapshot
            assert snapshot["pdfWorkerBackend"] == "module-worker", snapshot
            assert page.evaluate("window.__lwOcrTest.pdfStatus().state") == "ready"
            assert page.locator("#pdf-controls").is_visible()
            assert page.locator("#pdf-page-label").inner_text() == "1 / 2"
            assert page.locator("#canvas").evaluate("canvas => canvas.width > 0")
            assert page.locator("#copy-text").is_disabled()

            page.locator("#pdf-next").click()
            page.wait_for_function(
                "() => window.__lwOcrTest.snapshot().pdfCurrentPage === 2 && "
                "document.querySelector('#status').textContent.includes('准备好')",
                timeout=180_000,
            )
            page.locator("#pdf-prev").click()
            page.wait_for_function(
                "() => window.__lwOcrTest.snapshot().pdfCurrentPage === 1 && "
                "document.querySelector('#status').textContent.includes('准备好')",
                timeout=180_000,
            )

            page.locator("#pdf-scope").select_option("all")
            page.locator("#pdf-dpi").select_option("180")
            page.locator("#run").click()
            page.wait_for_function(
                "() => window.__lwOcrTest.snapshot().processedPages === 2 && "
                "document.querySelector('#run').textContent === '开始识别'",
                timeout=300_000,
            )
            result = page.evaluate("window.__lwOcrTest.structuredResult()")
            assert result["schema_version"] == 2
            assert result["source_type"] == "pdf"
            assert result["source"] == fixture.name
            assert result["document"] == {"page_count": 2, "processed_pages": 2}
            assert result["options"] == {
                "use_cls": False,
                "pdf_dpi": 180,
                "pdf_max_pixels": 5_000_000,
            }
            assert len(result["pages"]) == 2
            for page_result in result["pages"]:
                assert len(page_result["lines"]) == 16, page_result
                assert page_result["lines"][0]["text"] == EXPECTED_FIRST_LINE
                assert abs(
                    page_result["pdf"]["width_pt"] - sample_width * 72 / 180
                ) < 0.01
                assert abs(
                    page_result["pdf"]["height_pt"] - sample_height * 72 / 180
                ) < 0.01
                assert page_result["image"]["width"] > 0
                assert page_result["image"]["height"] > 0
                assert page_result["timing"]["render_ms"] > 0
                assert page_result["timing"]["inference_ms"] > 0
                for line in page_result["lines"]:
                    assert len(line["box"]) == 8
                    assert len(line["pdf_box"]) == 8
            assert result["timing"]["pdf_load_ms"] > 0
            assert result["timing"]["render_ms"] > 0
            assert result["timing"]["inference_ms"] > 0
            assert result["timing"]["total_ms"] > 0

            plain_text = "\n\n".join(
                f"===== Page {page_result['page_number']} / 2 =====\n\n"
                + "\n".join(line["text"] for line in page_result["lines"])
                for page_result in result["pages"]
            )
            # Check the browser-side string without transferring a top-level
            # Chinese string through CDP (which is lossy in some Windows builds).
            assert page.evaluate(
                "() => window.__lwOcrTest.structuredResult().pages.every(page => "
                "window.__lwOcrTest.plainTextResult().includes(page.lines[0].text))"
            )
            assert "===== Page 1 / 2 =====" in plain_text
            assert "===== Page 2 / 2 =====" in plain_text
            assert sum(
                line == EXPECTED_FIRST_LINE for line in plain_text.splitlines()
            ) == 2
            page.locator("#show-full-result").click()
            assert page.locator("#results .full-text").is_visible()
            page.locator("#show-page-result").click()
            assert page.locator("#results .line").count() == 16
            assert page.locator("#overlay polygon").count() == 16

            with page.expect_download() as txt_download_info:
                page.locator("#export-txt").click()
            txt_download = txt_download_info.value
            assert txt_download.suggested_filename == "sample-two-pages-ocr.txt"
            txt_path = txt_download.path()
            assert txt_path is not None
            assert Path(txt_path).read_text(encoding="utf-8") == plain_text + "\n"

            with page.expect_download() as json_download_info:
                page.locator("#export-json").click()
            json_download = json_download_info.value
            assert json_download.suggested_filename == "sample-two-pages-ocr.json"
            json_path = json_download.path()
            assert json_path is not None
            assert json.loads(Path(json_path).read_text(encoding="utf-8")) == result

            # Repeat one page after capacities are warm; the WASM heap must not
            # continue to grow for the same rendered geometry.
            page.locator("#pdf-scope").select_option("current")
            heap_history: list[int] = []
            for _ in range(2):
                previous_count = page.evaluate("window.__lwOcrTest.snapshot().runCount")
                page.locator("#run").click()
                page.wait_for_function(
                    "count => window.__lwOcrTest.snapshot().runCount > count && "
                    "document.querySelector('#run').textContent === '开始识别'",
                    arg=previous_count,
                    timeout=300_000,
                )
                heap_history.append(
                    page.evaluate("window.__lwOcrTest.snapshot().heapBytes")
                )
            assert heap_history[-1] == heap_history[-2], heap_history

            # Stop is cooperative: rendering is cancelled immediately, while
            # native OCR stops after the current page if it has already begun.
            page.locator("#pdf-scope").select_option("all")
            page.locator("#run").click()
            page.wait_for_function(
                "() => document.querySelector('#run').textContent === '停止'",
                timeout=30_000,
            )
            page.locator("#run").click()
            page.wait_for_function(
                "() => document.querySelector('#run').textContent === '开始识别' && "
                "document.querySelector('#status').textContent.includes('停止')",
                timeout=300_000,
            )

            # Selecting another source invalidates every PDF export immediately.
            page.locator("#file").set_input_files(str(sample))
            page.wait_for_function(
                "() => window.__lwOcrTest.snapshot().sourceKind === 'image' && "
                "!window.__lwOcrTest.snapshot().hasResults",
                timeout=180_000,
            )
            assert page.locator("#pdf-controls").is_hidden()
            assert page.locator("#export-json").is_disabled()

            page.locator("#file").set_input_files(
                {
                    "name": "broken.pdf",
                    "mimeType": "application/pdf",
                    "buffer": struct.pack("<4I", 1, 2, 3, 4),
                }
            )
            page.wait_for_function(
                "() => window.__lwOcrTest.snapshot().pdfErrorCode === "
                "'LW_PDF_LOAD_FAILED'",
                timeout=180_000,
            )
            assert not page.evaluate("window.__lwOcrTest.snapshot().hasResults")
            assert page.locator("#pdf-diagnostics").is_visible()
            diagnostics = json.loads(page.locator("#pdf-diagnostics-text").inner_text())
            assert diagnostics["error_code"] == "LW_PDF_LOAD_FAILED"
            assert diagnostics["error_phase"] == "open"
            assert diagnostics["pdf"]["environment"]["protocol"] == "file:"
            page.locator("#copy-pdf-diagnostics").click()
            page.wait_for_function(
                "() => document.querySelector('#status').textContent.includes('诊断信息已复制')"
            )

            # A malformed document must not poison the shared PDF worker.
            page.locator("#file").set_input_files(str(fixture))
            page.wait_for_function(
                "() => window.__lwOcrTest.snapshot().sourceKind === 'pdf' && "
                "window.__lwOcrTest.snapshot().pdfErrorCode === null && "
                "!document.querySelector('#run').disabled",
                timeout=180_000,
            )
            assert page.locator("#pdf-diagnostics").is_hidden()

            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate(
                "document.documentElement.scrollWidth <= "
                "document.documentElement.clientWidth"
            )
            page.close()

            # Simulate a restrictive/older mobile WebView after the OCR engine
            # has initialized: PDF Blob module Workers are blocked and
            # Blob.arrayBuffer and Promise.withResolvers are missing. PDF
            # parsing must install its standards-compatible Promise shim, then
            # fall back to the main thread and FileReader while OCR keeps its
            # existing Worker.
            compatibility_page = browser.new_page(
                viewport={"width": 390, "height": 844}
            )
            compatibility_page.on("console", capture_console)
            compatibility_page.on(
                "pageerror",
                lambda error: browser_messages.append(f"pageerror: {error}"),
            )
            compatibility_page.on("request", capture_request)
            compatibility_page.goto(
                html.as_uri(), wait_until="load", timeout=180_000
            )
            compatibility_page.wait_for_function(
                "() => window.__lwOcrTest && window.__lwOcrTest.snapshot().ready",
                timeout=180_000,
            )
            compatibility_page.evaluate(
                """() => {
                  window.Worker = function () {
                    throw new Error('module Worker blocked by mobile WebView');
                  };
                  Object.defineProperty(Blob.prototype, 'arrayBuffer', {
                    configurable: true,
                    writable: true,
                    value: undefined
                  });
                  Object.defineProperty(Promise, 'withResolvers', {
                    configurable: true,
                    writable: true,
                    value: undefined
                  });
                }"""
            )
            compatibility_page.locator("#file").set_input_files(str(fixture))
            compatibility_page.wait_for_function(
                "() => window.__lwOcrTest.snapshot().sourceKind === 'pdf' && "
                "window.__lwOcrTest.snapshot().pdfWorkerBackend === 'main-thread' && "
                "!document.querySelector('#run').disabled",
                timeout=180_000,
            )
            compatibility_status = compatibility_page.evaluate(
                "window.__lwOcrTest.pdfStatus()"
            )
            assert compatibility_status["environment"]["has_blob_array_buffer"] is False
            assert (
                compatibility_status["environment"]["promise_with_resolvers"]
                == "polyfill"
            )
            assert "mobile WebView" in compatibility_status["worker_fallback_reason"]
            assert "兼容模式" in compatibility_page.locator("#status").inner_text()
            compatibility_page.locator("#pdf-scope").select_option("current")
            compatibility_page.locator("#run").click()
            compatibility_page.wait_for_function(
                "() => window.__lwOcrTest.snapshot().processedPages === 1 && "
                "document.querySelector('#run').textContent === '开始识别'",
                timeout=300_000,
            )
            compatibility_result = compatibility_page.evaluate(
                "window.__lwOcrTest.structuredResult()"
            )
            assert len(compatibility_result["pages"]) == 1
            assert len(compatibility_result["pages"][0]["lines"]) == 16
            assert compatibility_result["pages"][0]["lines"][0]["text"] == (
                EXPECTED_FIRST_LINE
            )
            browser.close()

    if browser_messages:
        raise AssertionError("browser console diagnostics:\n" + "\n".join(browser_messages))
    if network_requests:
        raise AssertionError("standalone PDF made network requests: " + repr(network_requests))
    print(json.dumps({
        "fallback_backend": "main-thread",
        "lines": 32,
        "network_requests": 0,
        "pages": 2,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
