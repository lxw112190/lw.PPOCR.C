"""Verify desktop/mobile UI, Worker inference, exports, and memory reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import ConsoleMessage, Page, sync_playwright


EXPECTED_FIRST_LINE = "纯臻营养护发素"


def wait_for_run(page: Page, previous_count: int) -> dict[str, int | bool]:
    page.wait_for_function(
        "count => window.__lwOcrTest.snapshot().runCount > count",
        arg=previous_count,
        timeout=180_000,
    )
    snapshot = page.evaluate("window.__lwOcrTest.snapshot()")
    assert snapshot["ready"], snapshot
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument(
        "--browser-executable",
        type=Path,
        help="optional local Chrome/Edge executable; CI uses Playwright Chromium",
    )
    arguments = parser.parse_args()

    html = arguments.html.resolve()
    sample = arguments.sample.resolve()
    if not html.is_file() or not sample.is_file():
        raise SystemExit("standalone HTML or sample image is missing")

    browser_messages: list[str] = []
    with sync_playwright() as playwright:
        launch_options: dict[str, object] = {
            "headless": True,
            "args": ["--allow-file-access-from-files"],
        }
        if arguments.browser_executable:
            launch_options["executable_path"] = str(arguments.browser_executable.resolve())
        browser = playwright.chromium.launch(**launch_options)
        page = browser.new_page()
        page.add_init_script(
            """
            window.__lwCopiedText = null;
            Object.defineProperty(navigator, "clipboard", {
              configurable: true,
              value: {writeText: async value => { window.__lwCopiedText = value; }}
            });
            """
        )

        def capture_console(message: ConsoleMessage) -> None:
            if message.type == "error":
                browser_messages.append(f"{message.type}: {message.text}")

        page.on("console", capture_console)
        page.on("pageerror", lambda error: browser_messages.append(f"pageerror: {error}"))
        page.goto(html.as_uri(), wait_until="load", timeout=180_000)
        page.wait_for_function(
            "() => window.__lwOcrTest && window.__lwOcrTest.snapshot().ready",
            timeout=180_000,
        )
        assert page.evaluate("typeof window.lwPpocrDemo.recognize") == "function"
        assert page.locator("#camera").get_attribute("capture") == "environment"
        export_buttons = page.locator("#copy-text, #export-txt, #export-json")
        assert export_buttons.count() == 3
        assert all(export_buttons.nth(index).is_disabled() for index in range(3))

        # Exercise the complete DET -> CLS -> REC path used by the product page.
        page.locator("#use-cls").check()
        page.wait_for_function(
            "() => window.__lwOcrTest.snapshot().ready && !document.querySelector('#use-cls').disabled",
            timeout=180_000,
        )
        page.locator("#file").set_input_files(str(sample))
        page.wait_for_function(
            "() => !document.querySelector('#run').disabled && "
            "document.querySelector('#status').textContent.includes('点击')",
            timeout=180_000,
        )
        assert page.locator("#results .line").count() == 0
        assert all(export_buttons.nth(index).is_disabled() for index in range(3))
        page.locator("#run").click()
        snapshot = wait_for_run(page, 0)

        lines = page.locator("#results .line")
        assert lines.count() == 16, page.locator("#status").inner_text()
        assert EXPECTED_FIRST_LINE in lines.first.locator(".text").inner_text()
        assert snapshot["maxLineCapacity"] == 1000, snapshot
        assert 0 < snapshot["maxTextCapacity"] < 1024 * 1024, snapshot
        assert snapshot["backend"] == "worker", snapshot
        assert snapshot["prepareCount"] == 1, snapshot
        assert snapshot["prepared"], snapshot
        assert snapshot["hasResults"], snapshot
        assert snapshot["exportEnabled"], snapshot
        assert all(export_buttons.nth(index).is_enabled() for index in range(3))

        displayed_text = lines.locator("b").all_inner_texts()
        assert len(displayed_text) == 16
        assert page.evaluate("window.lwPpocrDemo.getPlainText()") == "\n".join(displayed_text)
        assert page.evaluate("window.lwPpocrDemo.getResult().lines.length") == 16
        page.locator("#copy-text").click()
        page.wait_for_function("() => window.__lwCopiedText !== null")
        assert page.evaluate("window.__lwCopiedText") == "\n".join(displayed_text)

        with page.expect_download() as txt_download_info:
            page.locator("#export-txt").click()
        txt_download = txt_download_info.value
        assert txt_download.suggested_filename == f"{sample.stem}-ocr.txt"
        txt_path = txt_download.path()
        assert txt_path is not None
        txt_content = Path(txt_path).read_text(encoding="utf-8")
        assert txt_content.endswith("\n")
        assert txt_content.splitlines() == displayed_text
        assert EXPECTED_FIRST_LINE in txt_content

        with page.expect_download() as json_download_info:
            page.locator("#export-json").click()
        json_download = json_download_info.value
        assert json_download.suggested_filename == f"{sample.stem}-ocr.json"
        json_path = json_download.path()
        assert json_path is not None
        exported = json.loads(Path(json_path).read_text(encoding="utf-8"))
        assert exported["schema_version"] == 1
        assert exported["source"] == sample.name
        assert exported["image"]["width"] > 0 and exported["image"]["height"] > 0
        assert exported["options"] == {"use_cls": True}
        assert exported["elapsed_ms"] > 0
        assert len(exported["lines"]) == 16
        assert [line["index"] for line in exported["lines"]] == list(range(16))
        assert [line["text"] for line in exported["lines"]] == displayed_text
        for line in exported["lines"]:
            assert len(line["box"]) == 8
            assert 0 <= line["det_score"] <= 1
            assert 0 <= line["rec_score"] <= 1
            assert 0 <= line["cls_score"] <= 1
            assert line["cls_label"] in (0, 1)
            assert line["rotation_degrees"] in (0, 180)

        # Reconfiguring CLS must preserve the Worker-owned input. The browser
        # should not decode or upload the same image again.
        page.locator("#use-cls").uncheck()
        page.wait_for_function(
            "() => window.__lwOcrTest.snapshot().ready && !document.querySelector('#use-cls').disabled",
            timeout=180_000,
        )
        previous_count = int(snapshot["runCount"])
        page.locator("#run").click()
        snapshot = wait_for_run(page, previous_count)
        assert snapshot["prepareCount"] == 1, snapshot
        assert lines.count() == 16
        page.locator("#use-cls").check()
        page.wait_for_function(
            "() => window.__lwOcrTest.snapshot().ready && !document.querySelector('#use-cls').disabled",
            timeout=180_000,
        )

        # Adaptive-width REC lazily constructs concrete graph widths. A CLS
        # reinitialization can restart that bounded warm-up, and Emscripten's
        # linear memory never shrinks afterward. Keep warm-up and verification
        # as separate phases so a legitimate expansion on the final warm-up
        # run is not mistaken for a leak.
        snapshot = page.evaluate("window.__lwOcrTest.snapshot()")
        initial_heap = int(snapshot["heapBytes"])
        heap_history = [initial_heap]
        stable_source = snapshot["sourceCapacity"]
        warmup_runs = 16
        verification_runs = 6
        verification_heaps: list[int] = []
        for run_index in range(warmup_runs + verification_runs):
            previous_count = int(snapshot["runCount"])
            page.locator("#run").click()
            snapshot = wait_for_run(page, previous_count)
            assert snapshot["sourceCapacity"] == stable_source, snapshot
            assert snapshot["prepareCount"] == 1, snapshot
            assert lines.count() == 16
            heap_bytes = int(snapshot["heapBytes"])
            heap_history.append(heap_bytes)
            if run_index >= warmup_runs:
                verification_heaps.append(heap_bytes)

        # Every post-warm-up run must observe the same high-water mark.
        assert len(verification_heaps) == verification_runs
        assert len(set(verification_heaps)) == 1, {
            "snapshot": snapshot,
            "heapHistory": heap_history,
            "verificationHeaps": verification_heaps,
        }
        assert heap_history[-1] <= initial_heap * 2, {
            "snapshot": snapshot,
            "heapHistory": heap_history,
        }

        # Selecting a new image invalidates the previous result snapshot until
        # the next successful OCR run, so stale data cannot be exported.
        page.locator("#file").set_input_files([])
        page.locator("#file").set_input_files(str(sample))
        page.wait_for_function(
            "() => !document.querySelector('#run').disabled && "
            "document.querySelector('#status').textContent.includes('点击')",
            timeout=180_000,
        )
        reset_snapshot = page.evaluate("window.__lwOcrTest.snapshot()")
        assert not reset_snapshot["hasResults"], reset_snapshot
        assert not reset_snapshot["exportEnabled"], reset_snapshot
        assert all(export_buttons.nth(index).is_disabled() for index in range(3))

        # The same HTML remains a two-column desktop tool, while a phone-sized
        # viewport exposes touch-sized camera/gallery controls and panel tabs.
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.locator(".mobile-source-actions").is_visible()
        run_box = page.locator("#run").bounding_box()
        assert run_box is not None and run_box["height"] >= 44, run_box
        assert page.evaluate(
            "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
        )
        page.locator("#show-results").click()
        assert page.locator(".result-card").is_visible()
        assert not page.locator(".image-card").is_visible()
        page.locator("#show-image").click()
        assert page.locator(".image-card").is_visible()
        assert not page.locator(".result-card").is_visible()

        browser.close()

    if browser_messages:
        raise AssertionError("browser console diagnostics:\n" + "\n".join(browser_messages))
    print(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
