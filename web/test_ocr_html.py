"""Verify the standalone demo UI, exports, and mobile layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from playwright.sync_api import ConsoleMessage, sync_playwright


EXPECTED_FIRST_LINE = "纯臻营养护发素"


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
    assert '<input id="use-cls" type="checkbox" disabled>' in html.read_text(
        encoding="utf-8"
    )

    browser_messages: list[str] = []
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
        assert page.locator("#use-cls").is_enabled()
        assert page.locator("#status").get_attribute("data-loading") is None

        assert page.evaluate("typeof window.LwPpocr.create") == "function"
        assert page.evaluate("window.LwPpocr.webAbiVersion") == 1
        assert page.evaluate("typeof window.lwPpocrDemo.recognize") == "function"
        assert page.evaluate(
            "window.lwPpocrDemo.ready().then(() => window.lwPpocrDemo.getStatus().ready)"
        )
        missing_image_error = page.evaluate(
            """() => window.lwPpocrDemo.recognize()
              .then(() => ({code: "", message: ""}))
              .catch(error => ({code: error.code, message: error.message}))"""
        )
        assert missing_image_error["code"] == "LW_OCR_INPUT_REQUIRED"
        assert "图片" in missing_image_error["message"]
        assert page.locator("#camera").get_attribute("capture") == "environment"

        export_buttons = page.locator("#copy-text, #export-txt, #export-json")
        assert export_buttons.count() == 3
        assert all(export_buttons.nth(index).is_disabled() for index in range(3))

        # Delay the next create() deterministically so the test observes the
        # complete reconfiguration lifecycle instead of relying on machine speed.
        page.evaluate(
            """() => {
              const original = window.LwPpocr;
              window.__lwOriginalNamespace = original;
              window.LwPpocr = {
                ...original,
                create: async options => {
                  await new Promise(resolve => setTimeout(resolve, 250));
                  return original.create(options);
                }
              };
            }"""
        )
        page.locator("#use-cls").check()
        assert page.locator("#use-cls").is_disabled()
        assert not page.evaluate("window.__lwOcrTest.snapshot().ready")
        page.wait_for_function(
            "() => window.__lwOcrTest.snapshot().ready && "
            "!document.querySelector('#use-cls').disabled",
            timeout=180_000,
        )
        page.evaluate("window.LwPpocr = window.__lwOriginalNamespace")
        page.locator("#file").set_input_files(str(sample))
        page.wait_for_function(
            "() => !document.querySelector('#run').disabled && "
            "document.querySelector('#status').textContent.includes('点击')",
            timeout=180_000,
        )
        assert page.locator("#results .line").count() == 0
        assert all(export_buttons.nth(index).is_disabled() for index in range(3))

        # Keep the v1 demo adapter as a compatibility layer while all actual
        # OCR work now goes through the public LwPpocr SDK.
        result = page.evaluate(
            "() => window.lwPpocrDemo.recognize(undefined, {useCls: true})"
        )
        snapshot = page.evaluate("window.__lwOcrTest.snapshot()")
        assert snapshot["ready"] and snapshot["backend"] == "worker", snapshot
        assert snapshot["runCount"] == 1, snapshot
        assert snapshot["prepareCount"] == 1, snapshot
        assert snapshot["prepared"] and snapshot["hasResults"], snapshot
        assert snapshot["exportEnabled"], snapshot

        assert result["schema_version"] == 1
        assert result["source"] == sample.name
        assert result["options"] == {
            "use_cls": True,
            "reading_order": "horizontal-ltr",
        }
        assert result["image"]["width"] > 0 and result["image"]["height"] > 0
        assert result["elapsed_ms"] > 0
        assert len(result["lines"]) == 16
        timing = page.evaluate("window.__lwOcrTest.timingBreakdown()")
        assert timing["prepareMilliseconds"] > 0
        assert timing["sdkTotalMilliseconds"] > 0
        assert timing["uiMilliseconds"] >= 0
        expected_elapsed = round(
            timing["prepareMilliseconds"]
            + timing["sdkTotalMilliseconds"]
            + timing["uiMilliseconds"],
            3,
        )
        assert abs(result["elapsed_ms"] - expected_elapsed) <= 0.001, {
            "result": result["elapsed_ms"],
            "breakdown": timing,
        }

        lines = page.locator("#results .line")
        assert lines.count() == 16
        displayed_text = lines.locator("b").all_inner_texts()
        assert displayed_text[0] == EXPECTED_FIRST_LINE
        assert [line["text"] for line in result["lines"]] == displayed_text
        assert page.evaluate("window.lwPpocrDemo.getPlainText()") == "\n".join(
            displayed_text
        )
        assert page.locator("#overlay polygon").count() == 16
        assert page.locator("#overlay").is_visible()
        overlay_snapshot = page.evaluate("window.__lwOcrTest.snapshot()")
        page.locator("#toggle-overlay").click()
        assert page.locator("#toggle-overlay").inner_text() == "显示标注"
        assert page.locator("#toggle-overlay").get_attribute("aria-pressed") == "false"
        assert page.locator("#overlay").is_hidden()
        assert page.locator("#canvas").is_visible()
        hidden_snapshot = page.evaluate("window.__lwOcrTest.snapshot()")
        assert hidden_snapshot["overlayVisible"] is False
        assert hidden_snapshot["runCount"] == overlay_snapshot["runCount"]
        page.locator("#toggle-overlay").click()
        assert page.locator("#toggle-overlay").inner_text() == "隐藏标注"
        assert page.locator("#toggle-overlay").get_attribute("aria-pressed") == "true"
        assert page.locator("#overlay").is_visible()
        assert page.evaluate("window.__lwOcrTest.snapshot().overlayVisible") is True
        assert all(export_buttons.nth(index).is_enabled() for index in range(3))

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

        with page.expect_download() as json_download_info:
            page.locator("#export-json").click()
        json_download = json_download_info.value
        assert json_download.suggested_filename == f"{sample.stem}-ocr.json"
        json_path = json_download.path()
        assert json_path is not None
        exported = json.loads(Path(json_path).read_text(encoding="utf-8"))
        assert exported == result
        for line in exported["lines"]:
            assert len(line["box"]) == 8
            assert 0 <= line["det_score"] <= 1
            assert 0 <= line["rec_score"] <= 1
            assert 0 <= line["cls_score"] <= 1

        # A newly selected image invalidates old export data immediately.
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

        # The same file remains a desktop two-panel tool and a phone-friendly
        # camera/gallery experience. The sponsor QR is intentionally visible.
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.locator(".mobile-source-actions").is_visible()
        assert page.locator("#support").get_attribute("open") is not None
        sponsor_image = page.locator("#support img")
        assert sponsor_image.is_visible()
        sponsor_box = sponsor_image.bounding_box()
        assert sponsor_box is not None and sponsor_box["width"] >= 112, sponsor_box
        assert sponsor_image.get_attribute("src").startswith(
            "data:image/jpeg;base64,"
        )
        run_box = page.locator("#run").bounding_box()
        assert run_box is not None and run_box["height"] >= 44, run_box
        assert page.evaluate(
            "document.documentElement.scrollWidth <= "
            "document.documentElement.clientWidth"
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
