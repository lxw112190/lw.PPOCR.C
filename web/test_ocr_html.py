"""Run the standalone OCR page in Chromium and verify repeated inference."""

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
    arguments = parser.parse_args()

    html = arguments.html.resolve()
    sample = arguments.sample.resolve()
    if not html.is_file() or not sample.is_file():
        raise SystemExit("standalone HTML or sample image is missing")

    browser_messages: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=["--allow-file-access-from-files"],
        )
        page = browser.new_page()

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

        # Exercise the complete DET -> CLS -> REC path used by the product page.
        page.locator("#use-cls").check()
        page.wait_for_function(
            "() => window.__lwOcrTest.snapshot().ready && !document.querySelector('#use-cls').disabled",
            timeout=180_000,
        )
        page.locator("#file").set_input_files(str(sample))
        snapshot = wait_for_run(page, 0)

        lines = page.locator("#results .line")
        assert lines.count() == 16, page.locator("#status").inner_text()
        assert EXPECTED_FIRST_LINE in lines.first.locator(".text").inner_text()
        assert snapshot["maxLineCapacity"] == 1000, snapshot
        assert 0 < snapshot["maxTextCapacity"] < 1024 * 1024, snapshot

        # The first run may grow WASM memory. Identical later runs must reuse
        # the source/output buffers and keep the heap size unchanged.
        stable_heap = snapshot["heapBytes"]
        stable_source = snapshot["sourceCapacity"]
        for _ in range(4):
            previous_count = int(snapshot["runCount"])
            page.locator("#run").click()
            snapshot = wait_for_run(page, previous_count)
            assert snapshot["heapBytes"] == stable_heap, snapshot
            assert snapshot["sourceCapacity"] == stable_source, snapshot
            assert lines.count() == 16

        browser.close()

    if browser_messages:
        raise AssertionError("browser console diagnostics:\n" + "\n".join(browser_messages))
    print(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
