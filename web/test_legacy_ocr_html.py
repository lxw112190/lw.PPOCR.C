"""Validate the Tiny scalar Legacy HTML against the paired modern artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from playwright.sync_api import Browser, sync_playwright

EXPECTED_FIRST_LINE = "纯臻营养护发素"


def line_texts(result: dict) -> list[str]:
    return [line["text"] for line in result["lines"]]


def text_sha256(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def edit_distance(expected: str, actual: str) -> int:
    previous = list(range(len(actual) + 1))
    for expected_index, expected_character in enumerate(expected, start=1):
        current = [expected_index]
        for actual_index, actual_character in enumerate(actual, start=1):
            current.append(min(
                current[-1] + 1,
                previous[actual_index] + 1,
                previous[actual_index - 1] + (expected_character != actual_character),
            ))
        previous = current
    return previous[-1]


def recognize(browser: Browser, html: Path, sample: Path, repeats: int) -> tuple:
    page = browser.new_page()
    try:
        page.goto(html.resolve().as_uri(), wait_until="load", timeout=180_000)
        page.set_viewport_size({"width": 360, "height": 800})
        page.wait_for_function(
            "() => window.LwPpocr && window.__lwOcrTest", timeout=180_000
        )
        assert page.locator("#pick-camera").is_visible()
        assert page.locator("#pick-file").is_visible()
        assert page.locator("#pick-camera").is_enabled()
        assert page.locator("#pick-file").is_enabled()
        assert page.locator("#show-image").inner_text() == "\u9884\u89c8"
        assert page.locator("#show-results").inner_text() == "\u7ed3\u679c"
        with page.expect_file_chooser() as chooser_info:
            page.locator("#pick-file").click()
        chooser_info.value.set_files(str(sample.resolve()))
        page.wait_for_function(
            "() => window.__lwOcrTest.snapshot().sourceKind === 'image' && "
            "window.__lwOcrTest.snapshot().prepared", timeout=180_000
        )
        picker = page.evaluate("() => window.__lwOcrTest.snapshot().picker")
        assert picker["lastSource"] == "gallery", picker
        assert picker["state"] == "selected", picker
        assert picker["changeCount"] == 1, picker
        build = page.evaluate("() => window.LwPpocr.buildInfo")
        page.wait_for_function(
            "() => window.__lwOcrTest.snapshot().ready", timeout=180_000
        )
        page.wait_for_function(
            "() => !document.querySelector('#run').disabled", timeout=180_000
        )
        results = [
            page.evaluate(
                "() => window.lwPpocrDemo.recognize(undefined, {useCls: true})"
            )
            for _ in range(repeats)
        ]
        prepare_count = page.evaluate("() => window.__lwOcrTest.snapshot().prepareCount")
        with page.expect_file_chooser() as chooser_info:
            page.locator("#pick-file").click()
        chooser_info.value.set_files(str(sample.resolve()))
        page.wait_for_function(
            "() => window.__lwOcrTest.snapshot().prepareCount > " + str(prepare_count),
            timeout=180_000,
        )
        picker = page.evaluate("() => window.__lwOcrTest.snapshot().picker")
        assert picker["changeCount"] == 2, picker
        status = page.evaluate("() => window.__lwOcrTest.snapshot()")
        return build, results, status
    finally:
        page.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--reference-html", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    contract = json.loads(args.golden.read_text(encoding="utf-8"))
    expected_count = int(contract["expected_line_count"])
    expected_reference_hash = contract["expected_text_sha256"]
    started = time.perf_counter()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True, args=["--allow-file-access-from-files"]
        )
        try:
            reference_build, reference_results, _ = recognize(
                browser, args.reference_html, args.sample, 1
            )
            legacy_build, legacy_results, legacy_status = recognize(
                browser, args.html, args.sample, 2
            )
        finally:
            browser.close()

    reference_text = line_texts(reference_results[0])
    legacy_text = line_texts(legacy_results[0])
    legacy_repeat_text = line_texts(legacy_results[1])
    reference_hash = text_sha256(reference_text)
    legacy_hash = text_sha256(legacy_text)
    legacy_repeat_hash = text_sha256(legacy_repeat_text)

    compared_count = min(len(reference_text), len(legacy_text))
    exact_count = sum(
        reference_text[index] == legacy_text[index]
        for index in range(compared_count)
    )
    exact_line_rate = exact_count / expected_count
    reference_characters = sum(len(text) for text in reference_text)
    character_edits = sum(
        edit_distance(reference_text[index], legacy_text[index])
        for index in range(compared_count)
    )
    character_edits += sum(len(text) for text in reference_text[compared_count:])
    character_edits += sum(len(text) for text in legacy_text[compared_count:])
    character_error_rate = character_edits / max(1, reference_characters)
    mismatches = [
        {
            "index": index,
            "reference": reference_text[index],
            "legacy": legacy_text[index],
        }
        for index in range(compared_count)
        if reference_text[index] != legacy_text[index]
    ]

    report = {
        "schema_version": 1,
        "build_flavor": "legacy",
        "model": "PP-OCRv6 Tiny",
        "wasm_simd128": False,
        "pdf": False,
        "min_chrome": 70,
        "ocr_options": {"use_cls": True},
        "html_bytes": args.html.stat().st_size,
        "reference_build": reference_build,
        "legacy_build": legacy_build,
        "reference_text_sha256": reference_hash,
        "legacy_text_sha256": legacy_hash,
        "legacy_repeat_text_sha256": legacy_repeat_hash,
        "matches_reference_checksum": legacy_hash == reference_hash,
        "deterministic": legacy_text == legacy_repeat_text,
        "line_count": len(legacy_text),
        "exact_line_count": exact_count,
        "exact_line_rate": round(exact_line_rate, 6),
        "character_edit_distance": character_edits,
        "reference_character_count": reference_characters,
        "character_error_rate": round(character_error_rate, 6),
        "mismatches": mismatches,
        "ocr_ms": legacy_results[0].get("elapsed_ms"),
        "heap_bytes": legacy_status.get("heapBytes"),
        "wall_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False))

    assert reference_build["flavor"] == "modern", reference_build
    assert reference_hash == expected_reference_hash, report
    assert legacy_build == {
        "flavor": "legacy",
        "wasmSimd128": False,
        "pdf": False,
        "minChromeVersion": 70,
        "jsTarget": "chrome70",
    }, legacy_build
    assert len(reference_text) == expected_count, report
    assert len(legacy_text) == expected_count, report
    assert len(legacy_repeat_text) == expected_count, report
    assert legacy_text == legacy_repeat_text, report
    assert legacy_text[0] == EXPECTED_FIRST_LINE, report
    assert legacy_hash == expected_reference_hash, report
    assert legacy_repeat_hash == expected_reference_hash, report
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
