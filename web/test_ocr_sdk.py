"""Verify the public browser SDK contract with the bundled OCR sample."""

from __future__ import annotations

import argparse
import json
import tempfile
from html import escape
from pathlib import Path

from playwright.sync_api import ConsoleMessage, Page, sync_playwright


EXPECTED_FIRST_LINE = "纯臻营养护发素"


def run_sample(page: Page) -> dict:
    return page.evaluate(
        """async () => {
          const file = document.querySelector("#sample").files[0];
          const result = await window.__sdkEngine.recognize(file);
          return {result, status: window.__sdkEngine.getStatus()};
        }"""
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument(
        "--browser-executable",
        type=Path,
        help="optional local Chrome/Edge executable; CI uses Playwright Chromium",
    )
    arguments = parser.parse_args()

    sdk = arguments.sdk.resolve()
    sample = arguments.sample.resolve()
    if not sdk.is_file() or not sample.is_file():
        raise SystemExit("browser SDK or sample image is missing")

    browser_messages: list[str] = []
    with tempfile.TemporaryDirectory(prefix="lw-ppocr-sdk-") as temporary:
        page_path = Path(temporary) / "sdk-smoke.html"
        page_path.write_text(
            "<!doctype html><meta charset='utf-8'>"
            "<input id='sample' type='file' accept='image/*'>"
            "<canvas id='source'></canvas>"
            f"<script src='{escape(sdk.as_uri(), quote=True)}'></script>",
            encoding="utf-8",
        )

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

            def capture_console(message: ConsoleMessage) -> None:
                if message.type == "error":
                    browser_messages.append(f"{message.type}: {message.text}")

            page.on("console", capture_console)
            page.on(
                "pageerror", lambda error: browser_messages.append(f"pageerror: {error}")
            )
            page.goto(page_path.as_uri(), wait_until="load", timeout=180_000)
            page.wait_for_function("() => window.LwPpocr", timeout=180_000)
            page.locator("#sample").set_input_files(str(sample))

            public_contract = page.evaluate(
                """async () => {
                  let invalidOptionsCode = "";
                  try {
                    await LwPpocr.create({maxImageSide: -1});
                  } catch (error) {
                    invalidOptionsCode = error.code;
                  }
                  window.__sdkEngine = await LwPpocr.create({
                    useCls: true,
                    maxImageSide: 1600
                  });
                  return {
                    version: LwPpocr.version,
                    webAbiVersion: LwPpocr.webAbiVersion,
                    frozen: Object.isFrozen(LwPpocr),
                    invalidOptionsCode,
                    status: window.__sdkEngine.getStatus()
                  };
                }"""
            )
            assert public_contract["version"]
            assert public_contract["webAbiVersion"] == 1
            assert public_contract["frozen"]
            assert public_contract["invalidOptionsCode"] == "LW_OCR_OPTIONS"
            assert public_contract["status"]["state"] == "READY"
            assert public_contract["status"]["backend"] == "worker"

            # recognize() changes state synchronously, so a second overlapping
            # request must fail predictably instead of corrupting shared buffers.
            first = page.evaluate(
                """async () => {
                  const file = document.querySelector("#sample").files[0];
                  const pending = window.__sdkEngine.recognize(file);
                  let busyCode = "";
                  try {
                    await window.__sdkEngine.recognize(file);
                  } catch (error) {
                    busyCode = error.code;
                  }
                  const result = await pending;
                  return {
                    busyCode,
                    result,
                    status: window.__sdkEngine.getStatus()
                  };
                }"""
            )
            assert first["busyCode"] == "LW_OCR_BUSY"
            result = first["result"]
            assert result["result_version"] == 1
            assert result["source"] == sample.name
            assert result["image"]["width"] > 0 and result["image"]["height"] > 0
            assert result["options"] == {"use_cls": True}
            assert result["timing"]["decode_ms"] >= 0
            assert result["timing"]["inference_ms"] > 0
            assert result["timing"]["total_ms"] >= result["timing"]["inference_ms"]
            assert len(result["lines"]) == 16
            assert result["lines"][0]["text"] == EXPECTED_FIRST_LINE
            expected_text = [line["text"] for line in result["lines"]]
            expected_boxes = [line["box"] for line in result["lines"]]
            for line in result["lines"]:
                assert len(line["box"]) == 8
                assert 0 <= line["det_score"] <= 1
                assert 0 <= line["rec_score"] <= 1
                assert 0 <= line["cls_score"] <= 1
                assert line["cls_label"] in (0, 1)
                assert line["rotation_degrees"] in (0, 180)

            # All documented source types are accepted. Canvas and ImageData
            # intentionally report "image" because neither has a filename.
            source_results = page.evaluate(
                """async () => {
                  const file = document.querySelector("#sample").files[0];
                  const bitmap = await createImageBitmap(file);
                  const canvas = document.querySelector("#source");
                  canvas.width = bitmap.width;
                  canvas.height = bitmap.height;
                  const context = canvas.getContext("2d");
                  context.drawImage(bitmap, 0, 0);
                  bitmap.close();
                  const blob = new Blob([await file.arrayBuffer()], {type: file.type});
                  const fromBlob = await window.__sdkEngine.recognize(blob);
                  const fromCanvas = await window.__sdkEngine.recognize(canvas);
                  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
                  const fromImageData = await window.__sdkEngine.recognize(imageData);
                  return {
                    blob: fromBlob,
                    canvas: fromCanvas,
                    imageData: fromImageData,
                    status: window.__sdkEngine.getStatus()
                  };
                }"""
            )
            for source_result in (
                source_results["blob"],
                source_results["canvas"],
                source_results["imageData"],
            ):
                assert source_result["source"] == "image"
                assert [line["text"] for line in source_result["lines"]] == expected_text
                assert source_result["image"] == result["image"]

            # Adaptive REC widths can grow the Emscripten heap during bounded
            # warm-up. Verify a stable high-water mark only after that phase.
            snapshot = source_results["status"]
            initial_heap = int(snapshot["heapBytes"])
            heap_history = [initial_heap]
            warmup_runs = 16
            verification_runs = 6
            verification_heaps: list[int] = []
            for run_index in range(warmup_runs + verification_runs):
                current = run_sample(page)
                current_result = current["result"]
                snapshot = current["status"]
                assert [line["text"] for line in current_result["lines"]] == expected_text
                assert [line["box"] for line in current_result["lines"]] == expected_boxes
                heap_bytes = int(snapshot["heapBytes"])
                heap_history.append(heap_bytes)
                if run_index >= warmup_runs:
                    verification_heaps.append(heap_bytes)

            assert len(set(verification_heaps)) == 1, {
                "status": snapshot,
                "heapHistory": heap_history,
                "verificationHeaps": verification_heaps,
            }
            assert heap_history[-1] <= initial_heap * 2, {
                "status": snapshot,
                "heapHistory": heap_history,
            }

            lifecycle = page.evaluate(
                """async () => {
                  window.__sdkEngine.destroy();
                  window.__sdkEngine.destroy();
                  let destroyedCode = "";
                  try {
                    await window.__sdkEngine.recognize(
                      document.querySelector("#sample").files[0]
                    );
                  } catch (error) {
                    destroyedCode = error.code;
                  }
                  const destroyedStatus = window.__sdkEngine.getStatus();
                  const withoutCls = await LwPpocr.create({
                    useCls: false,
                    maxImageSide: 1600
                  });
                  const result = await withoutCls.recognize(
                    document.querySelector("#sample").files[0]
                  );
                  const status = withoutCls.getStatus();
                  withoutCls.destroy();
                  return {destroyedCode, destroyedStatus, result, status};
                }"""
            )
            assert lifecycle["destroyedCode"] == "LW_OCR_DESTROYED"
            assert lifecycle["destroyedStatus"]["state"] == "DESTROYED"
            without_cls = lifecycle["result"]
            assert without_cls["options"] == {"use_cls": False}
            # Disabling CLS intentionally changes orientation handling, so its
            # text is not required to match the CLS-enabled checksum.
            assert len(without_cls["lines"]) == 16
            for line in without_cls["lines"]:
                assert "cls_score" not in line
                assert "cls_label" not in line
                assert "rotation_degrees" not in line

            # Product documentation promises a main-thread fallback when a
            # browser or CSP rejects Blob Workers. Force the Worker constructor
            # to fail and run the same real-model Golden OCR through that path.
            fallback_page = browser.new_page()
            fallback_page.add_init_script(
                """
                Object.defineProperty(window, "Worker", {
                  configurable: true,
                  value: function Worker() {
                    throw new Error("Worker blocked by SDK fallback test");
                  }
                });
                """
            )
            fallback_page.on("console", capture_console)
            fallback_page.on(
                "pageerror",
                lambda error: browser_messages.append(f"pageerror: {error}"),
            )
            fallback_page.goto(page_path.as_uri(), wait_until="load", timeout=180_000)
            fallback_page.wait_for_function("() => window.LwPpocr", timeout=180_000)
            fallback_page.locator("#sample").set_input_files(str(sample))
            fallback = fallback_page.evaluate(
                """async () => {
                  const engine = await LwPpocr.create({
                    useCls: true,
                    maxImageSide: 1600
                  });
                  const readyStatus = engine.getStatus();
                  const result = await engine.recognize(
                    document.querySelector("#sample").files[0]
                  );
                  engine.destroy();
                  return {
                    readyStatus,
                    destroyedStatus: engine.getStatus(),
                    result
                  };
                }"""
            )
            assert fallback["readyStatus"]["backend"] == "main-thread", fallback
            assert fallback["readyStatus"]["state"] == "READY", fallback
            assert len(fallback["result"]["lines"]) == 16
            assert fallback["result"]["lines"][0]["text"] == EXPECTED_FIRST_LINE
            assert fallback["destroyedStatus"]["state"] == "DESTROYED"
            fallback_page.close()

            browser.close()

    if browser_messages:
        raise AssertionError("browser console diagnostics:\n" + "\n".join(browser_messages))
    print(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
