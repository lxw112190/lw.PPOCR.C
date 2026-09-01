#!/usr/bin/env node

/* Dependency-free Node smoke and Full OCR regression for runtime.cjs. */
"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");

function argument(name, fallback = null) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : fallback;
}

function fail(message) {
  throw new Error(message);
}

function packagePath(root, relative) {
  const resolvedRoot = path.resolve(root);
  const resolved = path.resolve(root, relative);
  if (resolved !== resolvedRoot && !resolved.startsWith(`${resolvedRoot}${path.sep}`)) {
    fail(`package path escapes archive root: ${relative}`);
  }
  return resolved;
}

function readPpm(file) {
  const bytes = fs.readFileSync(file);
  let offset = 0;
  function token() {
    while (offset < bytes.length && /\s/.test(String.fromCharCode(bytes[offset]))) offset++;
    if (bytes[offset] === 35) {
      while (offset < bytes.length && bytes[offset] !== 10) offset++;
      return token();
    }
    const start = offset;
    while (offset < bytes.length && !/\s/.test(String.fromCharCode(bytes[offset]))) offset++;
    return bytes.subarray(start, offset).toString("ascii");
  }
  if (token() !== "P6") fail("Node smoke fixture must be a binary PPM (P6)");
  const width = Number(token());
  const height = Number(token());
  const maximum = Number(token());
  if (!Number.isInteger(width) || !Number.isInteger(height) || maximum !== 255) {
    fail("invalid PPM header");
  }
  // Consume the single separator after maxval. Do not skip an arbitrary
  // number of bytes here: a binary PPM payload may legitimately start with a
  // pixel component whose value is <= 32.
  if (bytes[offset] === 13 && bytes[offset + 1] === 10) offset += 2;
  else if (bytes[offset] <= 32) offset++;
  const rgb = bytes.subarray(offset);
  if (rgb.length !== width * height * 3) fail("invalid PPM payload length");
  const bgr = Buffer.allocUnsafe(rgb.length);
  for (let i = 0; i < rgb.length; i += 3) {
    bgr[i] = rgb[i + 2];
    bgr[i + 1] = rgb[i + 1];
    bgr[i + 2] = rgb[i];
  }
  return {width, height, bgr};
}

function verifyPackage(root) {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, "manifest.json"), "utf8"));
  if (manifest.schemaVersion !== 1 || manifest.runtime.target !== "node-wasm") {
    fail("unsupported Node/WASM manifest");
  }
  for (const asset of Object.values(manifest.assets)) {
    const file = packagePath(root, asset.path);
    const digest = crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
    if (digest !== asset.sha256) fail(`manifest checksum mismatch: ${asset.path}`);
  }
  const checksumLines = fs.readFileSync(packagePath(root, "SHA256SUMS.txt"), "utf8")
    .trim().split(/\r?\n/);
  if (checksumLines.length < 1) fail("empty package checksum list");
  for (const line of checksumLines) {
    const match = line.match(/^([0-9a-f]{64})  (.+)$/);
    if (!match) fail(`invalid package checksum line: ${line}`);
    const file = packagePath(root, match[2]);
    const digest = crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
    if (digest !== match[1]) fail(`package checksum mismatch: ${match[2]}`);
  }
  return manifest;
}

async function createRuntime(root, useCls) {
  const factory = require(path.join(root, "runtime.cjs"));
  const runtime = await factory({});
  runtime.FS.mkdir("/models");
  for (const name of ["det.lwm", "cls.lwm", "rec.lwm", "ppocr_keys.txt"]) {
    runtime.FS.writeFile(`/models/${name}`, fs.readFileSync(path.join(root, name)));
  }
  const status = runtime._lw_web_init(useCls ? 1 : 0);
  if (status !== 0) fail(`lw_web_init failed: ${status}`);
  const infoPointer = runtime._lw_web_malloc(20);
  if (!infoPointer || runtime._lw_web_get_info(infoPointer) !== 0) fail("unable to query ABI info");
  const u32 = pointer => runtime.HEAPU32[pointer >> 2] >>> 0;
  const info = {
    abi: u32(infoPointer),
    maxLines: u32(infoPointer + 4),
    maxText: u32(infoPointer + 8),
    lineSize: u32(infoPointer + 12),
    resultSize: u32(infoPointer + 16)
  };
  runtime._lw_web_free(infoPointer);
  if (info.abi !== 1 || info.lineSize !== 60 || info.resultSize !== 16) fail("unsupported WASM Host ABI");
  return {runtime, info, useCls};
}

function runOcr(engine, image) {
  const {runtime, info} = engine;
  const source = runtime._lw_web_malloc(image.bgr.length);
  const lines = runtime._lw_web_malloc(info.maxLines * info.lineSize);
  const text = runtime._lw_web_malloc(info.maxText);
  const result = runtime._lw_web_malloc(info.resultSize);
  if (!source || !lines || !text || !result) fail("WASM output allocation failed");
  try {
    runtime.HEAPU8.set(image.bgr, source);
    const status = runtime._lw_web_run(
      source, image.bgr.length, image.width, image.height, image.width * 3,
      lines, info.maxLines, text, info.maxText, result
    );
    if (status !== 0) fail(`lw_web_run failed: ${status}`);
    const lineCount = runtime.HEAPU32[result >> 2] >>> 0;
    if (lineCount === 0 || lineCount > info.maxLines) fail("invalid OCR line count");
    const decoder = new TextDecoder();
    const output = [];
    for (let index = 0; index < lineCount; index++) {
      const pointer = lines + index * info.lineSize;
      const offset = runtime.HEAPU32[(pointer + 52) >> 2] >>> 0;
      const length = runtime.HEAPU32[(pointer + 56) >> 2] >>> 0;
      if (offset > info.maxText || length > info.maxText - offset) fail("invalid OCR text range");
      output.push({
        text: decoder.decode(runtime.HEAPU8.subarray(text + offset, text + offset + length)),
        detection: runtime.HEAPF32[(pointer + 32) >> 2],
        recognition: runtime.HEAPF32[(pointer + 36) >> 2]
      });
    }
    return output;
  } finally {
    for (const pointer of [source, lines, text, result]) runtime._lw_web_free(pointer);
  }
}

async function main() {
  const packageArgument = argument("--package");
  const sampleArgument = argument("--sample");
  if (!packageArgument || !sampleArgument) {
    fail("usage: smoke.cjs --package <directory> --sample <P6 ppm>");
  }
  const root = path.resolve(packageArgument);
  const sample = path.resolve(sampleArgument);
  if (!fs.existsSync(root) || !fs.existsSync(sample)) fail("Node/WASM package or PPM fixture is missing");
  const manifest = verifyPackage(root);
  if (manifest.compatibility.node !== ">=18") fail("unexpected Node compatibility declaration");
  const expectedBackend = argument("--expected-backend");
  if (expectedBackend) {
    if (manifest.runtime.backend !== expectedBackend) {
      fail(`unexpected WASM backend: ${manifest.runtime.backend}`);
    }
    const simdEnabled = manifest.runtime.simd && manifest.runtime.simd.wasm128 === true;
    if (simdEnabled !== (expectedBackend === "wasm128")) {
      fail("manifest SIMD capability does not match the declared backend");
    }
  }
  const image = readPpm(sample);
  let expectedTexts = null;
  for (const useCls of [true, false, true]) {
    const engine = await createRuntime(root, useCls);
    const lines = runOcr(engine, image);
    if (!expectedTexts) expectedTexts = lines.map(line => line.text);
    if (useCls && lines.map(line => line.text).join("\n") !== expectedTexts.join("\n")) {
      fail("Node OCR text changed across lifecycle runs");
    }
    if (lines.some(line => !Number.isFinite(line.detection) || !Number.isFinite(line.recognition))) {
      fail("Node OCR returned a non-finite score");
    }
    engine.runtime._lw_web_shutdown();
  }
  const expectedCount = Number(argument("--expected-line-count", "16"));
  if (expectedTexts.length !== expectedCount) fail(`expected ${expectedCount} OCR lines, got ${expectedTexts.length}`);
  const expectedFirst = argument("--expected-first-line", "纯臻营养护发素");
  if (expectedFirst && expectedTexts[0] !== expectedFirst) fail(`unexpected first OCR line: ${expectedTexts[0]}`);
  const textSha256 = crypto.createHash("sha256").update(expectedTexts.join("\n"), "utf8").digest("hex");
  const expectedTextSha256 = argument("--expected-text-sha256");
  if (expectedTextSha256 && textSha256 !== expectedTextSha256) {
    fail(`unexpected full OCR text SHA-256: ${textSha256}`);
  }
  console.log(JSON.stringify({
    ok: true,
    node: process.versions.node,
    lines: expectedTexts.length,
    firstLine: expectedTexts[0],
    textSha256
  }));
}

main().catch(error => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exitCode = 1;
});
