#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import process from "node:process";
import * as esbuild from "esbuild";

function parseArgs(argv) {
  const values = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("--") || i + 1 >= argv.length) {
      throw new Error("expected --name value arguments");
    }
    values[token.slice(2)] = argv[++i];
  }
  for (const name of ["input", "output", "target", "report"]) {
    if (!values[name]) throw new Error("missing --" + name);
  }
  return values;
}

function sha256(text) {
  return crypto.createHash("sha256").update(text, "utf8").digest("hex");
}

function count(source, pattern) {
  return (source.match(pattern) || []).length;
}

const args = parseArgs(process.argv.slice(2));
const source = fs.readFileSync(args.input, "utf8");
if (!source.includes("LwPpocrModule")) {
  throw new Error("input is not the expected Emscripten runtime");
}

const transformed = await esbuild.transform(source, {
  loader: "js",
  target: [args.target],
  minify: false,
  sourcemap: false,
  legalComments: "inline",
  charset: "utf8"
});

const prelude = `(function () {
  var root = typeof self !== "undefined"
    ? self
    : (typeof window !== "undefined" ? window : this);
  if (root && typeof root.globalThis === "undefined") root.globalThis = root;
})();`;
const output = prelude + "\n" + transformed.code;
if (!output.includes("LwPpocrModule")) {
  throw new Error("transpiled runtime lost LwPpocrModule");
}
if (!output.trim()) throw new Error("transpiled runtime is empty");

fs.mkdirSync(path.dirname(args.output), { recursive: true });
fs.writeFileSync(args.output, output, "utf8");
fs.mkdirSync(path.dirname(args.report), { recursive: true });
const report = {
  schema_version: 1,
  target: args.target,
  transpiler: "esbuild",
  transpiler_version: esbuild.version,
  input: {
    bytes: Buffer.byteLength(source, "utf8"),
    sha256: sha256(source)
  },
  output: {
    bytes: Buffer.byteLength(output, "utf8"),
    sha256: sha256(output)
  },
  features: {
    optional_chaining_before: count(source, /\?\.(?!\d)/g),
    optional_chaining_after: count(output, /\?\.(?!\d)/g),
    nullish_before: count(source, /\?\?(?!=)/g),
    nullish_after: count(output, /\?\?(?!=)/g),
    nullish_assignment_before: count(source, /\?\?=/g),
    nullish_assignment_after: count(output, /\?\?=/g)
  }
};
fs.writeFileSync(args.report, JSON.stringify(report, null, 2) + "\n", "utf8");
console.log(JSON.stringify(report));