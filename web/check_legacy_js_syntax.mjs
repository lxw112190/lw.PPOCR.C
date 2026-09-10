#!/usr/bin/env node
import fs from "node:fs";
import process from "node:process";
import { parse } from "acorn";
import { parse as parseHtml } from "parse5";

function parseArgs(argv) {
  const values = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("--") || i + 1 >= argv.length) {
      throw new Error("expected --name value arguments");
    }
    values[token.slice(2)] = argv[++i];
  }
  for (const name of ["sdk", "html", "ecma-version"]) {
    if (!values[name]) throw new Error("missing --" + name);
  }
  return values;
}

function scriptText(node) {
  return (node.childNodes || [])
    .filter((child) => child.nodeName === "#text")
    .map((child) => child.value)
    .join("");
}

function formatParseError(source, label, error) {
  const location = error.loc
    ? " line " + error.loc.line + ", column " + error.loc.column
    : "";
  if (!error.loc) return new Error(label + ":" + location + ": " + error.message);

  const lines = source.split(/\r?\n/);
  const line = lines[error.loc.line - 1] || "";
  const start = Math.max(0, error.loc.column - 80);
  const excerpt = line.slice(start, error.loc.column + 80);
  const caret = " ".repeat(Math.max(0, error.loc.column - start)) + "^";
  return new Error(
    label + ":" + location + ": " + error.message + "\n" + excerpt + "\n" + caret
  );
}

function parseWithVersion(source, ecmaVersion) {
  return parse(source, {
    ecmaVersion,
    sourceType: "script",
    allowHashBang: true
  });
}

function walkAst(node, callback) {
  if (!node || typeof node !== "object") return;
  callback(node);
  for (const value of Object.values(node)) {
    if (Array.isArray(value)) {
      for (const child of value) walkAst(child, callback);
    } else if (value && typeof value === "object" && value.type) {
      walkAst(value, callback);
    }
  }
}

function rejectUnsupportedChrome70Syntax(ast, label) {
  let unsupported = null;
  walkAst(ast, (node) => {
    if (unsupported) return;
    if (node.type === "ChainExpression") {
      unsupported = "optional chaining";
    } else if (
      (node.type === "LogicalExpression" && node.operator === "??") ||
      (node.type === "AssignmentExpression" &&
        ["??=", "&&=", "||="].includes(node.operator))
    ) {
      unsupported = "nullish/logical assignment syntax";
    } else if (
      node.type === "MetaProperty" &&
      node.meta && node.meta.name === "import" &&
      node.property && node.property.name === "meta"
    ) {
      unsupported = "import.meta";
    } else if (
      node.type === "Literal" &&
      (typeof node.value === "number" || typeof node.value === "bigint") &&
      typeof node.raw === "string" && /_/.test(node.raw)
    ) {
      unsupported = "numeric separator syntax";
    } else if (node.type === "PrivateIdentifier" || node.type === "PropertyDefinition") {
      unsupported = "private/class-field syntax";
    } else if (node.type === "StaticBlock") {
      unsupported = "class static block syntax";
    }
  });
  if (unsupported) {
    throw new Error(label + ": unsupported by Chrome 70: " + unsupported);
  }
}

function parseJavaScript(source, label, ecmaVersion) {
  const requestedVersion = Number(ecmaVersion);
  try {
    const ast = parseWithVersion(source, requestedVersion);
    if (requestedVersion >= 2020) {
      rejectUnsupportedChrome70Syntax(ast, label);
    }
    return;
  } catch (error) {
    // Emscripten may emit BigInt literals for a 32-bit WASM build. They are
    // valid on Chrome 70, but Acorn's ES2018 grammar rejects them. Retry with
    // the ES2020 parser and then apply the explicit Chrome 70 feature gate.
    if (requestedVersion === 2018) {
      try {
        const ast = parseWithVersion(source, 2020);
        rejectUnsupportedChrome70Syntax(ast, label);
        return;
      } catch (compatibilityError) {
        if (compatibilityError.message.includes("unsupported by Chrome 70")) {
          throw compatibilityError;
        }
      }
    }
    throw formatParseError(source, label, error);
  }
}

function walk(node, callback) {
  callback(node);
  for (const child of node.childNodes || []) walk(child, callback);
}

const args = parseArgs(process.argv.slice(2));
const ecmaVersion = Number(args["ecma-version"]);
if (!Number.isInteger(ecmaVersion)) throw new Error("ecma-version must be an integer");

const sdk = fs.readFileSync(args.sdk, "utf8");
parseJavaScript(sdk, args.sdk, ecmaVersion);

const html = fs.readFileSync(args.html, "utf8");
const document = parseHtml(html);
let inlineScripts = 0;
walk(document, (node) => {
  if (node.nodeName !== "script") return;
  const type = (node.attrs || []).find((attr) => attr.name === "type");
  const scriptType = type ? type.value.toLowerCase() : "";
  if (scriptType === "application/json") return;
  if (scriptType === "module") {
    throw new Error(args.html + ": module scripts are not supported by Legacy");
  }
  parseJavaScript(scriptText(node), args.html + " <script>", ecmaVersion);
  inlineScripts += 1;
});
if (!inlineScripts) throw new Error(args.html + ": no inline JavaScript found");
console.log("legacy whole-artifact Chrome 70 syntax gate: ok");
