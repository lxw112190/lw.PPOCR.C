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

function parseJavaScript(source, label, ecmaVersion) {
  try {
    parse(source, {
      ecmaVersion: Number(ecmaVersion),
      sourceType: "script",
      allowHashBang: true
    });
  } catch (error) {
    const location = error.loc
      ? " line " + error.loc.line + ", column " + error.loc.column
      : "";
    throw new Error(label + ":" + location + ": " + error.message);
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
console.log("legacy whole-artifact ES" + ecmaVersion + " syntax gate: ok");