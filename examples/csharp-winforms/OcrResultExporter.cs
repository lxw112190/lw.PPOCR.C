using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Web.Script.Serialization;
using LwPpocrCSharp;

namespace LwPpocrWinForms
{
    internal sealed class OcrExportSnapshot
    {
        internal OcrExportSnapshot(string plainText, string json, string baseName, int lineCount)
        {
            PlainText = plainText;
            Json = json;
            BaseName = baseName;
            LineCount = lineCount;
        }

        internal string PlainText { get; private set; }
        internal string Json { get; private set; }
        internal string BaseName { get; private set; }
        internal int LineCount { get; private set; }
    }

    internal static class OcrResultExporter
    {
        internal static OcrExportSnapshot Create(string sourcePath, OcrResponse response,
            bool useClassifier, double elapsedMilliseconds)
        {
            if (response == null) throw new ArgumentNullException("response");
            if (response.result == null) throw new ArgumentException("OCR result is missing", "response");
            if (Double.IsNaN(elapsedMilliseconds) || Double.IsInfinity(elapsedMilliseconds) ||
                elapsedMilliseconds < 0.0)
                throw new ArgumentOutOfRangeException("elapsedMilliseconds");

            string source = Path.GetFileName(sourcePath ?? String.Empty);
            string baseName = Path.GetFileNameWithoutExtension(source);
            if (String.IsNullOrEmpty(baseName)) baseName = "ocr-result";

            StringBuilder plainText = new StringBuilder();
            List<object> lines = new List<object>(response.result.Count);
            for (int index = 0; index < response.result.Count; ++index)
            {
                OcrLine line = response.result[index];
                if (index > 0) plainText.Append(Environment.NewLine);
                plainText.Append(line.text ?? String.Empty);

                Dictionary<string, object> exportedLine = new Dictionary<string, object>();
                exportedLine.Add("index", index);
                exportedLine.Add("text", line.text ?? String.Empty);
                exportedLine.Add("box", new float[] {
                    line.x1, line.y1, line.x2, line.y2,
                    line.x3, line.y3, line.x4, line.y4 });
                exportedLine.Add("det_score", line.det_score);
                exportedLine.Add("rec_score", line.score);
                if (useClassifier)
                {
                    exportedLine.Add("cls_score", line.cls_score);
                    exportedLine.Add("cls_label", line.cls_label);
                    exportedLine.Add("rotation_degrees", line.rotation);
                }
                lines.Add(exportedLine);
            }

            Dictionary<string, object> image = new Dictionary<string, object>();
            image.Add("width", response.image_width);
            image.Add("height", response.image_height);
            Dictionary<string, object> options = new Dictionary<string, object>();
            options.Add("use_cls", useClassifier);
            Dictionary<string, object> root = new Dictionary<string, object>();
            root.Add("schema_version", 1);
            root.Add("source", source);
            root.Add("image", image);
            root.Add("options", options);
            root.Add("elapsed_ms", elapsedMilliseconds);
            root.Add("lines", lines);

            JavaScriptSerializer serializer = new JavaScriptSerializer();
            serializer.MaxJsonLength = Int32.MaxValue;
            string json = PrettyPrint(serializer.Serialize(root)) + Environment.NewLine;
            return new OcrExportSnapshot(plainText.ToString(), json, baseName + "-ocr",
                response.result.Count);
        }

        private static string PrettyPrint(string json)
        {
            StringBuilder result = new StringBuilder(json.Length + 128);
            int indentation = 0;
            bool quoted = false;
            bool escaped = false;
            for (int index = 0; index < json.Length; ++index)
            {
                char value = json[index];
                if (quoted)
                {
                    result.Append(value);
                    if (escaped) escaped = false;
                    else if (value == '\\') escaped = true;
                    else if (value == '"') quoted = false;
                    continue;
                }

                if (value == '"')
                {
                    quoted = true;
                    result.Append(value);
                }
                else if (value == '{' || value == '[')
                {
                    result.Append(value);
                    char closing = value == '{' ? '}' : ']';
                    if (index + 1 < json.Length && json[index + 1] == closing) continue;
                    result.AppendLine();
                    ++indentation;
                    AppendIndentation(result, indentation);
                }
                else if (value == '}' || value == ']')
                {
                    char opening = value == '}' ? '{' : '[';
                    if (index > 0 && json[index - 1] == opening)
                    {
                        result.Append(value);
                        continue;
                    }
                    result.AppendLine();
                    --indentation;
                    AppendIndentation(result, indentation);
                    result.Append(value);
                }
                else if (value == ',')
                {
                    result.Append(value);
                    result.AppendLine();
                    AppendIndentation(result, indentation);
                }
                else if (value == ':') result.Append(": ");
                else if (!Char.IsWhiteSpace(value)) result.Append(value);
            }
            return result.ToString();
        }

        private static void AppendIndentation(StringBuilder target, int indentation)
        {
            target.Append(' ', indentation * 2);
        }
    }
}
