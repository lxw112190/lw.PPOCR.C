using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Windows.Forms;
using LwPpocrCSharp;

internal static class WinFormsPackageSmoke
{
    [STAThread]
    private static int Main(string[] args)
    {
        try
        {
            if (args.Length != 2) throw new ArgumentException("usage: smoke <package> <golden-dir>");
            string package = Path.GetFullPath(args[0]);
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            // Load the actual packaged assembly, not a duplicate compile of its sources.
            Assembly assembly = typeof(NativeOcr).Assembly;
            string version = System.Text.RegularExpressions.Regex.Match(
                File.ReadAllText(Path.Combine(package, "BUILD-INFO.txt")),
                @"Product: lw\.PPOCR\.C ([0-9]+\.[0-9]+\.[0-9]+)").Groups[1].Value;
            if (version.Length == 0 || assembly.GetName().Version.ToString() != version + ".0")
                throw new InvalidOperationException("unexpected demo version");
            Type formType = assembly.GetType("LwPpocrWinForms.MainForm", true);
            using (Form form = (Form)Activator.CreateInstance(formType, new object[] { Path.Combine(package, "models") }))
            {
                if (form.Controls.Count == 0) throw new InvalidOperationException("empty WinForms UI");
            }
            foreach (string variant in new string[] { "tiny", "small", "medium" })
            {
                string models = variant == "tiny" ? Path.Combine(package, "models") : Path.Combine(package, "models", variant);
                string golden = File.ReadAllText(Path.Combine(args[1], "web-ppocrv6-" + variant + ".json"));
                string expected = System.Text.RegularExpressions.Regex.Match(golden,
                    "\"expected_text_sha256\"\\s*:\\s*\"([a-f0-9]{64})\"").Groups[1].Value;
                if (expected.Length != 64) throw new InvalidOperationException("missing golden checksum");
                foreach (uint workers in new uint[] { 1u, 4u })
                using (NativeOcr engine = new NativeOcr(Path.Combine(models, "det.onnx"),
                    Path.Combine(models, "cls.onnx"), Path.Combine(models, "rec.onnx"),
                    Path.Combine(models, "ppocr_keys.txt"), true, workers))
                {
                    OcrResponse result = engine.RecognizeEncoded(File.ReadAllBytes(Path.Combine(models, "sample.jpg")));
                    if (!result.ok || result.result.Count != 16) throw new InvalidOperationException("invalid result");
                    string text = String.Join("\n", result.result.Select(line => line.text));
                    string actual;
                    using (SHA256 hash = SHA256.Create())
                        actual = BitConverter.ToString(hash.ComputeHash(Encoding.UTF8.GetBytes(text))).Replace("-", "").ToLowerInvariant();
                    if (actual != expected) throw new InvalidOperationException(variant + " full text checksum: " + actual);
                    Console.WriteLine(variant + " workers=" + workers + " lines=16 sha256=" + actual);
                }
            }
            Console.WriteLine("packaged WinForms constructor and six real OCR calls: PASS");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            return 1;
        }
    }
}
