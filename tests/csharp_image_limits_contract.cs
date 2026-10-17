using System;
using System.IO;
using System.Reflection;
using System.Windows.Forms;
using LwPpocrCSharp;
using LwPpocrWinForms;

internal static class ImageLimitsContract
{
    private static int checks;

    private static void Check(bool value, string message)
    {
        if (!value) throw new InvalidOperationException(message);
        ++checks;
    }

    private static T Reject<T>(Action action) where T : Exception
    {
        try { action(); }
        catch (T error) { ++checks; return error; }
        throw new InvalidOperationException("expected " + typeof(T).Name);
    }

    private static NativeOcr Engine(string models, ulong limit)
    {
        return new NativeOcr(Path.Combine(models, "det.onnx"), Path.Combine(models, "cls.onnx"),
            Path.Combine(models, "rec.onnx"), Path.Combine(models, "ppocr_keys.txt"), true, 1u, limit);
    }

    [STAThread]
    private static int Main(string[] args)
    {
        try
        {
            string models = args[0];
            byte[] sample = File.ReadAllBytes(args[1]);
            byte[] large = File.ReadAllBytes(args[2]);
            Check(NativeOcr.ValidateImageDimensions(8000, 5000, 40000000UL) == 40000000L, "inclusive default boundary");
            InvalidOperationException error = Reject<InvalidOperationException>(delegate {
                NativeOcr.ValidateImageDimensions(8000, 5001, 40000000UL);
            });
            Check(error.Message.Contains("8000 x 5001") && error.Message.Contains("40008000") &&
                error.Message.Contains("40000000"), "dimension diagnostics");
            Check(NativeOcr.ValidateImageDimensions(8000, 5200, 50000000UL) == 41600000L, "custom boundary");
            Check(NativeOcr.ValidateImageDimensions(1, 1, 1UL) == 1L, "one-pixel lower boundary");
            Reject<InvalidOperationException>(delegate { NativeOcr.ValidateImageDimensions(-1, 1, 40000000UL); });
            Reject<InvalidOperationException>(delegate { NativeOcr.ValidateImageDimensions(0, 1, 40000000UL); });
            Reject<InvalidOperationException>(delegate { NativeOcr.ValidateImageDimensions(Int32.MaxValue, Int32.MaxValue, 50000000UL); });
            Reject<ArgumentOutOfRangeException>(delegate { Engine(models, 0UL).Dispose(); });
            Reject<ArgumentOutOfRangeException>(delegate { Engine(models, 200000001UL).Dispose(); });
            Reject<ArgumentOutOfRangeException>(delegate { Engine(models, UInt64.MaxValue).Dispose(); });
            Reject<ArgumentException>(delegate { NativeOcr.DecodeToBgr(new byte[0], 50000000UL); });

            // Old constructors retain their exact source signatures and 40 MP default.
            using (NativeOcr legacy = new NativeOcr(Path.Combine(models, "det.onnx"), Path.Combine(models, "cls.onnx"),
                Path.Combine(models, "rec.onnx"), Path.Combine(models, "ppocr_keys.txt"), true))
            {
                Check(legacy.MaxImagePixels == 40000000UL, "legacy default");
                Reject<InvalidOperationException>(delegate { legacy.RecognizeEncoded(large); });
                Check(legacy.RecognizeEncoded(sample).result.Count == 16, "recovery after rejection");
            }
            using (NativeOcr exact = Engine(models, 250000UL))
            {
                Check(exact.RecognizeEncoded(sample).result.Count == 16, "exact 500x500 boundary");
            }
            using (NativeOcr low = Engine(models, 249999UL))
            {
                Reject<InvalidOperationException>(delegate { low.RecognizeEncoded(sample); });
                DecodedBgrImage decoded = NativeOcr.DecodeToBgr(sample);
                Reject<InvalidOperationException>(delegate { low.RecognizeDecoded(decoded); });
            }
            using (NativeOcr custom = Engine(models, 50000000UL))
            {
                Check(custom.MaxImagePixels == 50000000UL, "instance policy");
                LwOcrInfo info = (LwOcrInfo)typeof(NativeOcr).GetField("info", BindingFlags.Instance | BindingFlags.NonPublic).GetValue(custom);
                Check(info.MaxCropPixels == 16000000UL, "separate crop policy unchanged");
                OcrResponse result = custom.RecognizeEncoded(large);
                Check(result.ok && result.image_width == 8000 && result.image_height == 5200,
                    "real 41.6 MP decode and native DET must accept custom policy");
                Check(result.result.Count > 0 && result.detected_count > 0 &&
                    result.result[0].text == "纯臻营养护发素", "large image runs full text OCR");
                Check(custom.RecognizeEncoded(sample).result.Count == 16, "small image after large image");
                Console.WriteLine("41.6 MP real decode + native OCR: PASS (detected=" + result.detected_count + ")");
            }

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            using (MainForm form = new MainForm(models))
            {
                Type type = typeof(MainForm);
                BindingFlags flags = BindingFlags.NonPublic | BindingFlags.Instance;
                NumericUpDown input = (NumericUpDown)type.GetField("maxImageMegapixelsInput", flags).GetValue(form);
                Check(input.Value == 40 && input.Maximum == 200, "WinForms default/range");
                MethodInfo ensure = type.GetMethod("EnsureEngine", flags);
                FieldInfo engineField = type.GetField("engine", flags);
                ensure.Invoke(form, null);
                NativeOcr first = (NativeOcr)engineField.GetValue(form);
                ensure.Invoke(form, null);
                Check(Object.ReferenceEquals(first, engineField.GetValue(form)), "unchanged settings reuse engine");
                input.Value = 50;
                ensure.Invoke(form, null);
                NativeOcr second = (NativeOcr)engineField.GetValue(form);
                Check(!Object.ReferenceEquals(first, second) && second.MaxImagePixels == 50000000UL,
                    "pixel policy change rebuilds engine");
                type.GetMethod("SetBusy", flags).Invoke(form, new object[] { true });
                Check(!input.Enabled, "disable settings during recognition");
                type.GetMethod("SetBusy", flags).Invoke(form, new object[] { false });
                Check(input.Enabled, "reenable settings");
                // Preview uses the same bound and retains the previous image on failure.
                string imagePath = (string)type.GetField("imagePath", flags).GetValue(form);
                input.Value = 40;
                TargetInvocationException previewError = Reject<TargetInvocationException>(delegate {
                    type.GetMethod("LoadImage", flags).Invoke(form, new object[] { args[2] });
                });
                Check(previewError.InnerException is InvalidOperationException, "preview policy");
                Check(imagePath == (string)type.GetField("imagePath", flags).GetValue(form), "failed preview preserves selection");
                type.GetMethod("DisposeRuntimeObjects", flags).Invoke(form, null);
            }
            Console.WriteLine("C# image limits: " + checks + " checks passed");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error);
            return 1;
        }
    }
}
