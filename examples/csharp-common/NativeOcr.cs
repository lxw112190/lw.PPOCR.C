using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace LwPpocrCSharp
{
    // Preserve detail for wide document lines (news pages, receipts, tables).
    // The native REC graph supports dynamic widths; 320 is faster but can
    // compress a long line into too few pixels for reliable recognition.
    internal static class OcrRecognitionDefaults
    {
        internal const uint LongTextTargetWidth = 960u;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct LwError
    {
        public uint StructSize;
        public int Code;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256, ArraySubType = UnmanagedType.I1)]
        public byte[] Message;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct LwDetectorOptions
    {
        public uint StructSize;
        public uint LimitSideLength;
        public uint MaxCandidates;
        public uint UseDilation;
        public float BitmapThreshold;
        public float BoxThreshold;
        public float UnclipRatio;
        public uint Reserved;
        public ulong MaxModelFileSize;
        public ulong MaxWorkspaceSize;
        public ulong MaxTensorSize;
        public ulong MaxImagePixels;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct LwClassifierOptions
    {
        public uint StructSize;
        public uint Reserved;
        public ulong MaxModelFileSize;
        public ulong MaxWorkspaceSize;
        public ulong MaxTensorSize;
        public ulong MaxImagePixels;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct LwRecognizerOptions
    {
        public uint StructSize;
        public uint TargetWidth;
        public uint Reserved0;
        public uint Reserved1;
        public ulong MaxModelFileSize;
        public ulong MaxWorkspaceSize;
        public ulong MaxTensorSize;
        public ulong MaxImagePixels;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct LwOcrOptions
    {
        public uint StructSize;
        public uint UseDirectionClassification;
        public float ClassifierThreshold;
        public uint WorkerCount;
        public ulong MaxCropPixels;
        public LwDetectorOptions Detector;
        public LwClassifierOptions Classifier;
        public LwRecognizerOptions Recognizer;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct LwOcrInfo
    {
        public uint StructSize;
        public uint UseDirectionClassification;
        public uint MaxLineCapacity;
        public uint WorkerCount;
        public ulong MaxTextCapacity;
        public ulong MaxTextCapacityPerLine;
        public ulong MaxCropPixels;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct LwDetectionBox
    {
        public float X1;
        public float Y1;
        public float X2;
        public float Y2;
        public float X3;
        public float Y3;
        public float X4;
        public float Y4;
        public float Score;
        public uint Reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct LwOcrLine
    {
        public LwDetectionBox Box;
        public float RecognitionScore;
        public float ClassificationScore;
        public uint ClassificationLabel;
        public uint AppliedRotationDegrees;
        public uint EmittedCount;
        public uint Reserved;
        public ulong TextOffset;
        public ulong TextLength;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct LwOcrResult
    {
        public uint StructSize;
        public uint LineCount;
        public uint RequiredLineCapacity;
        public uint DetectedCount;
        public uint DetectorResizedWidth;
        public uint DetectorResizedHeight;
        public uint Reserved0;
        public uint Reserved1;
        public ulong RequiredTextCapacity;
    }

    public sealed class NativeOcr : IDisposable
    {
        private const string DllName = "lw_ppocr_c.dll";
        private const long MaxDecodedPixels = 40000000L;
        private readonly object syncRoot = new object();
        private IntPtr handle;
        private LwOcrInfo info;
        private bool disposed;

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void lw_error_init(ref LwError error);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void lw_ocr_options_init(ref LwOcrOptions options);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void lw_ocr_info_init(ref LwOcrInfo value);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void lw_ocr_result_init(ref LwOcrResult value);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int lw_ocr_create(
            IntPtr detectorPathUtf8,
            IntPtr classifierPathUtf8,
            IntPtr recognizerPathUtf8,
            IntPtr dictionaryPathUtf8,
            ref LwOcrOptions options,
            out IntPtr ocr,
            ref LwError error);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void lw_ocr_free(IntPtr ocr);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int lw_ocr_get_info(IntPtr ocr, ref LwOcrInfo value);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int lw_ocr_run_bgr_u8(
            IntPtr ocr,
            IntPtr source,
            ulong sourceByteCount,
            uint sourceWidth,
            uint sourceHeight,
            uint sourceStride,
            IntPtr lines,
            uint lineCapacity,
            IntPtr textUtf8,
            ulong textCapacity,
            ref LwOcrResult result,
            ref LwError error);

        public NativeOcr(
            string detectorPath,
            string classifierPath,
            string recognizerPath,
            string dictionaryPath,
            bool useDirectionClassification)
            : this(detectorPath, classifierPath, recognizerPath, dictionaryPath,
                useDirectionClassification, 0u)
        {
        }

        public NativeOcr(
            string detectorPath,
            string classifierPath,
            string recognizerPath,
            string dictionaryPath,
            bool useDirectionClassification,
            uint workerCount)
        {
            if (workerCount > 16u)
                throw new ArgumentOutOfRangeException("workerCount", "workerCount必须在0到16之间");
            ValidateAbi();
            ValidateFile(detectorPath, "DET模型");
            ValidateFile(recognizerPath, "REC模型");
            ValidateFile(dictionaryPath, "字典");
            if (useDirectionClassification) ValidateFile(classifierPath, "CLS模型");

            IntPtr detector = IntPtr.Zero;
            IntPtr classifier = IntPtr.Zero;
            IntPtr recognizer = IntPtr.Zero;
            IntPtr dictionary = IntPtr.Zero;
            try
            {
                detector = AllocUtf8(detectorPath);
                classifier = useDirectionClassification ? AllocUtf8(classifierPath) : IntPtr.Zero;
                recognizer = AllocUtf8(recognizerPath);
                dictionary = AllocUtf8(dictionaryPath);
                LwOcrOptions options = new LwOcrOptions();
                lw_ocr_options_init(ref options);
                // Explicitly retain the C ABI size markers after changing a
                // nested value. This also protects the .NET 3.5 P/Invoke path
                // from value-type copy semantics when the REC width is set.
                options.StructSize = (uint)Marshal.SizeOf(typeof(LwOcrOptions));
                options.Recognizer.StructSize =
                    (uint)Marshal.SizeOf(typeof(LwRecognizerOptions));
                options.Recognizer.Reserved0 = 0u;
                options.Recognizer.Reserved1 = 0u;
                options.UseDirectionClassification = useDirectionClassification ? 1u : 0u;
                options.Recognizer.TargetWidth = OcrRecognitionDefaults.LongTextTargetWidth;
                if (workerCount != 0u) options.WorkerCount = workerCount;
                LwError error = CreateError();
                int status = lw_ocr_create(detector, classifier, recognizer, dictionary,
                    ref options, out handle, ref error);
                if (status != 0 || handle == IntPtr.Zero)
                    throw NativeFailure("OCR初始化失败", status, error);

                info = new LwOcrInfo();
                lw_ocr_info_init(ref info);
                status = lw_ocr_get_info(handle, ref info);
                if (status != 0 || info.MaxLineCapacity == 0 || info.MaxTextCapacity == 0)
                    throw new InvalidOperationException("无法读取OCR输出容量，错误码=" + status);
                if (info.MaxLineCapacity > int.MaxValue || info.MaxTextCapacity > int.MaxValue)
                    throw new InvalidOperationException("OCR输出容量超过.NET数组限制");
            }
            catch
            {
                if (handle != IntPtr.Zero)
                {
                    lw_ocr_free(handle);
                    handle = IntPtr.Zero;
                }
                throw;
            }
            finally
            {
                Marshal.FreeHGlobal(detector);
                Marshal.FreeHGlobal(classifier);
                Marshal.FreeHGlobal(recognizer);
                Marshal.FreeHGlobal(dictionary);
            }
        }

        public OcrResponse RecognizeEncoded(byte[] encodedImage)
        {
            DecodedBgrImage decoded = DecodeToBgr(encodedImage);
            return RecognizeDecoded(decoded);
        }

        public OcrResponse RecognizeFile(string path)
        {
            if (String.IsNullOrEmpty(path)) throw new ArgumentNullException("path");
            return RecognizeEncoded(File.ReadAllBytes(path));
        }

        // The WinForms benchmark decodes an image once and repeatedly calls the
        // native pipeline. Keeping this helper internal avoids exposing the
        // mutable pixel container as part of the public C# example API.
        internal OcrResponse RecognizeDecoded(DecodedBgrImage image)
        {
            if (image == null || image.Pixels == null || image.Width <= 0 || image.Height <= 0 ||
                image.Stride < checked(image.Width * 3) ||
                image.Pixels.LongLength < checked((long)image.Stride * image.Height))
                throw new ArgumentException("BGR图片数据无效", "image");
            lock (syncRoot)
            {
                ThrowIfDisposed();
                LwOcrLine[] nativeLines = new LwOcrLine[(int)info.MaxLineCapacity];
                byte[] text = new byte[(int)info.MaxTextCapacity];
                GCHandle imagePin = new GCHandle();
                GCHandle linesPin = new GCHandle();
                GCHandle textPin = new GCHandle();
                try
                {
                    imagePin = GCHandle.Alloc(image.Pixels, GCHandleType.Pinned);
                    linesPin = GCHandle.Alloc(nativeLines, GCHandleType.Pinned);
                    textPin = GCHandle.Alloc(text, GCHandleType.Pinned);
                    LwOcrResult nativeResult = new LwOcrResult();
                    lw_ocr_result_init(ref nativeResult);
                    LwError error = CreateError();
                    int status = lw_ocr_run_bgr_u8(
                        handle, imagePin.AddrOfPinnedObject(), (ulong)image.Pixels.LongLength,
                        (uint)image.Width, (uint)image.Height, (uint)image.Stride,
                        linesPin.AddrOfPinnedObject(), info.MaxLineCapacity,
                        textPin.AddrOfPinnedObject(), info.MaxTextCapacity,
                        ref nativeResult, ref error);
                    if (status != 0) throw NativeFailure("OCR识别失败", status, error);
                    if (nativeResult.LineCount > info.MaxLineCapacity ||
                        nativeResult.RequiredTextCapacity > info.MaxTextCapacity)
                        throw new InvalidOperationException("原生OCR返回了无效容量");

                    OcrResponse response = new OcrResponse();
                    response.ok = true;
                    response.api_version = 1;
                    response.image_width = image.Width;
                    response.image_height = image.Height;
                    response.detected_count = (int)nativeResult.DetectedCount;
                    response.result = new List<OcrLine>((int)nativeResult.LineCount);
                    for (int index = 0; index < (int)nativeResult.LineCount; ++index)
                    {
                        LwOcrLine line = nativeLines[index];
                        if (line.TextOffset > nativeResult.RequiredTextCapacity ||
                            line.TextLength > nativeResult.RequiredTextCapacity - line.TextOffset ||
                            line.TextOffset + line.TextLength >= (ulong)text.LongLength)
                            throw new InvalidOperationException("原生OCR返回了无效文本范围");
                        int offset = checked((int)line.TextOffset);
                        int length = checked((int)line.TextLength);
                        if (text[offset + length] != 0)
                            throw new InvalidOperationException("原生OCR文本缺少NUL终止符");
                        response.result.Add(ToManagedLine(line,
                            Encoding.UTF8.GetString(text, offset, length)));
                    }
                    return response;
                }
                finally
                {
                    if (textPin.IsAllocated) textPin.Free();
                    if (linesPin.IsAllocated) linesPin.Free();
                    if (imagePin.IsAllocated) imagePin.Free();
                }
            }
        }

        public uint WorkerCount
        {
            get
            {
                lock (syncRoot)
                {
                    ThrowIfDisposed();
                    return info.WorkerCount;
                }
            }
        }

        internal static DecodedBgrImage DecodeToBgr(byte[] encodedImage)
        {
            if (encodedImage == null || encodedImage.Length == 0)
                throw new ArgumentException("图片数据为空", "encodedImage");
            using (MemoryStream stream = new MemoryStream(encodedImage, false))
            using (Image source = Image.FromStream(stream, true, true))
            {
                long pixels = checked((long)source.Width * source.Height);
                if (source.Width <= 0 || source.Height <= 0 || pixels > MaxDecodedPixels)
                    throw new InvalidOperationException("解码后的图片尺寸无效或超过4000万像素");
                using (Bitmap bitmap = new Bitmap(
                    source.Width, source.Height, PixelFormat.Format24bppRgb))
                {
                    using (Graphics graphics = Graphics.FromImage(bitmap))
                    {
                        graphics.Clear(Color.White);
                        graphics.DrawImage(source, new Rectangle(0, 0, source.Width, source.Height),
                            0, 0, source.Width, source.Height, GraphicsUnit.Pixel);
                    }
                    Rectangle bounds = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
                    BitmapData data = bitmap.LockBits(
                        bounds, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
                    try
                    {
                        int rowBytes = checked(bitmap.Width * 3);
                        byte[] bgr = new byte[checked(rowBytes * bitmap.Height)];
                        int absoluteStride = Math.Abs(data.Stride);
                        for (int y = 0; y < bitmap.Height; ++y)
                        {
                            int sourceRow = data.Stride >= 0 ? y : bitmap.Height - 1 - y;
                            IntPtr row = new IntPtr(data.Scan0.ToInt64() +
                                (long)sourceRow * absoluteStride);
                            Marshal.Copy(row, bgr, y * rowBytes, rowBytes);
                        }
                        DecodedBgrImage decoded = new DecodedBgrImage();
                        decoded.Pixels = bgr;
                        decoded.Width = bitmap.Width;
                        decoded.Height = bitmap.Height;
                        decoded.Stride = rowBytes;
                        return decoded;
                    }
                    finally
                    {
                        bitmap.UnlockBits(data);
                    }
                }
            }
        }

        private static OcrLine ToManagedLine(LwOcrLine source, string text)
        {
            OcrLine line = new OcrLine();
            line.text = text;
            line.score = source.RecognitionScore;
            line.det_score = source.Box.Score;
            line.cls_label = (int)source.ClassificationLabel;
            line.cls_score = source.ClassificationScore;
            line.rotation = (int)source.AppliedRotationDegrees;
            line.x1 = source.Box.X1; line.y1 = source.Box.Y1;
            line.x2 = source.Box.X2; line.y2 = source.Box.Y2;
            line.x3 = source.Box.X3; line.y3 = source.Box.Y3;
            line.x4 = source.Box.X4; line.y4 = source.Box.Y4;
            return line;
        }

        private static void ValidateAbi()
        {
            if (Marshal.SizeOf(typeof(LwError)) != 264 ||
                Marshal.SizeOf(typeof(LwOcrOptions)) != 176 ||
                Marshal.SizeOf(typeof(LwOcrInfo)) != 40 ||
                Marshal.SizeOf(typeof(LwOcrLine)) != 80 ||
                Marshal.SizeOf(typeof(LwOcrResult)) != 40)
                throw new PlatformNotSupportedException("C#与原生OCR ABI结构大小不匹配");
        }

        private static LwError CreateError()
        {
            LwError error = new LwError();
            error.Message = new byte[256];
            lw_error_init(ref error);
            return error;
        }

        private static Exception NativeFailure(string operation, int status, LwError error)
        {
            int length = 0;
            if (error.Message != null)
                while (length < error.Message.Length && error.Message[length] != 0) ++length;
            string message = error.Message == null ? String.Empty :
                Encoding.UTF8.GetString(error.Message, 0, length);
            return new InvalidOperationException(operation + "(" + status + "): " + message);
        }

        private static IntPtr AllocUtf8(string value)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value + "\0");
            IntPtr pointer = Marshal.AllocHGlobal(bytes.Length);
            Marshal.Copy(bytes, 0, pointer, bytes.Length);
            return pointer;
        }

        private static void ValidateFile(string path, string name)
        {
            if (String.IsNullOrEmpty(path) || !File.Exists(path))
                throw new FileNotFoundException(name + "不存在", path);
        }

        private void ThrowIfDisposed()
        {
            if (disposed) throw new ObjectDisposedException("NativeOcr");
        }

        public void Dispose()
        {
            lock (syncRoot)
            {
                if (disposed) return;
                if (handle != IntPtr.Zero)
                {
                    lw_ocr_free(handle);
                    handle = IntPtr.Zero;
                }
                disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~NativeOcr()
        {
            Dispose();
        }
    }
}
