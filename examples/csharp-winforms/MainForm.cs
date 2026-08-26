using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Web.Script.Serialization;
using System.Windows.Forms;
using LwPpocrCSharp;

namespace LwPpocrWinForms
{
    internal sealed class MainForm : Form
    {
        private static class NativeMethods
        {
            [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
            internal static extern IntPtr GetModuleHandle(string moduleName);

            [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
            internal static extern uint GetModuleFileName(
                IntPtr module, StringBuilder path, int capacity);
        }

        private sealed class RecognitionTestResult
        {
            public OcrResponse Response;
            public double DecodeMilliseconds;
            public double[] OcrMilliseconds;
            public uint WarmupCount;
            public uint WorkerCount;
            public bool ClassifierEnabled;
        }

        private readonly Button openButton = new Button();
        private readonly Button initializeButton = new Button();
        private readonly Button recognizeButton = new Button();
        private readonly Button benchmarkButton = new Button();
        private readonly Button releaseButton = new Button();
        private readonly Button copyButton = new Button();
        private readonly Button clearHistoryButton = new Button();
        private readonly Button browseModelsButton = new Button();
        private readonly CheckBox classifierCheck = new CheckBox();
        private readonly NumericUpDown workerCountInput = new NumericUpDown();
        private readonly NumericUpDown warmupCountInput = new NumericUpDown();
        private readonly NumericUpDown iterationCountInput = new NumericUpDown();
        private readonly TextBox modelDirectoryInput = new TextBox();
        private readonly PictureBox picture = new PictureBox();
        private readonly RichTextBox output = new RichTextBox();
        private readonly ListView performanceHistory = new ListView();
        private readonly ToolStripStatusLabel statusLabel = new ToolStripStatusLabel();
        private readonly ToolStripStatusLabel runtimeLabel = new ToolStripStatusLabel();
        private NativeOcr engine;
        private bool engineUsesClassifier;
        private uint engineWorkerCount;
        private string engineModelDirectory;
        private string imagePath;
        private Bitmap originalImage;
        private Bitmap annotatedImage;
        private bool busy;

        public MainForm(string modelDirectory)
        {
            Text = "lw.PPOCR.C - C# WinForms OCR测试工具";
            Width = 1280;
            Height = 820;
            MinimumSize = new Size(940, 620);
            StartPosition = FormStartPosition.CenterScreen;
            AllowDrop = true;

            Controls.Add(CreateMainArea());
            Controls.Add(CreateToolbar(modelDirectory));
            Controls.Add(CreateStatusBar());

            openButton.Click += OpenImage;
            initializeButton.Click += InitializeEngine;
            recognizeButton.Click += delegate { StartRecognition(false); };
            benchmarkButton.Click += delegate { StartRecognition(true); };
            releaseButton.Click += delegate { ReleaseEngine(); };
            copyButton.Click += CopyResult;
            clearHistoryButton.Click += delegate { performanceHistory.Items.Clear(); };
            browseModelsButton.Click += BrowseModelDirectory;
            DragEnter += OnImageDragEnter;
            DragDrop += OnImageDragDrop;
            FormClosing += delegate { DisposeRuntimeObjects(); };

            statusLabel.Text = "请选择或拖入图片";
            UpdateRuntimeStatus();
            string samplePath = Path.Combine(modelDirectory, "sample.jpg");
            if (File.Exists(samplePath)) TryLoadImage(samplePath);
        }

        private Control CreateToolbar(string modelDirectory)
        {
            Panel toolbar = new Panel();
            toolbar.Dock = DockStyle.Top;
            toolbar.Height = 88;
            toolbar.Padding = new Padding(8, 6, 8, 4);

            FlowLayoutPanel actions = new FlowLayoutPanel();
            actions.Dock = DockStyle.Top;
            actions.Height = 40;
            actions.WrapContents = false;

            openButton.Text = "选择图片";
            initializeButton.Text = "初始化模型";
            recognizeButton.Text = "单次识别";
            benchmarkButton.Text = "性能测试";
            benchmarkButton.Width = 82;
            releaseButton.Text = "释放模型";
            copyButton.Text = "复制结果";
            clearHistoryButton.Text = "清空记录";

            classifierCheck.Text = "方向分类";
            classifierCheck.Checked = true;
            classifierCheck.AutoSize = true;
            classifierCheck.Margin = new Padding(12, 7, 3, 3);

            ConfigureNumeric(workerCountInput, 1, 16, IntPtr.Size == 8 ? 4 : 1, 42);
            ConfigureNumeric(warmupCountInput, 0, 20, 1, 42);
            ConfigureNumeric(iterationCountInput, 1, 100, 5, 48);

            actions.Controls.Add(openButton);
            actions.Controls.Add(initializeButton);
            actions.Controls.Add(recognizeButton);
            actions.Controls.Add(benchmarkButton);
            actions.Controls.Add(releaseButton);
            actions.Controls.Add(classifierCheck);
            actions.Controls.Add(CreateToolbarLabel("工作器"));
            actions.Controls.Add(workerCountInput);
            actions.Controls.Add(CreateToolbarLabel("预热"));
            actions.Controls.Add(warmupCountInput);
            actions.Controls.Add(CreateToolbarLabel("次数"));
            actions.Controls.Add(iterationCountInput);
            actions.Controls.Add(copyButton);
            actions.Controls.Add(clearHistoryButton);

            TableLayoutPanel modelRow = new TableLayoutPanel();
            modelRow.Dock = DockStyle.Fill;
            modelRow.ColumnCount = 3;
            modelRow.RowCount = 1;
            modelRow.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            modelRow.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
            modelRow.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));

            Label modelLabel = new Label();
            modelLabel.Text = "模型目录";
            modelLabel.AutoSize = true;
            modelLabel.Anchor = AnchorStyles.Left;
            modelLabel.Margin = new Padding(4, 7, 7, 3);
            modelDirectoryInput.Text = modelDirectory;
            modelDirectoryInput.Dock = DockStyle.Fill;
            modelDirectoryInput.Margin = new Padding(0, 3, 7, 3);
            browseModelsButton.Text = "浏览...";
            browseModelsButton.AutoSize = true;

            modelRow.Controls.Add(modelLabel, 0, 0);
            modelRow.Controls.Add(modelDirectoryInput, 1, 0);
            modelRow.Controls.Add(browseModelsButton, 2, 0);
            toolbar.Controls.Add(modelRow);
            toolbar.Controls.Add(actions);
            return toolbar;
        }

        private static void ConfigureNumeric(
            NumericUpDown input, decimal minimum, decimal maximum, decimal value, int width)
        {
            input.Minimum = minimum;
            input.Maximum = maximum;
            input.Value = value;
            input.Width = width;
            input.Margin = new Padding(2, 4, 5, 3);
            input.TextAlign = HorizontalAlignment.Center;
        }

        private static Label CreateToolbarLabel(string text)
        {
            Label label = new Label();
            label.Text = text;
            label.AutoSize = true;
            label.Margin = new Padding(7, 7, 0, 3);
            return label;
        }

        private Control CreateMainArea()
        {
            SplitContainer split = new SplitContainer();
            split.Dock = DockStyle.Fill;
            split.SplitterDistance = 760;

            picture.Dock = DockStyle.Fill;
            picture.BackColor = Color.FromArgb(34, 40, 49);
            picture.SizeMode = PictureBoxSizeMode.Zoom;
            split.Panel1.Padding = new Padding(8);
            split.Panel1.Controls.Add(picture);

            TabControl resultTabs = new TabControl();
            resultTabs.Dock = DockStyle.Fill;
            TabPage resultPage = new TabPage("识别结果");
            TabPage performancePage = new TabPage("性能记录");

            output.Dock = DockStyle.Fill;
            output.Font = new Font("Consolas", 10F);
            output.ReadOnly = true;
            output.WordWrap = false;
            resultPage.Controls.Add(output);

            ConfigurePerformanceHistory();
            performancePage.Controls.Add(performanceHistory);
            resultTabs.TabPages.Add(resultPage);
            resultTabs.TabPages.Add(performancePage);
            split.Panel2.Padding = new Padding(0, 8, 8, 8);
            split.Panel2.Controls.Add(resultTabs);
            return split;
        }

        private void ConfigurePerformanceHistory()
        {
            performanceHistory.Dock = DockStyle.Fill;
            performanceHistory.View = View.Details;
            performanceHistory.FullRowSelect = true;
            performanceHistory.GridLines = true;
            performanceHistory.HideSelection = false;
            performanceHistory.Columns.Add("时间", 66);
            performanceHistory.Columns.Add("架构", 48);
            performanceHistory.Columns.Add("工作器", 55);
            performanceHistory.Columns.Add("CLS", 38);
            performanceHistory.Columns.Add("行数", 44);
            performanceHistory.Columns.Add("预热/次数", 68);
            performanceHistory.Columns.Add("平均ms", 68);
            performanceHistory.Columns.Add("P95ms", 68);
            performanceHistory.Columns.Add("最小ms", 68);
            performanceHistory.Columns.Add("最大ms", 68);
            performanceHistory.Columns.Add("解码ms", 68);
            performanceHistory.Columns.Add("图片", 150);
        }

        private Control CreateStatusBar()
        {
            StatusStrip status = new StatusStrip();
            statusLabel.Spring = true;
            statusLabel.TextAlign = ContentAlignment.MiddleLeft;
            runtimeLabel.TextAlign = ContentAlignment.MiddleRight;
            status.Items.Add(statusLabel);
            status.Items.Add(runtimeLabel);
            return status;
        }

        private void OpenImage(object sender, EventArgs e)
        {
            using (OpenFileDialog dialog = new OpenFileDialog())
            {
                dialog.Filter = "图片|*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tif;*.tiff|所有文件|*.*";
                if (dialog.ShowDialog(this) != DialogResult.OK) return;
                TryLoadImage(dialog.FileName);
            }
        }

        private void OnImageDragEnter(object sender, DragEventArgs e)
        {
            e.Effect = !busy && e.Data.GetDataPresent(DataFormats.FileDrop)
                ? DragDropEffects.Copy
                : DragDropEffects.None;
        }

        private void OnImageDragDrop(object sender, DragEventArgs e)
        {
            if (busy) return;
            string[] paths = e.Data.GetData(DataFormats.FileDrop) as string[];
            if (paths != null && paths.Length > 0) TryLoadImage(paths[0]);
        }

        private void TryLoadImage(string path)
        {
            try
            {
                LoadImage(path);
            }
            catch (Exception ex)
            {
                ShowFailure("图片加载失败", ex);
            }
        }

        private void LoadImage(string path)
        {
            Bitmap nextImage;
            using (FileStream stream = new FileStream(path, FileMode.Open,
                FileAccess.Read, FileShare.ReadWrite))
            using (Image loaded = Image.FromStream(stream, true, true))
                nextImage = new Bitmap(loaded);

            DisposeImages();
            originalImage = nextImage;
            annotatedImage = new Bitmap(originalImage);
            picture.Image = annotatedImage;
            imagePath = path;
            output.Clear();
            statusLabel.Text = "已选择: " + path + " | " + originalImage.Width + "x" +
                originalImage.Height;
        }

        private void BrowseModelDirectory(object sender, EventArgs e)
        {
            using (FolderBrowserDialog dialog = new FolderBrowserDialog())
            {
                dialog.Description = "选择包含 det.lwm、cls.lwm、rec.lwm 和 ppocr_keys.txt 的目录";
                if (Directory.Exists(modelDirectoryInput.Text))
                    dialog.SelectedPath = modelDirectoryInput.Text;
                if (dialog.ShowDialog(this) != DialogResult.OK) return;
                modelDirectoryInput.Text = dialog.SelectedPath;
                ReleaseEngine();
                statusLabel.Text = "模型目录已更改，请重新初始化";
            }
        }

        private void InitializeEngine(object sender, EventArgs e)
        {
            try
            {
                SetBusy(true);
                double milliseconds = EnsureEngine();
                statusLabel.Text = "模型初始化成功 | 工作器=" + engine.WorkerCount +
                    " | CLS=" + (engineUsesClassifier ? "开" : "关") +
                    " | " + milliseconds.ToString("F1") + " ms";
            }
            catch (Exception ex)
            {
                ShowFailure("模型初始化失败", ex);
            }
            finally
            {
                SetBusy(false);
            }
        }

        private double EnsureEngine()
        {
            string selectedDirectory = Path.GetFullPath(modelDirectoryInput.Text.Trim());
            uint selectedWorkers = (uint)workerCountInput.Value;
            bool selectedClassifier = classifierCheck.Checked;
            if (engine != null && engineUsesClassifier == selectedClassifier &&
                engineWorkerCount == selectedWorkers &&
                String.Equals(engineModelDirectory, selectedDirectory,
                    StringComparison.OrdinalIgnoreCase))
                return 0.0;

            ReleaseEngine();
            string detector = Path.Combine(selectedDirectory, "det.lwm");
            string classifier = Path.Combine(selectedDirectory, "cls.lwm");
            string recognizer = Path.Combine(selectedDirectory, "rec.lwm");
            string dictionary = Path.Combine(selectedDirectory, "ppocr_keys.txt");
            Stopwatch watch = Stopwatch.StartNew();
            engine = new NativeOcr(detector, classifier, recognizer, dictionary,
                selectedClassifier, selectedWorkers);
            watch.Stop();
            engineUsesClassifier = selectedClassifier;
            engineWorkerCount = engine.WorkerCount;
            engineModelDirectory = selectedDirectory;
            UpdateRuntimeStatus();
            return watch.Elapsed.TotalMilliseconds;
        }

        private void StartRecognition(bool performanceTest)
        {
            if (String.IsNullOrEmpty(imagePath) || !File.Exists(imagePath))
            {
                MessageBox.Show(this, "请先选择或拖入图片。", "提示",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            double initializationMilliseconds;
            try
            {
                SetBusy(true);
                initializationMilliseconds = EnsureEngine();
            }
            catch (Exception ex)
            {
                SetBusy(false);
                ShowFailure("模型初始化失败", ex);
                return;
            }

            string selectedPath = imagePath;
            uint warmups = performanceTest ? (uint)warmupCountInput.Value : 0u;
            uint iterations = performanceTest ? (uint)iterationCountInput.Value : 1u;
            uint selectedWorkers = engine.WorkerCount;
            bool selectedClassifier = engineUsesClassifier;
            NativeOcr selectedEngine = engine;
            BackgroundWorker worker = new BackgroundWorker();
            worker.DoWork += delegate(object workerSender, DoWorkEventArgs args)
            {
                args.Result = RunRecognitionTest(selectedEngine, selectedPath, warmups, iterations,
                    selectedWorkers, selectedClassifier);
            };
            worker.RunWorkerCompleted += delegate(object workerSender,
                RunWorkerCompletedEventArgs args)
            {
                try
                {
                    if (args.Error != null) throw args.Error;
                    RecognitionTestResult result = (RecognitionTestResult)args.Result;
                    ShowResult(result, initializationMilliseconds, performanceTest);
                }
                catch (Exception ex)
                {
                    ShowFailure(performanceTest ? "性能测试失败" : "OCR识别失败", ex);
                }
                finally
                {
                    worker.Dispose();
                    SetBusy(false);
                }
            };
            statusLabel.Text = performanceTest
                ? "正在性能测试：工作器=" + selectedWorkers + "，预热=" + warmups +
                    "，次数=" + iterations + "..."
                : "正在识别...";
            worker.RunWorkerAsync();
        }

        private static RecognitionTestResult RunRecognitionTest(
            NativeOcr selectedEngine, string selectedPath, uint warmups, uint iterations,
            uint selectedWorkers, bool selectedClassifier)
        {
            Stopwatch watch = Stopwatch.StartNew();
            DecodedBgrImage decoded = NativeOcr.DecodeToBgr(File.ReadAllBytes(selectedPath));
            watch.Stop();

            string expectedSignature = null;
            OcrResponse response = null;
            uint index;
            for (index = 0u; index < warmups; ++index)
            {
                response = selectedEngine.RecognizeDecoded(decoded);
                expectedSignature = VerifyStableResult(expectedSignature, response);
            }

            double[] times = new double[(int)iterations];
            for (index = 0u; index < iterations; ++index)
            {
                Stopwatch runWatch = Stopwatch.StartNew();
                response = selectedEngine.RecognizeDecoded(decoded);
                runWatch.Stop();
                times[(int)index] = runWatch.Elapsed.TotalMilliseconds;
                expectedSignature = VerifyStableResult(expectedSignature, response);
            }
            response.timing_ms = new TimingInfo();
            response.timing_ms.server_total = times[times.Length - 1];

            RecognitionTestResult result = new RecognitionTestResult();
            result.Response = response;
            result.DecodeMilliseconds = watch.Elapsed.TotalMilliseconds;
            result.OcrMilliseconds = times;
            result.WarmupCount = warmups;
            result.WorkerCount = selectedWorkers;
            result.ClassifierEnabled = selectedClassifier;
            return result;
        }

        private static string VerifyStableResult(string expectedSignature, OcrResponse response)
        {
            string actual = BuildResultSignature(response);
            if (expectedSignature != null && !String.Equals(expectedSignature, actual,
                StringComparison.Ordinal))
                throw new InvalidOperationException("重复识别结果不一致，性能数据已作废");
            return actual;
        }

        private static string BuildResultSignature(OcrResponse response)
        {
            StringBuilder signature = new StringBuilder();
            signature.Append(response.detected_count).Append('|').Append(response.result.Count);
            for (int index = 0; index < response.result.Count; ++index)
            {
                OcrLine line = response.result[index];
                signature.Append('|').Append(line.text).Append('|')
                    .Append(line.x1.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(line.y1.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(line.x2.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(line.y2.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(line.x3.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(line.y3.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(line.x4.ToString("R", CultureInfo.InvariantCulture)).Append(',')
                    .Append(line.y4.ToString("R", CultureInfo.InvariantCulture));
            }
            return signature.ToString();
        }

        private void ShowResult(
            RecognitionTestResult test, double initializationMilliseconds, bool performanceTest)
        {
            OcrResponse response = test.Response;
            DrawResult(response);

            double minimum;
            double maximum;
            double mean;
            double p95;
            CalculateStatistics(test.OcrMilliseconds, out minimum, out maximum, out mean, out p95);
            StringBuilder summary = new StringBuilder();
            summary.AppendLine("图片: " + imagePath);
            summary.AppendLine("尺寸: " + response.image_width + " x " + response.image_height);
            summary.AppendLine("工作器: " + test.WorkerCount +
                "    方向分类: " + (test.ClassifierEnabled ? "开启" : "关闭"));
            if (initializationMilliseconds > 0.0)
                summary.AppendLine("模型初始化: " + initializationMilliseconds.ToString("F1") +
                    " ms");
            summary.AppendLine("图片解码: " + test.DecodeMilliseconds.ToString("F1") + " ms");
            summary.AppendLine("OCR次数: " + test.OcrMilliseconds.Length +
                "    预热次数: " + test.WarmupCount);
            summary.AppendLine("OCR耗时: 平均 " + mean.ToString("F1") +
                " ms | P95 " + p95.ToString("F1") +
                " ms | 最小 " + minimum.ToString("F1") +
                " ms | 最大 " + maximum.ToString("F1") + " ms");
            summary.AppendLine("结果数: " + response.result.Count +
                "    检测数: " + response.detected_count);
            summary.AppendLine(new String('-', 64));
            for (int index = 0; index < response.result.Count; ++index)
            {
                OcrLine line = response.result[index];
                summary.AppendLine((index + 1) + ". " + line.text +
                    "  [rec=" + line.score.ToString("F3") +
                    ", det=" + line.det_score.ToString("F3") +
                    ", cls=" + line.cls_score.ToString("F3") +
                    ", rot=" + line.rotation + "]");
            }
            JavaScriptSerializer serializer = new JavaScriptSerializer();
            serializer.MaxJsonLength = Int32.MaxValue;
            output.Text = summary + Environment.NewLine + "完整JSON" + Environment.NewLine +
                serializer.Serialize(response);

            if (performanceTest) AddPerformanceRecord(test, minimum, maximum, mean, p95);
            statusLabel.Text = (performanceTest ? "性能测试完成" : "识别成功") +
                " | 工作器=" + test.WorkerCount + " | " + response.result.Count +
                " 行 | 平均 " + mean.ToString("F1") + " ms";
        }

        private void DrawResult(OcrResponse response)
        {
            if (annotatedImage != null) annotatedImage.Dispose();
            annotatedImage = new Bitmap(originalImage);
            using (Graphics graphics = Graphics.FromImage(annotatedImage))
            using (Pen pen = new Pen(Color.Red, 2F))
            using (Brush labelBrush = new SolidBrush(Color.FromArgb(230, Color.Red)))
            {
                graphics.SmoothingMode = SmoothingMode.AntiAlias;
                for (int index = 0; index < response.result.Count; ++index)
                {
                    OcrLine line = response.result[index];
                    PointF[] points = new PointF[] {
                        new PointF(line.x1, line.y1), new PointF(line.x2, line.y2),
                        new PointF(line.x3, line.y3), new PointF(line.x4, line.y4) };
                    graphics.DrawPolygon(pen, points);
                    graphics.DrawString((index + 1).ToString(), Font, labelBrush,
                        Math.Max(0F, line.x1), Math.Max(0F, line.y1 - 16F));
                }
            }
            picture.Image = annotatedImage;
        }

        private static void CalculateStatistics(double[] values, out double minimum,
            out double maximum, out double mean, out double p95)
        {
            double sum = 0.0;
            minimum = Double.MaxValue;
            maximum = Double.MinValue;
            for (int index = 0; index < values.Length; ++index)
            {
                double value = values[index];
                sum += value;
                if (value < minimum) minimum = value;
                if (value > maximum) maximum = value;
            }
            mean = sum / values.Length;
            double[] sorted = (double[])values.Clone();
            Array.Sort(sorted);
            int p95Index = (int)Math.Ceiling(sorted.Length * 0.95) - 1;
            p95 = sorted[Math.Max(0, p95Index)];
        }

        private void AddPerformanceRecord(RecognitionTestResult test, double minimum,
            double maximum, double mean, double p95)
        {
            string[] values = new string[] {
                DateTime.Now.ToString("HH:mm:ss"),
                IntPtr.Size == 8 ? "x64" : "x86",
                test.WorkerCount.ToString(),
                test.ClassifierEnabled ? "开" : "关",
                test.Response.result.Count.ToString(),
                test.WarmupCount + "/" + test.OcrMilliseconds.Length,
                mean.ToString("F1"),
                p95.ToString("F1"),
                minimum.ToString("F1"),
                maximum.ToString("F1"),
                test.DecodeMilliseconds.ToString("F1"),
                Path.GetFileName(imagePath)
            };
            performanceHistory.Items.Insert(0, new ListViewItem(values));
        }

        private void CopyResult(object sender, EventArgs e)
        {
            if (String.IsNullOrEmpty(output.Text)) return;
            try
            {
                Clipboard.SetText(output.Text);
                statusLabel.Text = "识别结果已复制到剪贴板";
            }
            catch (Exception ex)
            {
                ShowFailure("复制失败", ex);
            }
        }

        private void ShowFailure(string title, Exception error)
        {
            output.Text = error.ToString();
            statusLabel.Text = title + ": " + error.Message;
            MessageBox.Show(this, error.Message, title,
                MessageBoxButtons.OK, MessageBoxIcon.Error);
        }

        private void SetBusy(bool value)
        {
            busy = value;
            openButton.Enabled = !value;
            initializeButton.Enabled = !value;
            recognizeButton.Enabled = !value;
            benchmarkButton.Enabled = !value;
            releaseButton.Enabled = !value;
            classifierCheck.Enabled = !value;
            workerCountInput.Enabled = !value;
            warmupCountInput.Enabled = !value;
            iterationCountInput.Enabled = !value;
            modelDirectoryInput.Enabled = !value;
            browseModelsButton.Enabled = !value;
            UseWaitCursor = value;
        }

        private void ReleaseEngine()
        {
            if (engine != null)
            {
                engine.Dispose();
                engine = null;
            }
            engineWorkerCount = 0u;
            engineModelDirectory = null;
            UpdateRuntimeStatus();
        }

        private void UpdateRuntimeStatus()
        {
            string expectedPath = Path.Combine(Application.StartupPath, "lw_ppocr_c.dll");
            string dllState = File.Exists(expectedPath) ? expectedPath : "等待加载: " + expectedPath;
            IntPtr module = NativeMethods.GetModuleHandle("lw_ppocr_c.dll");
            if (module != IntPtr.Zero)
            {
                StringBuilder loadedPath = new StringBuilder(1024);
                if (NativeMethods.GetModuleFileName(module, loadedPath, loadedPath.Capacity) > 0u)
                    dllState = loadedPath.ToString();
            }
            runtimeLabel.Text = (IntPtr.Size == 8 ? "x64" : "x86") + " | DLL: " + dllState;
        }

        private void DisposeImages()
        {
            picture.Image = null;
            if (annotatedImage != null)
            {
                annotatedImage.Dispose();
                annotatedImage = null;
            }
            if (originalImage != null)
            {
                originalImage.Dispose();
                originalImage = null;
            }
        }

        private void DisposeRuntimeObjects()
        {
            ReleaseEngine();
            DisposeImages();
        }
    }
}
