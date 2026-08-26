using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Text;
using System.Web.Script.Serialization;
using System.Windows.Forms;
using LwPpocrCSharp;

namespace LwPpocrWinForms
{
    internal sealed class MainForm : Form
    {
        private readonly string modelDirectory;
        private readonly Button openButton = new Button();
        private readonly Button initializeButton = new Button();
        private readonly Button recognizeButton = new Button();
        private readonly Button releaseButton = new Button();
        private readonly CheckBox classifierCheck = new CheckBox();
        private readonly PictureBox picture = new PictureBox();
        private readonly RichTextBox output = new RichTextBox();
        private readonly ToolStripStatusLabel statusLabel = new ToolStripStatusLabel();
        private NativeOcr engine;
        private bool engineUsesClassifier;
        private string imagePath;
        private Bitmap originalImage;
        private Bitmap annotatedImage;

        public MainForm(string modelDirectory)
        {
            this.modelDirectory = modelDirectory;
            Text = "lw.PPOCR.C - C# WinForms OCR Demo";
            Width = 1220;
            Height = 780;
            MinimumSize = new Size(850, 560);
            StartPosition = FormStartPosition.CenterScreen;

            FlowLayoutPanel toolbar = new FlowLayoutPanel();
            toolbar.Dock = DockStyle.Top;
            toolbar.Height = 46;
            toolbar.Padding = new Padding(8, 8, 8, 4);
            toolbar.WrapContents = false;
            openButton.Text = "选择图片";
            initializeButton.Text = "初始化模型";
            recognizeButton.Text = "OCR识别";
            releaseButton.Text = "释放模型";
            classifierCheck.Text = "启用方向分类";
            classifierCheck.Checked = true;
            classifierCheck.AutoSize = true;
            classifierCheck.Margin = new Padding(14, 7, 3, 3);
            toolbar.Controls.Add(openButton);
            toolbar.Controls.Add(initializeButton);
            toolbar.Controls.Add(recognizeButton);
            toolbar.Controls.Add(releaseButton);
            toolbar.Controls.Add(classifierCheck);

            SplitContainer split = new SplitContainer();
            split.Dock = DockStyle.Fill;
            split.SplitterDistance = 760;
            picture.Dock = DockStyle.Fill;
            picture.BackColor = Color.FromArgb(34, 40, 49);
            picture.SizeMode = PictureBoxSizeMode.Zoom;
            output.Dock = DockStyle.Fill;
            output.Font = new Font("Consolas", 10F);
            output.ReadOnly = true;
            split.Panel1.Padding = new Padding(8);
            split.Panel2.Padding = new Padding(0, 8, 8, 8);
            split.Panel1.Controls.Add(picture);
            split.Panel2.Controls.Add(output);

            StatusStrip status = new StatusStrip();
            status.Items.Add(statusLabel);
            Controls.Add(split);
            Controls.Add(toolbar);
            Controls.Add(status);

            openButton.Click += OpenImage;
            initializeButton.Click += InitializeEngine;
            recognizeButton.Click += RecognizeImage;
            releaseButton.Click += delegate { ReleaseEngine(); };
            FormClosing += delegate { DisposeRuntimeObjects(); };
            statusLabel.Text = "模型目录: " + modelDirectory;
        }

        private void OpenImage(object sender, EventArgs e)
        {
            using (OpenFileDialog dialog = new OpenFileDialog())
            {
                dialog.Filter = "图片|*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tif;*.tiff|所有文件|*.*";
                if (dialog.ShowDialog(this) != DialogResult.OK) return;
                LoadImage(dialog.FileName);
            }
        }

        private void LoadImage(string path)
        {
            DisposeImages();
            using (FileStream stream = new FileStream(path, FileMode.Open,
                FileAccess.Read, FileShare.ReadWrite))
            using (Image loaded = Image.FromStream(stream))
                originalImage = new Bitmap(loaded);
            annotatedImage = new Bitmap(originalImage);
            picture.Image = annotatedImage;
            imagePath = path;
            output.Clear();
            statusLabel.Text = "已选择: " + path;
        }

        private void InitializeEngine(object sender, EventArgs e)
        {
            try
            {
                SetBusy(true);
                EnsureEngine();
                statusLabel.Text = "模型初始化成功 | " +
                    (engineUsesClassifier ? "CLS已启用" : "CLS已关闭");
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

        private void EnsureEngine()
        {
            if (engine != null && engineUsesClassifier == classifierCheck.Checked) return;
            ReleaseEngine();
            string detector = Path.Combine(modelDirectory, "det.lwm");
            string classifier = Path.Combine(modelDirectory, "cls.lwm");
            string recognizer = Path.Combine(modelDirectory, "rec.lwm");
            string dictionary = Path.Combine(modelDirectory, "ppocr_keys.txt");
            engine = new NativeOcr(detector, classifier, recognizer, dictionary,
                classifierCheck.Checked);
            engineUsesClassifier = classifierCheck.Checked;
        }

        private void RecognizeImage(object sender, EventArgs e)
        {
            if (String.IsNullOrEmpty(imagePath) || !File.Exists(imagePath))
            {
                MessageBox.Show(this, "请先选择图片。", "提示",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }
            try
            {
                SetBusy(true);
                EnsureEngine();
            }
            catch (Exception ex)
            {
                SetBusy(false);
                ShowFailure("模型初始化失败", ex);
                return;
            }

            string selectedPath = imagePath;
            BackgroundWorker worker = new BackgroundWorker();
            worker.DoWork += delegate(object workerSender, DoWorkEventArgs args)
            {
                Stopwatch watch = Stopwatch.StartNew();
                OcrResponse response = engine.RecognizeFile(selectedPath);
                watch.Stop();
                response.timing_ms = new TimingInfo();
                response.timing_ms.server_total = watch.Elapsed.TotalMilliseconds;
                args.Result = response;
            };
            worker.RunWorkerCompleted += delegate(object workerSender, RunWorkerCompletedEventArgs args)
            {
                try
                {
                    if (args.Error != null) throw args.Error;
                    ShowResult((OcrResponse)args.Result);
                }
                catch (Exception ex)
                {
                    ShowFailure("OCR识别失败", ex);
                }
                finally
                {
                    worker.Dispose();
                    SetBusy(false);
                }
            };
            statusLabel.Text = "正在识别...";
            worker.RunWorkerAsync();
        }

        private void ShowResult(OcrResponse response)
        {
            if (annotatedImage != null) annotatedImage.Dispose();
            annotatedImage = new Bitmap(originalImage);
            using (Graphics graphics = Graphics.FromImage(annotatedImage))
            using (Pen pen = new Pen(Color.Red, 2F))
            using (Brush labelBrush = new SolidBrush(Color.FromArgb(230, Color.Red)))
            {
                graphics.SmoothingMode = SmoothingMode.AntiAlias;
                for (int i = 0; i < response.result.Count; ++i)
                {
                    OcrLine line = response.result[i];
                    PointF[] points = new PointF[] {
                        new PointF(line.x1, line.y1), new PointF(line.x2, line.y2),
                        new PointF(line.x3, line.y3), new PointF(line.x4, line.y4) };
                    graphics.DrawPolygon(pen, points);
                    graphics.DrawString((i + 1).ToString(), Font, labelBrush,
                        Math.Max(0F, line.x1), Math.Max(0F, line.y1 - 16F));
                }
            }
            picture.Image = annotatedImage;
            StringBuilder summary = new StringBuilder();
            summary.AppendLine("结果数: " + response.result.Count);
            summary.AppendLine("检测数: " + response.detected_count);
            summary.AppendLine("总耗时: " + response.timing_ms.server_total.ToString("F1") + " ms");
            summary.AppendLine(new String('-', 56));
            for (int i = 0; i < response.result.Count; ++i)
            {
                OcrLine line = response.result[i];
                summary.AppendLine((i + 1) + ". " + line.text +
                    "  [rec=" + line.score.ToString("F3") +
                    ", det=" + line.det_score.ToString("F3") +
                    ", rot=" + line.rotation + "]");
            }
            JavaScriptSerializer serializer = new JavaScriptSerializer();
            serializer.MaxJsonLength = Int32.MaxValue;
            output.Text = summary + Environment.NewLine + "完整JSON" + Environment.NewLine +
                serializer.Serialize(response);
            statusLabel.Text = "识别成功 | " + response.result.Count + " 行 | " +
                response.timing_ms.server_total.ToString("F1") + " ms";
        }

        private void ShowFailure(string title, Exception error)
        {
            output.Text = error.ToString();
            statusLabel.Text = title + ": " + error.Message;
            MessageBox.Show(this, error.Message, title,
                MessageBoxButtons.OK, MessageBoxIcon.Error);
        }

        private void SetBusy(bool busy)
        {
            openButton.Enabled = !busy;
            initializeButton.Enabled = !busy;
            recognizeButton.Enabled = !busy;
            releaseButton.Enabled = !busy;
            classifierCheck.Enabled = !busy;
            UseWaitCursor = busy;
        }

        private void ReleaseEngine()
        {
            if (engine != null)
            {
                engine.Dispose();
                engine = null;
                statusLabel.Text = "模型已释放";
            }
        }

        private void DisposeImages()
        {
            picture.Image = null;
            if (annotatedImage != null) { annotatedImage.Dispose(); annotatedImage = null; }
            if (originalImage != null) { originalImage.Dispose(); originalImage = null; }
        }

        private void DisposeRuntimeObjects()
        {
            ReleaseEngine();
            DisposeImages();
        }
    }
}
