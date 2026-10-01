using System;
using System.IO;
using System.Windows.Forms;

namespace LwPpocrWinForms
{
    internal static class Program
    {
        [STAThread]
        private static void Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            string modelDirectory = args.Length > 0 ? args[0] : FindModelDirectory();
            Application.Run(new MainForm(modelDirectory));
        }

        private static string FindModelDirectory()
        {
            DirectoryInfo directory = new DirectoryInfo(Application.StartupPath);
            while (directory != null)
            {
                string direct = Path.Combine(directory.FullName, "models");
                if (HasDetector(direct)) return direct;
                if (HasDetector(directory.FullName))
                    return directory.FullName;
                directory = directory.Parent;
            }
            return Path.Combine(Application.StartupPath, "models");
        }

        private static bool HasDetector(string directory)
        {
            return File.Exists(Path.Combine(directory, "det.onnx")) ||
                File.Exists(Path.Combine(directory, "det.lwm"));
        }
    }
}
