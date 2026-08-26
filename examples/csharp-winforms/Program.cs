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
                if (File.Exists(Path.Combine(direct, "det.lwm"))) return direct;
                if (File.Exists(Path.Combine(directory.FullName, "det.lwm")))
                    return directory.FullName;
                directory = directory.Parent;
            }
            return Path.Combine(Application.StartupPath, "models");
        }
    }
}
