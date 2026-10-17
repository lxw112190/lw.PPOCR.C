# lw.PPOCR.C WinForms 便携测试包 / Portable test Demo

这是 Windows x64 便携 Demo；`BUILD-INFO.txt` 区分本地测试构建和 CI 发布候选。
正式版本请从项目对应 tag 的 GitHub Release 下载，不要将本地测试包误认为正式产物。
支持 Windows 10/11 x64，系统需已安装 .NET Framework 4.8 或更新的 4.x。
不需要 Visual Studio、.NET SDK、Python、OpenCV、ONNX Runtime、VC++ Redistributable
或网络模型下载。原生 DLL 静态链接 VC 运行库；Windows 自带系统 DLL 不另行打包。
This portable Demo needs Windows 10/11 x64 and .NET Framework 4.8+, not a development SDK.

## 使用步骤

1. **完整解压 ZIP** 到可读写目录，不要直接在压缩软件里运行 EXE。
2. 双击根目录 `lw.PPOCR.C.WinForms.exe`，默认模型是 Tiny，自动打开内置样例。
   也可双击 `start-tiny.cmd`。
3. 选择或拖入 JPG/PNG/BMP 等图片，点击识别，查看检测框和识别文字。
4. 可切换工作器数量（例如 1 和 4）、方向分类，执行预热/重复性能测试。
   大图请先把“原图上限(百万像素)”从40调到100等适当值，再选择图片。x64可配置1到200，
   100代表1亿像素；单个文字框裁剪仍上限1600万。调高会增加内存需求，不会自动分块或缩图。
5. 支持复制全文、导出 TXT 和 JSON。此 WinForms 包只测试图片，不是 PDF 浏览器 Demo。
6. `start-small.cmd` / `start-medium.cmd` 用于测试另外两种模型，也可在界面中选择
   `models/small` 或 `models/medium`。Small/Medium 仍是 Preview，耗时和内存更高。

Double-click the EXE after extraction. Tiny is default; use the Small/Medium launchers
for the larger Preview models. Select/drop an image, recognize, then copy/export results.
All image processing is local CPU inference; images are not uploaded.

## 包内内容

- `lw.PPOCR.C.WinForms.exe` / `.exe.config`：.NET Framework 4.8 x64 窗口程序。
- `lw_ppocr_c.dll`：同一次构建的原生 C Runtime，REC 最大宽度 960。
- `models/`：Tiny 官方 ONNX、字典和样例；`small/`、`medium/` 包含对应配套模型和字典。
  三种模型使用同一个 CLS；不要混用 Tiny 与 Small/Medium 字典。
- `source/`：当前 C# Demo 与 P/Invoke 源码，包含无需 NuGet 的 .NET 4.8 项目
  `LwPpocrDemo.csproj`，方便二次开发；运行不需要编译。
- `MODEL-CATALOG.json`：官方模型来源、许可与 SHA-256。
- `BUILD-INFO.txt`：源码 commit、构建类型和运行要求。
- `LICENSE`、`THIRD-PARTY-NOTICES.md`、`licenses/`、`sbom.cdx.json`：许可及组件信息。
- `SHA256SUMS.txt`：包内校验；ZIP 同名 `.sha256` 是整包校验文件。

## 常见问题

- 启动提示缺 .NET：安装微软 .NET Framework 4.8 **Runtime**，不需要 SDK，亦不必启用 .NET 3.5。
  本包不捆绑系统 .NET；精简 Windows 可能未预装。
- `BadImageFormatException`：本包为 x64，不能运行在 32 位 Windows；不要替换成 x86 DLL。
- DLL/模型找不到：确保整包已解压，EXE 与 DLL 在同一目录，并保留 models 的完整层级。
- 文件下载被系统拦截：先核实来源和哈希，再按组织安全策略处理；不要关闭杀毒软件。
- 不同机器性能不同：记录 CPU、模型、CLS、worker、预热与次数再比较。
  性能测试不含模型初始化；首次识别和大模型初始化可能较慢。
- 界面内存提示只计算BGR；1亿像素仅BGR约286 MiB，还需要解码、预览、模型和工作区。
  多请求场景须限制大图并发。超限报错会显示实际宽高、像素数和当前上限。
- 不要混用旧发布包的 DLL、新 EXE 或其他来源的字典。

## 开发者重建

调用方可通过新增重载设置每个引擎的原图上限，旧构造保持4000万默认：

```csharp
using (var ocr = new NativeOcr(detPath, clsPath, recPath, dictPath,
    true, 4u, 100000000UL))
{
    var result = ocr.RecognizeFile(imagePath);
}
```

此参数同步 C# 解码与原生 DET，不改变 CLS/REC 的单裁剪保护。0不表示无限。

仓库原 WinForms 项目仍保留 .NET Framework 3.5；本便携包单独编译为 4.8，不改原兼容目标。
使用 VS 2022 C++/C# 工具与 .NET 4.8 targeting pack：

```powershell
cmake -S . -B build/winforms-portable -G "Visual Studio 17 2022" -A x64 `
  -DBUILD_TESTING=OFF -DLW_BUILD_HTTP_DEMO=OFF -DLW_BUILD_CSHARP_DEMOS=OFF `
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
cmake --build build/winforms-portable --config Release --target lw_ppocr_c_shared
python tools/package_winforms_demo.py --build-dir build/winforms-portable `
  --csc "C:/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/Roslyn/csc.exe" `
  --framework-dir "C:/Program Files (x86)/Reference Assemblies/Microsoft/Framework/.NETFramework/v4.8" `
  --output-dir build/winforms-distribution
```

输出目录必须是新的目录；打包器不会覆盖已存在的测试包。
