# lw.PPOCR.C

[English](README.md) | 简体中文

`lw.PPOCR.C` 是一个面向 PP-OCR 的轻量级纯 C 推理运行时。部署端不依赖
Python、OpenCV、ONNX Runtime、OpenVINO、TensorRT 或 protobuf，适合将文字识别能力
集成到桌面软件、嵌入式程序、本地服务和其他对依赖体积敏感的场景。

> 本项目不是通用 ONNX 推理框架。当前目标是可靠、高效地运行已经转换为 LWM 格式的
> PP-OCRv6 tiny 模型。

## 主要功能

- 完整 OCR：文字检测（DET）→ 可选方向分类（CLS）→ 文字识别（REC）；
- 独立的 DET、CLS、REC 公共 C API；
- 纯 C11 核心，公共 ABI 不暴露 C++、STL 或第三方库类型；
- 支持标量、SSE2 和 AVX2 运行时自动分派，不支持 SIMD 时自动回退；
- 完整 OCR 在 DET 后使用独立 CLS/REC worker 并行处理文字行；原生 64 位默认 4 个，
  x86 与 WebAssembly 默认 1 个，可通过 `lw_ocr_options.worker_count` 调整；
- 输入为调用方已经解码的 BGR8 图像，核心库不绑定具体图片解码库；
- 提供 C 命令行示例、C# WinForms Demo、原生 HTTP/Web Demo 和单文件离线 WASM Demo；
- 自定义 LWM v0.1 模型格式，加载时执行边界、结构和校验和检查；
- 调用方拥有输入和输出缓冲区，内存容量不足时返回明确错误，不在 ABI 两侧交叉释放内存。

## 处理流程

```text
PP-OCR ONNX 模型
        │
        ▼
开发期 ModelC 转换器
        │
        ▼
平台无关的 .lwm 模型
        │
        ▼
纯 C 推理运行时
        │
        ├── DET 文字检测
        ├── CLS 方向分类（可选）
        └── REC 文字识别
```

完整 OCR 接口接收一张 BGR8 图片，返回按阅读顺序排列的文字行。每行包含四点坐标、
检测分数、识别分数、方向分类结果以及 UTF-8 文本在调用方文本缓冲区中的偏移和长度。

## 当前支持范围

- 模型：PP-OCRv6 tiny；
- 精度与设备：FP32、CPU；
- 指令集：标量、SSE2、AVX2；
- 线程模型：单个 OCR 句柄仍由调用方串行使用，但句柄内部可并行处理不同文字行；
  多个句柄也可以由应用自行并行调度；
- 首要平台：Windows x64、Linux x64；
- 兼容目标：Windows 7 x86；
- 模型格式：LWM v0.1，目前尚未冻结为稳定格式。

平台支持分为源码兼容、CI 验证和实体机验证三个层次。不要仅凭某个平台能够编译，便认为
所有发行版和硬件都已经得到验证。具体说明请查看
[`docs/package.md`](docs/package.md) 和 [`docs/architecture.md`](docs/architecture.md)。

## 目录说明

```text
include/                     公共 C API
src/runtime/                 模型加载、校验、Shape/内存规划和图执行器
src/kernels/                 可移植的标量算子
src/simd/                    SSE2/AVX2 优化及 CPU 特性检测
src/ppocr/                   DET、CLS、REC 和完整 OCR 流程
converter/                   ONNX 到 LWM 的开发期转换工具
examples/                    C、C# WinForms、HTTP/Web 示例
models/                      示例模型、字典和测试图片
tests/                       ABI、算子、流水线、真实模型及安装包测试
docs/                        设计、API、模型和性能文档
```

## 编译环境

- CMake 3.20 或更高版本；
- 支持 C11 的编译器；
- 构建默认启用的 HTTP Demo 时，需要支持 C++11 的编译器；
- Python 3.9 或更高版本，用于模型转换和部分自动化测试；
- 转换工具依赖见 `requirements-converter.txt`。

Windows 推荐使用 Visual Studio 2022 的 MSVC 工具链；也可以在 VS Code 中配合 CMake 和
Ninja 使用。Linux 推荐 GCC 或 Clang。

## 编译与测试

### Windows（Visual Studio 生成器）

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

### Windows 或 Linux（Ninja）

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Windows 上使用 MSVC + Ninja 时，请先打开“x64 Native Tools Command Prompt for VS 2022”，
或者先执行 Visual Studio 的开发环境初始化脚本。仅把 `cl.exe` 加入 `PATH` 不够，因为
编译器还需要标准库头文件、Windows SDK 头文件和链接库路径。

默认构建会生成：

- 静态和动态纯 C 库；
- `lw-recognize-ppm`、`lw-classify-ppm`、`lw-detect-ppm`、`lw-ocr-ppm` 示例；
- `lw-rec-benchmark` 与 `lw-ocr-benchmark` 基准程序；
- `lwm-inspect` 模型检查工具；
- `lw.PPOCR.C.HttpServer` 原生 HTTP 服务及 Web 测试页。

## 快速体验

### C 命令行完整 OCR

核心库接收 BGR8 像素。为了保持示例简单且不引入图片库，C 命令行 Demo 使用 P6 PPM：

```powershell
.\build\Release\lw-ocr-ppm.exe `
  .\build\models\det.lwm `
  .\build\models\cls.lwm `
  .\build\models\rec.lwm `
  .\build\models\ppocr_keys.txt `
  .\build\models\sample.ppm
```

使用 Ninja 时，程序通常位于 `build/bin/` 或 CMake 输出中显示的位置，不需要
`Release` 这一层目录。

### HTTP OCR 与 Web 页面

HTTP Demo 使用原生 C++ 和 vendored `cpp-httplib`，没有 .NET 运行时依赖：

```powershell
.\build\bin\lw.PPOCR.C.HttpServer.exe --host 127.0.0.1 --port 8787
```

浏览器打开 `http://127.0.0.1:8787/`，选择常见格式图片即可测试。浏览器通过 Canvas
完成图片解码，再把像素转换成 P6 PPM 上传；服务端把 RGB 转成 BGR 后调用纯 C OCR API。

接口同时接受：

- 二进制 P6 PPM 请求体；
- JSON 中 Base64 编码的 P6 PPM。

详细接口、请求示例和安全边界请看
[`docs/managed-demos.md`](docs/managed-demos.md)。该 Demo 默认用于本机或可信网络；若对公网
开放，应在前面部署带有 HTTPS、身份认证、限流和请求大小控制的反向代理。

### 单文件离线 WASM Demo

安装并激活 Emscripten SDK 后（确保 `emcmake` 已在 `PATH` 中），可以用 Ninja 生成一个自包含的离线页面：

```powershell
emcmake cmake -S . -B build-wasm -G Ninja -DCMAKE_BUILD_TYPE=Release -DLW_BUILD_HTTP_DEMO=OFF -DBUILD_TESTING=OFF
cmake --build build-wasm --target lw-ocr-html
```

Windows 下可在 emsdk 目录运行 `emsdk_env.bat`，或按 emsdk 文档使用对应的环境初始化脚本；不同安装位置无需修改上述构建命令。

生成的 `build-wasm/ocr-demo.html` 已内嵌 WASM、三个 LWM 模型和字典，可以直接双击打开
并选择图片进行完整 OCR，不需要启动 HTTP 服务。页面在初始化时按照 Runtime 返回的真实
容量分配输出缓冲区，后续识别会复用这些缓冲区；输入缓冲区只在图片变大时扩容。

### C# WinForms Demo

WinForms Demo 面向 .NET Framework 3.5，通过 `DllImport` 直接调用公共 C ABI，并使用
`System.Drawing` 解码常见图片格式：

```powershell
cmake -S . -B build -DLW_BUILD_CSHARP_DEMOS=ON
cmake --build build --config Release --target lw_csharp_demos
```

项目文件位于 [`examples/csharp-winforms`](examples/csharp-winforms)。运行时请确保 EXE、
对应架构的 `lw_ppocr_c.dll` 和 `models` 目录来自同一次构建或同一个安装包。

## C API 使用要点

公共头文件为 [`include/lw_infer.h`](include/lw_infer.h)。推荐按以下顺序调用：

1. 调用对应的 `*_options_init` 初始化选项结构；
2. 使用 UTF-8 模型路径创建 DET、CLS、REC 或完整 OCR 句柄；
3. 首先使用空输出缓冲区查询需要的容量，或者按照 `*_get_info` 返回的上限分配；
4. 传入已解码的 BGR8 像素和调用方拥有的输出缓冲区；
5. 检查 `lw_status` 和 `lw_error`，不要依赖自然语言错误消息编写业务逻辑；
6. 使用对应的 `*_free` 释放句柄。

所有公开结构都带有 `struct_size`。调用初始化函数可以正确填写该字段并清零保留字段，
这也是以后扩展 ABI 时识别结构版本的重要基础。除非文档明确说明，同一个句柄不要在多个
线程中并发调用。

完整的字段、所有权和容量查询规则见 [`docs/c-api.md`](docs/c-api.md) 与
[`docs/full-ocr.md`](docs/full-ocr.md)。

## 模型转换

转换器属于开发工具，可以依赖 Python、ONNX、NumPy 和 protobuf；这些依赖不会进入部署端
纯 C 库。

```powershell
python -m pip install -r requirements-converter.txt
python converter/analyze_onnx.py `
  --json-output docs/ppocrv6-tiny-analysis.json `
  --markdown-output docs/SUPPORTED_OPS_V0.md
python -m unittest -v tests.test_analyze_onnx
```

模型、字典和第三方组件版本记录在 `dependencies.lock.json`，软件物料清单位于
`sbom.cdx.json`。请不要绕过转换器和加载器的格式、尺寸或校验和检查。

`opencv-python-headless` 只在完整 OCR 的透视裁剪测试中充当独立参考实现；纯 C 核心、
HTTP Demo 和正式发布包均不链接、加载或携带 OpenCV。

## 进一步阅读

- [项目设计](docs/PROJECT_DESIGN.md)
- [架构和兼容性边界](docs/architecture.md)
- [公共 C API](docs/c-api.md)
- [完整 OCR 流程](docs/full-ocr.md)
- [DET 流程](docs/det-pipeline.md)
- [CLS 流程](docs/cls-pipeline.md)
- [REC 流程](docs/rec-pipeline.md)
- [图执行器](docs/graph-executor.md)
- [标量算子](docs/scalar-kernels.md)
- [性能基线与优化](docs/performance-baseline.md)
- [开发包说明](docs/package.md)
- [C# 与 HTTP/Web Demo](docs/managed-demos.md)

## 许可证

项目使用 MIT License。第三方组件的版权与许可证说明见
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md) 和 [`licenses`](licenses)。

## 联系与支持

- 作者：天天代码码天天
- QQ：819069052
- QQ Group: C# 人工智能实践 | 群号: 758616458
- 项目地址：<https://github.com/lxw112190/lw.PPOCR.C>

如果项目对你有帮助，可以扫码支持维护：

<img src="docs/assets/sponsor.jpg" alt="捐赠二维码" width="240">
