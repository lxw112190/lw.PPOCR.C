# Node.js / WASM 发行包

## 定位

`lw.PPOCR.C` 提供官方 Node.js/WASM 原始运行时发行包。它复用浏览器
使用的同一套纯 C Runtime、LWM 模型和 WASM Host ABI v1，不是 Node.js
SDK、HTTP Server、图片解码器或 Worker Pool 框架。

Node.js 用户下载并解压 Release 中的：

```text
lw.PPOCR.C-<version>-node-wasm.zip
```

即可直接加载 `runtime.cjs` 和随包提供的 PP-OCRv6 tiny 模型。整个运行时
不联网、不下载资源，也不依赖 npm、OpenCV、ONNX Runtime 或 Python。

## 包内容

```text
lw.PPOCR.C-<version>-node-wasm/
├─ runtime.cjs
├─ det.lwm
├─ cls.lwm
├─ rec.lwm
├─ ppocr_keys.txt
├─ manifest.json
├─ SHA256SUMS.txt
├─ README.md
├─ LICENSE
├─ THIRD-PARTY-NOTICES.md
└─ licenses/PaddleOCR-APACHE-2.0.txt
```

模型保持平铺，方便下游应用替换和校验。LWM 文件仍然是 ISA-neutral，
Node/WASM 包不会生成单独的 AVX2、NEON 或 WASM 模型版本。

## 构建

安装 Emscripten 4.0.15、CMake 3.20+、Ninja 和 Python 3.9+ 后执行：

```bash
emcmake cmake -S . -B build-wasm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLW_BUILD_HTTP_DEMO=OFF \
  -DLW_BUILD_CSHARP_DEMOS=OFF \
  -DBUILD_TESTING=OFF

cmake --build build-wasm --target lw-node-wasm
cmake --build build-wasm --target lw-node-wasm-package
```

产物位于 `build-wasm/`：

```text
node-wasm/runtime.cjs
node-wasm/{det,cls,rec}.lwm
node-wasm/ppocr_keys.txt
lw.PPOCR.C-<version>-node-wasm.zip
lw.PPOCR.C-<version>-node-wasm.zip.sha256
```

打包脚本只接受构建好的 Runtime、LWM、字典、许可证和版本参数，禁止
解析 HTML、执行外部 JavaScript 或下载网络资源。

## 最小加载示例

```javascript
const fs = require("node:fs");
const path = require("node:path");
const LwPpocrModule = require("./runtime.cjs");

async function main() {
  const runtime = await LwPpocrModule({});
  runtime.FS.mkdir("/models");
  for (const name of ["det.lwm", "cls.lwm", "rec.lwm", "ppocr_keys.txt"]) {
    runtime.FS.writeFile(`/models/${name}`,
      fs.readFileSync(path.join(__dirname, name)));
  }
  const status = runtime._lw_web_init(1);
  if (status !== 0) throw new Error(`OCR initialization failed: ${status}`);
  console.log("lw.PPOCR.C WASM runtime ready");
  runtime._lw_web_shutdown();
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
```

Node 18、20、22 均属于支持范围。`runtime.cjs` 使用 CommonJS `require()`
加载，导出的函数仍保持现有 `lw_web_*` 名称：

```text
_lw_web_init
_lw_web_shutdown
_lw_web_info_size
_lw_web_get_info
_lw_web_run
_lw_web_last_error
_lw_web_malloc
_lw_web_free
```

这些函数构成 WASM Host ABI v1。函数名暂不改为 `lw_wasm_*`，以保持现有
浏览器 SDK 和第三方集成兼容。ABI 结构大小、字段顺序和偏移均视为兼容
契约；不兼容变更必须升级 ABI 版本。

## 输入和线程模型

Runtime 只接收底层 ABI 约定的 BGR8 像素，不负责 JPEG、PNG、WebP 解码。
Node 应用可以使用自己的解码库，把结果转换为 BGR8 后写入 WASM 内存。

当前 Node/WASM Runtime 是单线程的：一个实例一次只处理一个请求。需要
并发时，由应用创建多个 Node Worker，每个 Worker 独立加载一个 Runtime
实例和 OCR Handle。官方包不提供 Worker Pool，以免替应用决定内存、吞吐
和延迟策略。

## 校验和与许可

`manifest.json` 区分发行包版本、WASM Host ABI 版本和 LWM 版本，并记录
Runtime、模型和字典的 SHA-256。`SHA256SUMS.txt` 校验包内文件，Release
旁边的 `.sha256` 校验完整 ZIP。

Runtime 使用项目许可证；PP-OCRv6 tiny 模型和字典的许可说明位于
`licenses/PaddleOCR-APACHE-2.0.txt`。使用前请同时阅读 `LICENSE` 和
`THIRD-PARTY-NOTICES.md`。

## CI 验收

WASM CI 会构建浏览器 SDK、离线 HTML 和 Node/WASM 包，并执行：

1. `require("runtime.cjs")` 和异步 Runtime 初始化；
2. manifest 与 SHA-256 校验；
3. PPM fixture 的 DET/CLS/REC 完整 OCR；
4. CLS 开关和 init/shutdown 生命周期回归；
5. Node 产物与浏览器构建并列验证，互不依赖 HTML 提取。

浏览器 `ocr-demo.html` 的打包流程保持不变，Node 包不会从 HTML 反向生成。
