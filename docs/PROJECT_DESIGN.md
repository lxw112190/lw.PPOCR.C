# lw.PPOCR.C 项目设计与开发方案

> **项目定位**：面向 PP-OCR 的轻量纯 C 推理 Runtime。
> **目标平台**：Windows 7 SP1 x86/x64 至现代 Windows，以及 Linux x64 / ARM64。
> **核心原则**：Pure C、PP-OCR Focused、Zero Heavy Runtime Dependency、Small / Fast / Portable / Auditable。
> **当前开发重点**：PP-OCRv6 tiny REC + FP32 + CPU + 自定义 `.lwm` 模型格式。

---

# 1. 项目名称

项目名称：

```text
lw.PPOCR.C
```

英文副标题：

```text
Tiny pure-C inference runtime for PP-OCR
```

中文定位：

> `lw.PPOCR.C` 是一个面向 PP-OCR 的轻量纯 C 推理 Runtime，目标支持 Windows 7 SP1 x86/x64 至现代 Windows，以及 Linux x64 / ARM64。

建议 GitHub 仓库：

```text
https://github.com/lxw112190/lw.PPOCR.C
```

---

# 2. 项目背景

现有项目：

```text
lw.PPOCR.Inference
```

已经提供统一 C ABI，并支持：

```text
OpenCV DNN
ONNX Runtime / DirectML
OpenVINO
TensorRT
```

这些 Runtime 均依赖第三方推理框架。

新项目 `lw.PPOCR.C` 不重复现有项目，而是实现：

> 一个只面向 PP-OCR 模型、部署端完全由纯 C 实现的专用推理 Runtime。

核心价值不是成为新的 ONNX Runtime，而是：

```text
小
纯 C
零大型运行时依赖
可审计
易移植
易部署
兼容 Win7 x86
适合工业现场
针对 PP-OCR 专门优化
```

未来成熟后，可作为 `lw.PPOCR.Inference` 的第六个 Runtime：

```text
lw.PPOCR.Inference
│
├─ OpenCV Runtime
├─ DirectML Runtime
├─ ONNX Runtime
├─ OpenVINO Runtime
├─ TensorRT Runtime
└─ C Runtime
      ↓
   lw.PPOCR.C
```

原则上不要求修改现有 `lw.PPOCR.Inference` 已冻结的公共 C ABI。

---

# 3. 最终平台策略

## 3.1 支持平台

| 平台 | 支持级别 | SIMD / 优化策略 |
|---|---|---|
| Windows 7 SP1 x86 | Compatibility | Scalar + SSE2 |
| Windows 7 SP1 x64 | 正式支持 | SSE2 + AVX2 |
| Windows 10/11 x64 | Primary | SSE2 + AVX2 |
| Linux x64 | Primary | SSE2 + AVX2 |
| Linux ARM64 | Primary | NEON |

核心原则：

> **一套源码、一种 `.lwm` 模型格式、多个部署目标。**

禁止维护：

```text
win32 runtime 分支
win64 runtime 分支
arm64 runtime 分支
```

所有平台共享：

```text
src/runtime/
src/kernels/
src/ppocr/
```

仅：

```text
src/platform/
src/simd/
```

做平台差异。

---

# 4. Windows 7 x86 的定位

Win7 x86 需要支持，但它不是整个架构的中心。

定义：

```text
Windows 7 x86 Compatibility Profile
```

目标是：

> 保证 PP-OCR 推理可以稳定运行、结果正确、内存可控。

不承诺：

```text
与 x64 完全相同的吞吐
与 x64 完全相同的 batch
与 AVX2 相同的性能
支持超大模型
```

建议 Win7 x86 默认策略：

```text
Precision: FP32
SIMD: Scalar / SSE2
Batch: 1～4
Threads: 1～2
Workspace: 受 32 位地址空间限制
GPU: 不支持
```

Win7 x86 不作为 v0.1 阻塞项，可在核心 Runtime 稳定后正式纳入验证。

---

# 5. 核心目标

最终数据链路：

```text
PP-OCR ONNX
      ↓
离线 Model Converter
      ↓
自定义 .lwm 模型
      ↓
Pure C Runtime
      ↓
PP-OCR
```

部署端不依赖：

```text
Python
OpenCV
ONNX Runtime
OpenVINO
TensorRT
protobuf
Paddle Runtime
C++ STL
```

第一阶段：

```text
CPU
FP32
single-thread
Pure C
PP-OCRv6 tiny REC
```

---

# 6. 明确不做什么

这是项目最重要的边界。

## 6.1 不做通用 ONNX Runtime

不承诺：

```text
任意 ONNX 模型
任意 opset
任意算子
Loop
If
Sequence
Map
训练
自动求导
复杂控制流
任意动态 Shape
```

只支持：

> 经过正式验证的 PP-OCR 模型。

Converter 遇到不支持模型时应明确报错：

```text
Unsupported operator: XXX
```

不能因为第三方模型出现新算子，就无条件扩展 Runtime。

---

## 6.2 第一阶段不做 GPU

暂不做：

```text
CUDA
OpenCL
DirectML
Vulkan
Metal
NPU
```

---

## 6.3 第一阶段不做 INT8

统一：

```text
FP32
```

后续在稳定后再考虑：

```text
FP16
INT8
```

---

# 7. 为什么部署 Runtime 不直接解析 ONNX

本项目采用：

```text
开发环境

PP-OCR.onnx
      ↓
lw.PPOCR.ModelC
      ↓
*.lwm


部署环境

*.lwm
 ↓
lw.PPOCR.C
 ↓
Inference
```

理由：

ONNX 是模型交换格式，而不是本项目最适合的执行格式。

如果 Runtime 直接读取 ONNX，需要在部署端承担：

```text
protobuf parser
ModelProto
GraphProto
NodeProto
TensorProto
AttributeProto
opset compatibility
shape inference
constant folding
graph optimization
weight repacking
memory planning
```

这些工作都可以在开发机离线完成。

因此本项目采用类似“编译器 + 执行器”的思想：

```text
ONNX
 ↓
Model Compiler
 ↓
LWM
 ↓
Pure-C Executor
```

这样部署 Runtime 可以更小、更简单、更稳定。

---

# 8. Converter 不要求纯 C

重要原则：

> Runtime 必须纯 C，Converter 不需要纯 C。

Converter 推荐使用：

```text
Python
onnx
numpy
```

原因：

- Converter 只在开发环境运行；
- 可以方便解析 ONNX；
- 可以完成 Shape 推导、Constant Folding、图优化；
- 可以生成平台无关 `.lwm`。

部署端完全不需要 Python。

---

# 9. Converter 职责

`lw.PPOCR.ModelC` 最终负责：

```text
读取 ONNX
↓
模型验证
↓
统计 Operator
↓
读取 Tensor / Initializer
↓
Shape 推导
↓
Constant Folding
↓
删除无意义 Shape 节点
↓
低风险 Graph Simplification
↓
Tensor 生命周期分析
↓
Workspace 内存规划
↓
生成 .lwm
```

后期可增加：

```text
Conv + BN Fusion
Activation Fusion
Weight Packing
Transpose Elimination
```

第一阶段不要求复杂优化。

---

# 10. 第一件开发任务：模型分析

**新会话接手后，不要先写 Runtime。**

首先分析现有：

```text
PP-OCRv6 tiny

det.onnx
cls.onnx
rec.onnx
```

优先复用现有 `lw.PPOCR.Inference` 中的模型和测试资源。

创建：

```text
converter/analyze_onnx.py
```

输出：

```text
Model
Opset
Input
Output
Operator frequency
Operator attributes
Initializers
Initializer dtypes
Dynamic shapes
Constant nodes
Shape-only nodes
```

例如：

```text
Model: rec.onnx

Input:
float32 [1, 3, 48, ?]

Output:
float32 [...]

Operators:
Conv            47
Relu            18
Add             16
Mul              9
Reshape          7
Transpose        5
MatMul            3
Softmax           1
...

Dynamic dimensions:
input width
...
```

然后形成：

```text
docs/SUPPORTED_OPS_V0.md
```

必须回答：

1. REC 真实需要哪些 Operator？
2. DET 真实需要哪些 Operator？
3. CLS 真实需要哪些 Operator？
4. 三者并集有多少 Operator？
5. 哪些 Operator 占主要 FLOPs？
6. 哪些只是 Shape / Metadata 操作？
7. 哪些可以在 Converter 阶段消除？
8. REC 动态宽度如何传播？

**分析完成前，不批量实现 Kernel。**

---

# 11. 第一阶段只做 REC

第一个 MVP：

```text
PP-OCRv6 tiny Recognition
```

流程：

```text
cropped text image
        ↓
resize / normalize
        ↓
rec.lwm
        ↓
Pure C Runtime
        ↓
logits
        ↓
CTC Decode
        ↓
UTF-8 text
```

第一阶段暂不做：

```text
DET
CLS
DB PostProcess
完整 OCR
```

MVP 成功标准：

> 无 OpenCV / ORT / OpenVINO / TensorRT / protobuf 的情况下，Pure C Runtime 能正确识别真实裁剪文字图片。

---

# 12. 总体架构

```text
                     Development
                         │
                     ONNX Model
                         │
                         ▼
                lw.PPOCR.ModelC
                         │
              graph compile / optimize
                         │
                         ▼
                       *.lwm
                         │
──────────────── Deployment ────────────────
                         │
                         ▼
                  lw.PPOCR.C Runtime
                         │
             ┌───────────┴──────────┐
             │                      │
          Executor                Kernels
             │                      │
             │              Conv / GEMM / ...
             │
        Tensor / Memory
             │
             ▼
           PP-OCR
       preprocess / CTC /
        DB / crop / sort
```

---

# 13. 推荐项目目录

```text
lw.PPOCR.C/
│
├─ README.md
├─ LICENSE
├─ CMakeLists.txt
│
├─ converter/
│   ├─ analyze_onnx.py
│   ├─ modelc.py
│   ├─ graph.py
│   ├─ optimize.py
│   ├─ memory_plan.py
│   └─ lwm_writer.py
│
├─ include/
│   ├─ lw_infer.h
│   └─ lw_ppocr.h
│
├─ src/
│   │
│   ├─ runtime/
│   │   ├─ model.c
│   │   ├─ tensor.c
│   │   ├─ executor.c
│   │   ├─ memory.c
│   │   └─ validate.c
│   │
│   ├─ kernels/
│   │   ├─ conv.c
│   │   ├─ depthwise_conv.c
│   │   ├─ gemm.c
│   │   ├─ activation.c
│   │   ├─ elementwise.c
│   │   ├─ pooling.c
│   │   ├─ reshape.c
│   │   ├─ transpose.c
│   │   ├─ resize.c
│   │   └─ softmax.c
│   │
│   ├─ ppocr/
│   │   ├─ preprocess.c
│   │   ├─ ctc_decode.c
│   │   ├─ classifier.c
│   │   ├─ detector.c
│   │   ├─ db_postprocess.c
│   │   ├─ crop.c
│   │   └─ sort.c
│   │
│   ├─ simd/
│   │   ├─ scalar/
│   │   ├─ x86/
│   │   └─ arm/
│   │
│   └─ platform/
│       ├─ platform.h
│       ├─ win32.c
│       └─ posix.c
│
├─ tests/
│   ├─ test_model.c
│   ├─ test_tensor.c
│   ├─ test_ops.c
│   ├─ test_rec.c
│   └─ golden/
│
├─ models/
│   └─ README.md
│
└─ docs/
    ├─ architecture.md
    ├─ lwm-format.md
    ├─ supported-models.md
    ├─ SUPPORTED_OPS_V0.md
    ├─ platform-matrix.md
    └─ win7-x86-compatibility.md
```

不要初期继续拆太多层。

---

# 14. `.lwm` 模型格式

扩展名：

```text
.lwm
```

建议解释：

```text
LightWeight Model
```

目标：

```text
简单
只读
可验证
跨平台
支持 32/64 位
Little Endian
可 mmap
不保存原生指针
尽量无需复杂反序列化
```

---

# 15. `.lwm` 基本布局

第一版：

```text
┌───────────────────────────┐
│ Header                    │
├───────────────────────────┤
│ Tensor Table              │
├───────────────────────────┤
│ Node Table                │
├───────────────────────────┤
│ Operator Parameters       │
├───────────────────────────┤
│ String Table              │
├───────────────────────────┤
│ Weight Blob               │
└───────────────────────────┘
```

所有引用统一使用：

```text
offset
index
```

禁止保存：

```text
pointer
size_t
native handle
compiler-native struct
```

这样同一个 `.lwm` 文件应可以被：

```text
Win7 x86
Windows x64
Linux x64
Linux ARM64
```

读取。

---

# 16. Header 初步设计

```c
typedef struct lw_model_header {
    uint32_t magic;
    uint32_t version;

    uint32_t flags;
    uint32_t tensor_count;
    uint32_t node_count;

    uint64_t tensor_offset;
    uint64_t node_offset;
    uint64_t param_offset;
    uint64_t string_offset;
    uint64_t weight_offset;

    uint64_t file_size;
    uint64_t workspace_size;

    uint64_t checksum;
} lw_model_header;
```

建议：

```text
magic = "LWM1"
version = 1
```

加载时必须检查：

```text
magic
version
file_size
offset bounds
table bounds
tensor bounds
node bounds
weight bounds
checksum
integer overflow
```

禁止信任模型中的长度、数量和 offset。

---

# 17. Tensor 描述

```c
#define LW_MAX_DIMS 8

typedef enum {
    LW_DTYPE_F32 = 1,
    LW_DTYPE_I32 = 2,
    LW_DTYPE_I64 = 3,
    LW_DTYPE_U8  = 4
} lw_dtype;

typedef struct lw_tensor_desc {
    uint32_t dtype;
    uint32_t ndim;

    int32_t dims[LW_MAX_DIMS];

    uint32_t flags;
    uint32_t reserved;

    uint64_t data_offset;
    uint64_t data_size;

    uint64_t workspace_offset;
} lw_tensor_desc;
```

动态维度可以暂定：

```text
-1
```

但第一阶段只支持 PP-OCR 实际需要的动态 Shape。

不要实现通用 symbolic shape engine。

---

# 18. Node 描述

```c
#define LW_MAX_NODE_INPUTS  8
#define LW_MAX_NODE_OUTPUTS 4

typedef struct lw_node {
    uint16_t op;
    uint16_t input_count;
    uint16_t output_count;
    uint16_t flags;

    uint32_t inputs[LW_MAX_NODE_INPUTS];
    uint32_t outputs[LW_MAX_NODE_OUTPUTS];

    uint64_t param_offset;
} lw_node;
```

输入输出使用：

```text
Tensor ID
```

不保存 ONNX Tensor Name。

Runtime 不依赖大量字符串/hash table。

---

# 19. Operator 参数

不要让 `lw_node` 放一个巨型 union。

每种 Operator 定义自己的参数结构。

例如：

```c
typedef struct {
    int32_t kernel_h;
    int32_t kernel_w;

    int32_t stride_h;
    int32_t stride_w;

    int32_t pad_top;
    int32_t pad_bottom;
    int32_t pad_left;
    int32_t pad_right;

    int32_t dilation_h;
    int32_t dilation_w;

    int32_t groups;
} lw_conv_param;
```

Node 只保存：

```text
param_offset
```

---

# 20. Operator 策略

原则：

> 只实现 PP-OCR 实际模型需要的 Operator。

可能包括：

```text
Conv
Depthwise Conv
Add
Mul
Div
Relu
Sigmoid
HardSwish
MaxPool
AveragePool
Concat
Split
Reshape
Transpose
Slice
Gather
Unsqueeze
Squeeze
Resize
MatMul
Gemm
Softmax
```

最终名单以模型扫描结果为准。

每增加一个 Operator，必须回答：

> 哪个正式支持的 PP-OCR 模型需要它？

回答不出来，不实现。

---

# 21. Kernel 开发策略

每个 Kernel 第一版必须先有：

```text
Reference Scalar Implementation
```

例如：

```text
conv_scalar.c
gemm_scalar.c
```

先正确，再优化。

后续：

```text
Scalar
  ↓
SSE2
AVX2
NEON
```

推荐统一 dispatch：

```text
kernel API
   ↓
dispatch
   ├─ scalar
   ├─ SSE2
   ├─ AVX2
   └─ NEON
```

---

# 22. SIMD 平台策略

## Windows / Linux x64

```text
SSE2 baseline
AVX2 optional
```

x86-64 本身包含 SSE2，可将 SSE2 作为正式 x64 基线。

## Windows 7 x86

```text
Scalar baseline
SSE2 optional
```

启动时通过 CPUID：

```c
if (lw_cpu_has_sse2()) {
    kernels = &lw_kernels_sse2;
} else {
    kernels = &lw_kernels_scalar;
}
```

Win7 x86 不要求 AVX2 优化。

## Linux ARM64

```text
NEON baseline
```

---

# 23. Conv

第一版直接朴素实现：

```text
N
OC
OH
OW
IC
KH
KW
```

后续：

```text
v0.1
naive direct convolution

v0.3
tiled im2col + GEMM

v0.4
blocked GEMM
SSE2 / AVX2 / NEON
```

Depthwise Conv 必须有专用 Kernel，不能长期完全依赖普通 Conv。

---

# 24. GEMM

统一接口：

```c
void lw_sgemm(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k);
```

路线：

```text
v0.1 scalar
v0.3 cache blocking
v0.4 packing + SIMD
v0.4 multi-thread
```

目标不是击败 MKL。

目标是：

> 对 PP-OCR 模型达到可接受性能，并保持代码规模可控。

---

# 25. 内存系统

禁止每个节点：

```text
malloc
free
malloc
free
```

Converter 做：

```text
Tensor lifetime analysis
```

例如：

```text
Tensor A
node1 → node2 → node3
                 ↑
              last use
```

之后：

```text
A 的 workspace 可以复用
```

Converter 输出：

```text
workspace_size
tensor.workspace_offset
```

Runtime 只创建一次：

```c
workspace = malloc(model->workspace_size);
```

最终目标：

```text
推理阶段：
malloc = 0
free   = 0
```

---

# 26. Windows 7 x86 内存策略

Win7 x86 最大问题不是模型大小，而是 32 位地址空间。

因此必须从设计上控制：

```text
workspace
temporary buffers
batch
thread scratch
im2col
```

建议定义：

```text
runtime_memory_limit
```

例如默认：

```text
Win7 x86:
512 MB workspace soft limit

x64:
根据模型正常规划
```

如果模型需要：

```text
740 MB workspace
```

Win7 x86 应返回：

```text
LW_ERROR_MEMORY_LIMIT
```

而不是硬分配后崩溃。

---

# 27. 避免巨型 im2col

不要长期采用：

```text
Conv
 ↓
完整巨大 im2col
 ↓
GEMM
```

推荐：

```text
Conv
 ↓
tiled im2col
 ↓
small GEMM blocks
```

这样同时有利于：

```text
Win7 x86 地址空间
CPU Cache
现代 x64 性能
ARM64 性能
```

---

# 28. REC Batch 策略

Win7 x86：

```text
default batch = 1
recommended = 1～4
```

x64 / ARM64：

```text
可按实际性能扩大
```

不要为了吞吐一次创建巨大 REC batch。

---

# 29. 多线程策略

v0.1：

```text
single-thread
```

后续：

```text
fixed thread pool
```

不要每个 Operator 创建线程。

Win7 x86 推荐：

```text
1～2 threads
```

x64 / ARM64 再根据 CPU 自动设置。

原则：

```text
共享权重
共享模型
少量 thread-local scratch
```

不要：

```text
每线程复制一套模型
每线程复制完整 workspace
```

---

# 30. 模型加载

Windows：

```text
CreateFileW
CreateFileMappingW
MapViewOfFile
```

Linux：

```text
open
mmap
```

模型尽量直接 mmap。

Runtime 使用：

```text
base + offset
```

访问：

```text
Tensor Table
Node Table
Weight Blob
```

无法 mmap 时可提供普通 read fallback。

---

# 31. Runtime 与 PP-OCR 分层

必须严格分开。

## lw.infer

只知道：

```text
Tensor
Graph
Operator
Kernel
Memory
```

不知道：

```text
OCR
DB
CTC
文字
字典
```

## lw.ppocr

负责：

```text
Image preprocess
DET
DB Postprocess
Crop
Sort
CLS
REC
CTC Decode
```

这样 Runtime 可以独立测试。

---

# 32. PP-OCR 完整流程

最终：

```text
Image
  ↓
Resize / Normalize
  ↓
DET
  ↓
DB PostProcess
  ↓
Text Boxes
  ↓
Perspective Crop
  ↓
Sort
  ↓
CLS
  ↓
REC
  ↓
CTC Decode
  ↓
UTF-8 Text
```

开发顺序：

```text
REC
↓
CLS
↓
DET
↓
DB PostProcess
↓
Full OCR
```

---

# 33. REC 动态宽度

PP-OCR REC 常见：

```text
[1, 3, 48, W]
```

其中：

```text
W = runtime dynamic
```

第一版 Shape 系统只需要支持：

```text
batch fixed
channel fixed
height fixed
width dynamic
```

Converter 尽可能提前消除 Shape-only Graph。

不要为 PP-OCR 之外的模型实现通用动态 Shape 引擎。

---

# 34. 图优化

第一阶段只做低风险优化：

```text
Constant Folding
Remove Identity
Precompute Shape
Reshape Simplification
```

后续：

```text
Conv + BN Fusion
Conv + Bias Fusion
Activation Fusion
Transpose Elimination
```

所有优化必须有数值一致性测试。

---

# 35. 权重 Packing

第一版 `.lwm` 保存标准 FP32 权重。

例如 Conv：

```text
[OC, IC, KH, KW]
```

保持 `.lwm` 跨平台。

后续 Runtime 初始化时可以做轻量 Packing。

不要一开始做：

```text
x86 专用 lwm
arm64 专用 lwm
```

除非未来有明确性能收益。

---

# 36. Public C API

低层：

```c
typedef struct lw_model lw_model;
typedef struct lw_session lw_session;
```

建议：

```c
int lw_model_load(
    const char* path,
    lw_model** model);

void lw_model_free(
    lw_model* model);

int lw_session_create(
    lw_model* model,
    lw_session** session);

int lw_session_run(
    lw_session* session,
    const lw_tensor* input,
    lw_tensor* output);

void lw_session_free(
    lw_session* session);
```

---

# 37. PP-OCR C API

```c
typedef struct lw_ppocr lw_ppocr;
typedef struct lw_ppocr_result lw_ppocr_result;

typedef struct {
    const uint8_t* data;
    int width;
    int height;
    int stride;
    int channels;
} lw_image;
```

创建：

```c
int lw_ppocr_create(
    const char* det_model,
    const char* cls_model,
    const char* rec_model,
    const char* dictionary,
    lw_ppocr** engine);
```

推理：

```c
int lw_ppocr_run(
    lw_ppocr* engine,
    const lw_image* image,
    lw_ppocr_result** result);
```

释放：

```c
void lw_ppocr_result_free(
    lw_ppocr_result* result);

void lw_ppocr_destroy(
    lw_ppocr* engine);
```

内部统一 UTF-8。

---

# 38. 第一阶段 REC API

MVP 可以先：

```c
int lw_ppocr_recognize(
    lw_ppocr* engine,
    const lw_image* cropped_image,
    char* text,
    size_t text_capacity,
    float* score);
```

后续再规范成结构化 result。

---

# 39. C ABI 与 32/64 位要求

公共 C API：

```text
opaque handle
POD struct
fixed-width integer
UTF-8
explicit ownership
explicit free
```

不要在持久格式中使用：

```text
long
unsigned long
size_t
pointer
```

模型格式统一：

```text
uint32_t
uint64_t
int32_t
float
```

C API 中 buffer size 可以使用 `size_t`，但 ABI 稳定时需谨慎评估 Win32 / Win64 差异。

第一阶段 API 暂不冻结。

---

# 40. Windows 平台层

最低：

```text
Windows 7 SP1
```

允许使用 Win7 已有 API。

平台层建议封装：

```c
void* lw_map_file(...);
void  lw_unmap_file(...);

uint64_t lw_clock_ns(...);

int lw_cpu_count(...);

int lw_thread_create(...);
```

Runtime / Kernel / OCR Core 不直接调用 Windows API。

---

# 41. Windows 工具链

主力开发：

```text
Modern CMake
Modern MSVC
Ninja
```

Win7 x86 单独维护：

```text
Compatibility Build
```

不要为了 Win7 x86 把整个项目锁死在旧编译器。

CI 逻辑：

```text
Primary:
  Windows x64
  Linux x64
  Linux ARM64

Compatibility:
  Windows 7 x86
```

必须有真实 Win7 SP1 x86 运行验证，不能仅以“编译成功”作为支持依据。

---

# 42. LARGEADDRESSAWARE

Win7 x86 构建可以考虑：

```text
/LARGEADDRESSAWARE
```

但设计不能依赖：

```text
/3GB boot option
特殊系统配置
```

仍按普通 32 位地址空间可以正常运行进行设计。

---

# 43. Linux

目标：

```text
Linux x64
Linux ARM64
```

平台层优先：

```text
POSIX
pthread
mmap
clock_gettime
```

ARM64 使用：

```text
NEON
```

作为主力 SIMD。

---

# 44. 第三方依赖

Runtime：

```text
0 个大型第三方依赖
```

最好最终只有：

```text
libc
OS API
```

Converter 可以依赖：

```text
Python
onnx
numpy
```

测试可以使用：

```text
ONNX Runtime
NumPy
OpenCV
```

作为 reference，但不能进入部署 Runtime。

---

# 45. 正确性标准

任何自研 Runtime：

> 正确性优先于性能。

每个 Operator 都必须有 Reference Test。

方法：

```text
固定输入
↓
NumPy / ONNX Runtime reference
↓
lw.PPOCR.C
↓
误差比较
```

FP32 初始建议：

```text
absolute error < 1e-5
relative error < 1e-4
```

具体算子按数值特性调整。

---

# 46. Golden Test

至少建立：

```text
REC Golden Test
```

保存：

```text
input image
expected text
reference output
reference score
```

对比：

```text
ONNX Runtime
lw.PPOCR.Inference
lw.PPOCR.C
```

目标：

```text
文本一致
置信度接近
```

后续：

```text
CLS label
DET boxes
box IoU
Full OCR
```

---

# 47. 性能测试

从 v0.1 就记录：

```text
load time
model mapped bytes
workspace
inference latency
peak memory
```

正式 benchmark：

```text
mean
median
P95
min
max
```

参考对比：

```text
lw.PPOCR.Inference OpenCV DNN
```

第一版性能不作为硬验收条件。

---

# 48. Win7 x86 特别测试

必须额外测试：

```text
32-bit address overflow
workspace > limit
large tensor size multiply
model offset overflow
batch memory
multi-thread scratch
mmap / file mapping failure
```

32 位环境必须保证：

> 超限返回明确错误，不允许整数回绕、越界和崩溃。

---

# 49. 稳定性测试

成熟后至少：

```text
1000 create/destroy
10000 sequential inference
multi-thread
multi-instance
memory growth
handle growth
invalid input
corrupt model
```

畸形 `.lwm` 必测：

```text
wrong magic
wrong version
bad offset
bad node index
huge tensor
integer overflow
bad weight range
truncated file
checksum failure
```

错误模型只能：

```text
return error
```

不得：

```text
crash
OOB
abort
```

---

# 50. 错误处理

Pure C 不使用 exception。

建议：

```c
typedef enum {
    LW_OK = 0,

    LW_ERROR_INVALID_ARGUMENT = -1,
    LW_ERROR_IO = -2,
    LW_ERROR_MODEL_FORMAT = -3,
    LW_ERROR_UNSUPPORTED = -4,
    LW_ERROR_OUT_OF_MEMORY = -5,
    LW_ERROR_MEMORY_LIMIT = -6,
    LW_ERROR_INFERENCE = -7,
    LW_ERROR_INTERNAL = -8
} lw_status;
```

提供错误信息接口。

库代码禁止：

```text
abort
exit
printf 后直接终止进程
```

---

# 51. 日志

第一版只需：

```c
typedef void (*lw_log_callback)(
    int level,
    const char* message,
    void* user);
```

默认：

```text
OFF
```

不要实现复杂日志框架。

---

# 52. 编码风格

坚持：

```text
small
explicit
boring
predictable
```

不要：

```text
宏魔法
模拟 C++ 对象系统
过度泛型
复杂接口层
无必要代码生成
过度抽象
```

推荐：

```text
struct
enum
function
pointer
array
```

Pure C 就保持 Pure C 风格。

---

# 53. 安全原则

模型文件视为不可信输入。

所有：

```text
offset + size
count * sizeof
shape element multiply
workspace size
tensor byte size
node index
weight range
```

必须做整数溢出和边界检查。

32 位 Win7 尤其严格。

---

# 54. 版本路线

## v0.1.0 — REC MVP

```text
PP-OCRv6 tiny REC
FP32
CPU
single-thread
Scalar
.lwm
Pure C Runtime
Windows x64
Linux x64
CTC Decode
```

成功标准：

> 无 OpenCV / ORT / OpenVINO / TensorRT / protobuf，正确识别裁剪文字图。

Win7 x86 不阻塞 v0.1。

---

## v0.2.0 — Full PP-OCR

增加：

```text
CLS
DET
DB PostProcess
Crop
Sort
Full OCR
```

---

## v0.3.0 — Memory / Kernel Optimization

增加：

```text
Workspace planner
zero malloc inference
tiled im2col
optimized Conv
optimized GEMM
weight packing
```

---

## v0.4.0 — SIMD / Threads

增加：

```text
x64 SSE2
x64 AVX2
ARM64 NEON
x86 SSE2
CPU dispatch
thread pool
```

---

## v0.5.0 — Compatibility / Platform Validation

正式验证：

```text
Windows 7 SP1 x86
Windows 7 SP1 x64
Windows 10/11 x64
Linux x64
Linux ARM64
```

建立：

```text
Windows 7 x86 Compatibility Profile
```

---

## v1.0.0

仅在以下全部满足后进入：

```text
PP-OCRv6 Full OCR stable
API frozen
LWM format frozen
Win7 x86 verified
Windows x64 verified
Linux x64 verified
Linux ARM64 verified
golden tests
stability tests
memory regression tests
performance report
```

---

# 55. 第一阶段详细执行顺序

新会话收到本设计后，严格按以下顺序。

## Step 1

找到并确认：

```text
det.onnx
cls.onnx
rec.onnx
ppocr_keys.txt
sample images
```

优先复用 `lw.PPOCR.Inference` 中 PP-OCRv6 tiny 资源。

---

## Step 2

实现：

```text
converter/analyze_onnx.py
```

---

## Step 3

生成：

```text
docs/SUPPORTED_OPS_V0.md
```

---

## Step 4

仅针对 REC 设计：

```text
docs/lwm-format.md
```

定义：

```text
LWM v0
```

暂时不冻结为稳定格式。

---

## Step 5

实现：

```text
converter/lwm_writer.py
```

做到：

```text
rec.onnx
↓
rec.lwm
```

---

## Step 6

实现：

```text
src/runtime/model.c
src/runtime/validate.c
```

做到：

```text
rec.lwm
↓
成功加载
↓
打印/检查 tensor、node、weight
```

此时还不推理。

---

## Step 7

实现：

```text
tensor
executor
memory
```

形成：

```c
for each node:
    dispatch(op)
```

---

## Step 8

按 REC 实际算子逐个实现 Scalar Kernel。

每一个 Operator 都先写测试。

---

## Step 9

实现 REC preprocess：

```text
resize
normalize
HWC → CHW
```

不要依赖 OpenCV。

---

## Step 10

实现：

```text
CTC Decode
UTF-8 dictionary
```

---

## Step 11

跑第一张真实裁剪文字图。

流程：

```text
image
 ↓
Pure C preprocess
 ↓
rec.lwm
 ↓
Pure C inference
 ↓
CTC
 ↓
UTF-8 text
```

---

## Step 12

与 reference 对比：

```text
ONNX Runtime
lw.PPOCR.Inference
```

---

# 56. v0.1 验收标准

```text
[ ] rec.onnx 可转换 rec.lwm

[ ] Runtime 不依赖 OpenCV

[ ] Runtime 不依赖 ONNX Runtime

[ ] Runtime 不依赖 protobuf

[ ] Runtime 不依赖 Python

[ ] Runtime 源码为纯 C

[ ] rec.lwm 可正确 load / mmap

[ ] 模型格式有完整 bounds checking

[ ] REC 所需 Operator 已全部实现

[ ] 每个 Operator 有独立测试

[ ] 支持 REC 必要动态宽度

[ ] CTC Decode 正确

[ ] 至少 10 张真实文字图片结果与 reference 一致

[ ] Windows x64 通过

[ ] Linux x64 通过

[ ] Release 构建无明显 warning

[ ] 推理过程无持续内存增长
```

性能暂不作为硬指标。

---

# 57. Win7 x86 正式验收标准

到 v0.5 时增加：

```text
[ ] Win7 SP1 32-bit 实机可启动

[ ] rec.lwm 与 x64 使用同一模型文件

[ ] Scalar 可用

[ ] SSE2 CPU 可自动启用 SSE2

[ ] 无 SSE2 时可回退 Scalar

[ ] 默认 batch 1 可稳定运行

[ ] 2-thread 内存可控

[ ] workspace 超限返回 LW_ERROR_MEMORY_LIMIT

[ ] 32-bit integer/offset overflow 测试通过

[ ] 不依赖特殊 /3GB 系统配置

[ ] Full OCR 真实样本结果与 x64 一致
```

---

# 58. README 第一屏建议

```markdown
# lw.PPOCR.C

Tiny pure-C inference runtime for PP-OCR.

`lw.PPOCR.C` is a lightweight inference runtime designed
specifically for PP-OCR models.

## Runtime dependencies

- No Python
- No OpenCV
- No ONNX Runtime
- No OpenVINO
- No TensorRT
- No protobuf

## Platforms

- Windows 7 SP1 x86 — Compatibility
- Windows 7 SP1 x64
- Windows 10/11 x64
- Linux x86_64
- Linux ARM64

> This is not a general-purpose ONNX Runtime.
```

---

# 59. 与 lw.PPOCR.Inference 的边界

`lw.PPOCR.C` 负责：

```text
model format
runtime
executor
operator
kernel
PP-OCR execution
```

`lw.PPOCR.Inference` 继续负责：

```text
统一 C ABI
C#
CLI
HTTP Service
Demo
多 Runtime 管理
部署体系
```

成熟后可以输出：

```text
lw.PPOCR.Runtime.C.dll
liblw.PPOCR.Runtime.C.so
```

接入现有 Runtime 体系。

核心关系：

> `lw.PPOCR.C` 做发动机，`lw.PPOCR.Inference` 做统一 SDK 与部署平台。

---

# 60. 最重要的开发纪律

始终坚持：

> **只实现当前正式支持的 PP-OCR 模型真正需要的东西。**

不要因为 ONNX 有某项能力就实现它。

不要把项目做成：

```text
mini ONNX Runtime
```

而要保持：

```text
PP-OCR dedicated runtime
```

---

# 61. 新会话拿到本文件后的第一项工作

**不要重新讨论项目是否值得做。**

**不要直接开始写大批 Kernel。**

第一项任务：

> 对现有 PP-OCRv6 tiny 的 `det.onnx`、`cls.onnx`、`rec.onnx` 做完整模型结构分析，输出 Operator、Shape、Initializer、动态维度、opset 和 Shape-only 节点统计，并形成正式 `SUPPORTED_OPS_V0.md`。

分析完成后，再进入：

```text
LWM v0
↓
REC converter
↓
REC runtime
```

---

# 62. 第一阶段一句话目标

> **在完全不依赖 OpenCV、ONNX Runtime、OpenVINO、TensorRT、protobuf 和 Python 的部署环境中，使用纯 C Runtime 加载 `rec.lwm`，正确完成 PP-OCRv6 tiny 的文字识别。**

---

# 63. 给新会话 / Codex 的直接执行指令

请严格基于本设计继续开发，不重新扩大项目边界。

当前阶段固定为：

```text
PP-OCRv6 tiny
REC first
FP32
CPU
Pure C Runtime
custom .lwm
Windows x64 + Linux x64 first
Win7 x86 compatibility preserved by design
```

必须优先复用现有 `lw.PPOCR.Inference` 中 PP-OCRv6 tiny 的模型、字典、样例和 reference 结果，但新 Runtime 不依赖其中 OpenCV、ONNX Runtime、OpenVINO、TensorRT 的运行实现。

第一步先完成模型分析，明确：

```text
REC operators
DET operators
CLS operators
union
dynamic shape
shape-only ops
converter-removable ops
main FLOPs
```

然后生成：

```text
docs/SUPPORTED_OPS_V0.md
```

在此之前，不批量编写 Kernel。

后续严格按照：

```text
Model Analysis
→ LWM v0
→ REC Converter
→ Runtime Loader
→ Scalar Kernels
→ REC Preprocess
→ CTC
→ Golden Test
→ Full OCR
→ Memory Optimization
→ SIMD
→ Win7 x86 Compatibility
```

推进。

所有功能扩展必须有明确 PP-OCR 模型需求依据。

---

# 64. 最终设计摘要

整个项目最终应形成：

```text
                        same .lwm
                           │
                  ┌────────┴────────┐
                  │ lw.PPOCR.C Core │
                  └────────┬────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
 Windows 7 x86      Windows/Linux x64    Linux ARM64
 Scalar / SSE2        SSE2 / AVX2            NEON
 Compatibility          Primary              Primary
```

核心承诺：

```text
Pure C
PP-OCR focused
one model format
zero heavy runtime dependency
Win7 x86 compatible
x64 / ARM64 optimized
portable
auditable
small
stable
```

这就是 `lw.PPOCR.C` 的开发边界和长期方向。
