# C 与 C# OCR 引擎对比方法

本项目提供手动工作流 “C vs C# OCR Engine Comparison”，用于比较当前
lw.PPOCR.C 与固定版本 SimdPaddleOCR 的完整 OCR 延迟、内存和文字准确率。
它是优化决策工具，不是普通提交门禁，也不用于证明某个项目在所有硬件上更快。

## 固定合同

机器可读合同位于 ci/csharp-engine-comparison.json。当前 v1 固定：

- SimdPaddleOCR 仓库和 40 位 commit；
- PP-OCRv6 Tiny；
- 1 与 4 个 line workers；
- 3 个独立 GitHub-hosted Windows x64 replicas；
- 100 张生成图片，其中第 1 张为 warm-up，后续 99 张计入统计；
- Normalized fixed-320 与 Product 两种 profile；
- performance 与 accuracy 仅报告，contract/correctness 才会导致失败。

合同变化必须和报告器、测试及工作流放在同一提交中审查。不要只改 YAML 中的参数。

## 两种 profile

### Normalized fixed-320

该 profile 用于尽量隔离 Runtime 与算子实现差异：

| 项目 | C# | C |
| --- | --- | --- |
| 模型 | Tiny | Tiny |
| REC 策略 | fixed | fixed |
| REC width | 320 | 320 |
| ISA | AVX2 | AVX2 |
| DET/CLS | 合同中的显式参数 | 同一组显式参数 |

任一引擎实际 ISA 不是 AVX2，case 直接失败，不生成“近似公平”的结果。

### Product

该 profile 用于观察两个项目按各自产品策略运行时的实际表现：

| 项目 | C# | C |
| --- | --- | --- |
| REC 策略 | native adaptive | adaptive max |
| REC width 字段 | 320（不是上限） | 最大 960 |
| ISA | best available | 当前本机最佳 C backend |

这一节的宽度策略有意不同，因此不能把结果解释成纯 Runtime 效率差异。

## 可复现性与运行顺序

数据集只在 prepare-dataset job 生成一次，记录每个文件及 metadata.json 的
SHA-256，然后由全部 replicas 下载同一个 artifact。

Tiny LWM 和字典由当前 lw.PPOCR.C commit 转换一次并记录 SHA-256。Windows
comparison job 在自己的 runner 上现场编译 lw_ppocr_c.dll，再把 DLL、LWM 和字典
作为 external-only assets 交给 benchmark harness。缺少任意文件都会失败，不允许联网
下载替代 DLL 或模型。

每个 replica 内同时运行 C# 和 C。奇数 replica 按 C# → C，偶数 replica 按
C → C#，降低固定运行顺序对 CPU boost、温度和后台负载的影响。

最终只先在每个 replica 内计算 C/C# 比值，再汇总三个 paired ratios 的中位数与范围。
不同 runner CPU 的绝对延迟不会直接合并。

## 手动运行

在 GitHub Actions 页面选择 “C vs C# OCR Engine Comparison”：

1. include_product 控制是否额外运行 Product profile；
2. keep_detailed_results 控制最终 artifact 是否保留逐图片 disagreement JSON；
3. 点击 Run workflow。

Normalized profile 始终执行，replica 数、worker 数、模型和固定参数由合同控制，不能从
UI 临时改动。

## 输出

Actions Step Summary 直接显示：

- 两个仓库 commit、合同 SHA 和数据集 SHA；
- 每个 replica 的 CPU、内存和实际 ISA；
- 1/4 worker 的 paired latency ratio；
- Exact Line Rate 与 CER；
- peak working set 和 peak growth 的 paired ratio；
- 当前 gate policy。

最终 artifact 名为 lw-ppocr-csharp-comparison-<lw commit>，包含：

- SUMMARY.md；
- manifest.json；
- 每个 replica 实际加载的 lw_ppocr_c.dll SHA-256；
- 固定合同副本；
- dataset/model SHA256SUMS；
- 三个 replicas 的原始 benchmark JSON；
- disagreement 汇总；
- 可选的逐图片 disagreement JSON。

## 硬失败条件

以下情况会让工作流失败：

- harness、lw commit、合同、数据集或模型 SHA 不一致；
- C Engine 未使用 external-only assets；
- Normalized 实际 ISA 不是 AVX2；
- C 输出在 replicas 间不确定；
- C 与 C# 不在同一 replica machine；
- case 缺失、结果 schema/计数无效或参考行数不一致。

性能慢于另一实现、CER 较高或 Exact Line Rate 较低在 v1 中只属于分析信息。

## 结果边界

GitHub-hosted runner 结果适合发现趋势与优化方向，不等同于固定实体机基准。对正式发布的
性能声明，仍需在明确 CPU、内存、系统、电源策略和重复次数的实体机上复测。
