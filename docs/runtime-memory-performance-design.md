# Runtime 内存与性能开发设计

> 状态：开发中。M0 的 session 级内存报告、M1 workspace planner v2、M2 的 GELU 物理生命周期计划、M3 的 CTC terminal-output elision + AVX2 tiled scratch、M5 的 crop 批量预留/worker-local streaming 实验，以及 M6 的跨 REC 宽度 prepared 常量共享、M7.1 的 layout eligibility/转换遥测、M7.2 的 REC NHWC 预处理 parity 原型、M7.3 的 Conv1x1 局部 A/B、M7.4 的真实 REC shape profile，以及 M7.5 的 Conv1x1/MatMul FMA 开关拆分已在当前工作树实现；实验开关默认关闭，M6 为兼容性检查通过后自动生效的内部内存优化，尚未据此宣称发行版收益。三模型 100 图质量基线、M3/M5 三模型 1/4-worker 复核、同进程多图 benchmark，以及 Small/Medium 100 图 1-worker 三轮 M6 配对已完成；4-worker 重复和跨平台复核仍待补齐。
>
> 设计基线：2026-09-18，`lw.PPOCR.C` v1.0.0，提交 `17931ce4526ad61df9e5c2f2c7789e3a1192bffc`。
> 参考源码：`sdcb/SimdPaddleOCR` 提交 `3686197b32539b9a60c604d07752b8d9b7896b1d`。

## 1. 目标与边界

下一阶段优先减少无效张量存储、重复准备和数据搬运，再优化剩余算子热点。
不重新设计整个 Runtime，不改变 v1 C ABI，不把大规模 NHWC 改造作为前置条件。

主目标：

1. 降低 Tiny、Small、Medium 的 workspace 与完整 OCR 峰值内存。
2. 在相同模型、输入、REC 宽度策略、ISA 和 worker 数下，降低完整 OCR 延迟。
3. 将收益归因到具体模块，保留可关闭、可回退的基线实现。
4. 保持 Native、WASM、Android、Java 等入口共用同一套语义。

不变项：

- REC 最大宽度仍为 **960**；不通过把默认值改成 320 获得性能提升。
- 保留 192/320/480/640/960 自适应策略；固定 320 只用于诊断性对比。
- 三种模型继续复用现有 CLS；Small/Medium 继续使用对应的共享字典。
- 不改变检测框坐标、阅读顺序、CLS 应用规则、CTC blank/repeat 规则和 score 含义。
- 不添加 OpenCV、ONNX Runtime 或新的 Runtime 第三方依赖。
- 不在本机安装 WASM/Android 编译环境；这些平台由现有 CI 验证。
- 不复活已删除且出现负优化的 `perf/nhwc-runtime-v2` 分支。
- 不改写已发布的 v1.0.0 tag；未来发行版本另行决定。

本设计不承诺某个百分比收益。历史 benchmark 和其他项目的数据只能提出假设，不能作为本次验收结果。

## 2. 当前代码已经有什么，还缺什么

| 模块 | 当前状态 | 本轮缺口 |
| --- | --- | --- |
| Workspace | 原始图生命周期、空闲块合并、first-fit 分配、64 字节对齐 | 尚未针对全图离线排布；融合后不再写出的中间张量仍占规划空间 |
| 融合 | executor 已有符合条件的 AVX2 GELU 等路径 | 执行跳过节点，但内存仍按语义节点分配 |
| CTC | 紧凑 indices/probabilities，省掉调用层完整概率输出缓冲 | session 内仍保留完整 logits；generic workspace 未自动缩小 |
| 常量共享 | model/dictionary 共享；初始 REC clone 可共享 prepared constants；兼容的 REC 宽度会共享 packed arena | 需要 Small/Medium、多 worker 和长生命周期 A/B，确认共享收益及失败回退没有隐藏峰值 |
| 多行调度 | 已有宽度优先、动态领取和按原始索引写回 | 不必重写调度；应处理串行裁剪、整页 crop 缓冲及每次请求建线程成本 |
| Prepared Execution | 实验性绑定表与部分直接执行路径 | 默认仍关闭；需区分 dispatch 收益与内存收益 |
| CLS | 每 worker 独立实例 | 可评估只读资源共享，但不能先验认为是最大内存项 |
| 内存验证 | session、workspace 限额、Golden、WASM heap 测试 | 缺少组件级分解、规划下界与完整请求生命周期报告 |

主要入口：

- [memory.c](../src/runtime/memory.c)：`lw_plan_workspace`。
- [session.c](../src/runtime/session.c)：shape、workspace、常量准备与创建。
- [session_internal.h](../src/runtime/session_internal.h)：内部张量/session 元数据。
- [executor.c](../src/runtime/executor.c)：融合与 CTC 专用执行。
- [recognizer.c](../src/ppocr/recognizer.c)：宽度切换、缓存与输出缓冲。
- [ocr.c](../src/ppocr/ocr.c)：crop、worker、任务领取与结果写回。
- [prepared-execution.md](prepared-execution.md)：现有实验路径范围。

图执行阶段没有大块分配，不等于完整 OCR 请求没有分配。尺寸切换、裁剪扩容、后处理需要单独统计。

## 3. 如何借鉴 SimdPaddleOCR

### 3.1 借鉴机制，不复制性能结论

可借鉴的方向包括：生命周期离线排布、融合中间值消除、CTC 最后投影内存收缩、只读常量/shape plan 共享、安全的原地执行与 Concat 消拷贝，以及前处理与 layout/kernel 的联合评估。

实现必须服从本项目的 C ABI、内存限额和跨平台约束。
对方历史比较使用了其他版本的 C DLL，并存在 REC 策略差异，不能据此宣称当前 v1.0.0 快或慢多少。

### 3.2 CTC score 是明确的兼容性边界

参考源码的紧凑 ArgMax 路径使用最大 logit 作为紧凑 score；不能直接替换本项目的 Softmax 概率。
本项目保留：

```text
probability(t, best) = exp(logit(t, best) - max(logits(t)))
                      / sum_c exp(logit(t, c) - max(logits(t)))
```

只为最终发出的字符计算概率，不意味着可以省掉这些行的完整分母。
文字相同不能证明 score 兼容；score 必须参与正确性验证。

### 3.3 固定版本的源码参考

- [性能记录](https://github.com/sdcb/SimdPaddleOCR/blob/3686197b32539b9a60c604d07752b8d9b7896b1d/docs/perf.md)
- [Small 内存对比](https://github.com/sdcb/SimdPaddleOCR/blob/3686197b32539b9a60c604d07752b8d9b7896b1d/docs/small_memory_vs_c.md)
- [InferenceSession](https://github.com/sdcb/SimdPaddleOCR/blob/3686197b32539b9a60c604d07752b8d9b7896b1d/src/Sdcb.SimdPaddleOCR/OnnxSharp/InferenceSession.cs)
- [CompiledModel](https://github.com/sdcb/SimdPaddleOCR/blob/3686197b32539b9a60c604d07752b8d9b7896b1d/src/Sdcb.SimdPaddleOCR/OnnxSharp/CompiledModel.cs)
- [LayoutPlanner](https://github.com/sdcb/SimdPaddleOCR/blob/3686197b32539b9a60c604d07752b8d9b7896b1d/src/Sdcb.SimdPaddleOCR/OnnxSharp/LayoutPlanner.cs)
- [MatMul ArgMax](https://github.com/sdcb/SimdPaddleOCR/blob/3686197b32539b9a60c604d07752b8d9b7896b1d/src/Sdcb.SimdPaddleOCR/Kernels/MatMul.ArgMax.cs)

参考代码若实际引入，应逐文件核对许可证、保留归属；优先按算法思路独立实现。

### 3.4 对最新 SimdPaddleOCR 主线的核对结论

截至本设计编写时，参考项目 README 将其定位为纯 C#、CPU-only、直接接收交错像素内存的 PP-OCRv6 推理库；模型由程序集资源直接读取，调用方负责图片解码，不强制创建中间 BGR 图。其主线 `LayoutPlanner` 按 ISA 与连续算子段的 kernel 能力选择图级 NHWC，并在段边界插入显式 layout-convert；当前 README 已记录 AVX2、ns2、x64 scalar 和 .NET 10 AdvSimd 路径，其他不满足条件的段仍回退 NCHW。这个边界非常适合借鉴为本项目的实验开关，而不适合直接改写 LWM 逻辑 shape。

参考项目最新性能文档还显示：Conv1x1 仍是 x64 AVX2 的最大算子项，ARM64 上 MatMul/Erf 的占比更高；其跨引擎表同时显示 C 运行时通常工作集更小、但墙钟仍有优化空间。文档明确按同一 runner、同一 ISA、成对样本报告比值，而不是混合不同机器的绝对毫秒。对本项目的直接结论是：

1. 先完成生命周期/布局报告，再决定是否引入局部 NHWC；不能用参考项目的绝对速度推断本项目收益。
2. 输入端应优先评估 `BGR/RGB/BGRA/RGBA + stride` 的零中间拷贝入口，但不能破坏现有 C ABI；可先作为内部 crop/preprocess buffer 实验。
3. MatMul/CTC 的紧凑 ArgMax 只能作为候选内核；本项目的 rec score 仍要求完整 Softmax 分母语义，必须单独验证 score、文本、box 和确定性。
4. 参考项目的 NHWC 只覆盖满足条件的连续段；本项目应采用同样的局部、可回退策略，绝不恢复已判定为负优化的全图 NHWC 分支。

依据：参考项目 README 的输入/模型边界与平台说明，以及其当前性能记录中的同机配对、Conv1x1/MatMul/Erf 热点数据（见文末链接）。

> 主线更新核对（2026-09）：SimdPaddleOCR 当前 README 已记录 1.4 将图级 NHWC 从 AVX2 扩展到 ns2、x64 scalar 和 .NET 10 AdvSimd，并把预处理直接写入 NHWC；其公开报告声称 tiny-4-worker 的工作集由约 817 MiB 降到约 515 MiB（不同项目、runner 和 workload，不能直接移植为本项目收益）。因此本项目后续 NHWC 实验不应只覆盖 AVX2：应先实现统一的 layout eligibility/收益报告，再分别评估 AVX2、NEON/AdvSimd 和 scalar fallback，且保留 NCHW 回退。

## 4. 分阶段交付

| 阶段 | 交付 | 前置条件 | 首轮范围 |
| --- | --- | --- | --- |
| M0 | 内存分解、身份记录、可复现基线 | 无 | 是 |
| M1 | 离线 workspace planner v2 | M0 | 是 |
| M2 | 执行计划与融合内存生命周期一致 | M1 | 已实现实验版，先 GELU；默认关闭 |
| M3 | CTC greedy 专用计划、分块 logits scratch | M0/M1；与 M2 计划接口一致 | 已实现实验版：terminal Softmax 物理输出省略 + AVX2 packed 路径 8 行 scratch；默认关闭 |
| M4 | 安全原地执行、特定 Concat 消拷贝 | M2 | 按内存报告选择 |
| M5 | 每 worker crop 缓冲与流水处理 | M0 | 已实现批量预留和可回退 worker-local streaming；需多图/大图 A/B 后决定是否保留 |
| M6 | 跨宽度只读准备资源共享、计划缓存 | M0/M1 | 已实现兼容性签名与引用计数共享；默认安全生效，需多模型/多图 A/B 决定是否扩展 resident plan |
| M7.1 | layout eligibility/转换遥测 | M0/M6 | 已实现分析版，仅报告 NCHW fallback，不改变默认执行 |
| M7.2 | 小范围 NHWC 热点实验 | M7.1 | 已完成预处理 parity 原型；尚未接入生产调度 |
| M7.3 | NHWC Conv1x1 局部 A/B | M7.2 | 已完成分析驱动；当前候选为负优化，不接入生产 |
| M7.4 | 真实 REC Conv1x1 shape profile | M7.3 | 已实现诊断工具；不改变默认 dispatch |

每阶段独立提交、测试、报告。不要同时打开 FMA、Prepared Execution、新 planner 和 NHWC 后只给一组总结果。

## 5. M0：建立可解释的基线

### 5.1 统计内容

增加内部诊断，不扩展公开结构体、不添加公共导出函数。
拟新增 `src/runtime/memory_diagnostics.c/.h` 与测试 driver，具体文件拆分可按现有结构调整。

每个 DET/CLS/REC session 至少记录：

- 模型标识、input shape、ISA、执行策略、worker 编号、REC 宽度。
- model 文件字节数、独占/共享模型存储、唯一 packed arena 字节数。
- workspace 分配字节、对齐开销、语义图 live-bytes 下界、规划耗时。
- input、CTC indices/probabilities、CTC scratch、crop 的 capacity。
- shape/宽度缓存数量、缓存 workspace 总量、常量重复份数。
- create、首次推理、稳定推理、宽度切换时间及分配次数。

共享资源按唯一对象计数；不能把同一 arena 按 worker 累加。
区分逻辑引用量与实际拥有量，不能把文件大小再次累加成独立内存。
进程层另记 Windows Peak Working Set、可获得时的 private bytes，以及 Linux/macOS RSS。
WASM 记录线性内存和增长次数；浏览器/JS 内存不能混作 WASM heap。

### 5.2 下界与报告

`workspace-report-driver` 当前额外输出以下只读准备资源字段，便于把 workspace 与常量 arena 分开解释：

- `packed_weight_bytes`：当前 session 唯一拥有的 packed 权重 arena 字节数；共享后每个引用仍报告逻辑可用大小，但进程层只能按唯一 arena 计数。
- `prepared_constant_nodes`：具备 prepared constant 描述的节点数。
- `prepared_constants_ref_count`：该 arena 的引用计数，用于验证跨宽度/clone 是否真的共享。

报告仍不等价于进程 RSS。宽度切换时应同时保存旧 session、新 session 的报告及引用计数，避免把逻辑引用量误报为独占内存。

每个执行时点同时存活张量的对齐大小之和，其最大值是规划参考下界，不保证可达到。
M2 后另记物理执行计划下界，不能混用两种生命周期图。
alias group 只计一次，workspace 外的 input/constant 不放进 workspace 下界。

报告建议使用内部 `schema_version: 1`：

```text
identity        源码 SHA/dirty、编译器、flags、二进制/模型/数据 SHA
environment     OS、CPU、实际 ISA、CPU 配额、内存与计时条件
configuration   模型、worker、DET threads、REC 策略、CLS
sessions        shape/plan、workspace、共享/独占常量、缓冲容量
requests        create/first/warm/shape-switch 时序与内存快照
paired_results  原始 A/B 样本、配对比值、结果校验、置信信息
```

较重的张量 dump 单独运行；计时模式只保留低成本计数，不能让诊断改变热点。

### 5.3 基线可信度与验收

重编当前基线，不使用历史 `build-*` 中的现成 exe。
检查实际宽度和 flags；默认值与源码不符的旧二进制排除出报告。
动态链接时记录实际加载 DLL/so/dylib 路径及 SHA，防止两组误加载同一个库。

验收：三模型、1/4 worker、960 策略均有报告；初始化、空白页、密集页、大图后小图的内存变化可解释。

当前 M0 已先交付 session 级报告驱动 `workspace-report-driver`。它不属于公共 ABI，输出
`schema_version`、模型文件/校验和、shape、workspace 字节、语义 live-byte 下界和峰值节点。
生成候选构建后可直接运行：

```powershell
& build-memory-candidate/Release/workspace-report-driver.exe `
  build-memory-candidate/models/rec.lwm 320 1
```

若 workspace 小于语义下界，`session_planner` CTest 会失败；这项门禁同时覆盖 v1 和 v2。
进程 RSS、crop、packed arena 和三组件汇总仍属于后续 M0 扩展，不应从这个 session 报告推断。

## 6. M1：离线 workspace planner v2

### 6.1 保持语义不变

首版只改 workspace offset，不改算子顺序、kernel、数值计算或输出。
复用模型验证与 shape 解析，不绕过字节数上限和溢出检查。

内部 interval 包含 tensor id、aligned bytes、birth、last use、alignment。
使用闭区间：同一节点读 input 并写 output 时二者同时存活；尚未授权原地覆盖。
graph output 保持到运行结束；constant/input 仍由外部持有。

### 6.2 算法

1. 生成完整 interval 列表。
2. 大小降序；同大小按 birth、tensor id 稳定排序。
3. 对每个 interval 找到生命周期相交的已放置区间。
4. 按地址扫描冲突块，在最低可用的 64 字节对齐位置放置。
5. 检查 offset + size 溢出、总量限制和区间重叠。
6. 同时计算旧 planner；若新布局更大，首版选择旧布局并记录原因。

地址和字节数使用经过检查的 `size_t`/`uint64_t` 转换。
避免 uint32 与 64 位 SIZE_MAX 的恒假比较触发 GCC `-Wtype-limits`。
规划成本发生在创建阶段，但仍需测大节点数模型创建时间；不能过度搜索换很小的空间收益。
诊断期可双跑规划器，稳定后再决定是否保留双规划。
失败时释放临时内存，不发布半完成 layout。

### 6.3 测试与验收

扩展 [tests/test_session.c](../tests/test_session.c)，另加 synthetic interval 单元测试：

- chain、fork/join、长短寿命混合、相同大小、空图集合、合法零长度张量。
- input/output 同节点的边界、graph output 持久性。
- live interval pair 不重叠、offset 对齐、所有区间在 workspace 内。
- 限额刚好够/少一个字节、极大 shape、32 位地址空间、分配失败清理。
- 多次创建 offsets 相同；相同输入完整结果与基线一致。
- 三模型真实图 workspace 不大于旧布局，报告创建耗时变化。

收益为零时保留诊断成果即可，不能为制造收益降低输入分辨率或减少输出。

## 7. M2：融合与内存规划使用同一份执行计划

### 7.1 为什么不能只删中间 tensor

executor 能匹配 GELU 子图、直接写最终输出并跳过中间节点，但内存已经按完整语义图规划。
直接删除中间空间，会让没有命中融合的执行路径写入无效地址。
因此不能只在 `memory.c` 中识别几个 op 然后缩小空间。

### 7.2 两层计划

保留 LWM 语义图用于验证、诊断、profile 索引与 generic 模式。
新增内部物理执行计划，描述实际 kernel/group、读写 tensor、被消除的中间值、物理 birth/last-use、eligibility 和 scratch 生命周期。

建议创建顺序：

```text
模型/shape 验证
→ 确定执行策略与可用 kernel，生成物理计划
→ 按物理计划规划 workspace
→ 分配 workspace / 准备只读常量
→ 绑定指针与 Prepared Execution 元数据
→ 完整校验后发布 session
```

若 kernel eligibility 依赖预打包元数据，应提前生成必要元数据或调整准备顺序；不能在空间已裁剪后才发现 kernel 不可用。
语义与物理生命周期用不同字段，避免破坏依赖原图 last-use 的现有匹配器。

### 7.3 第一版仅做已有 GELU 融合

抽取共享纯匹配函数，plan 与 executor 复用同一结果，禁止维护不同条件的两份匹配代码。
沿用现有常量、shape、消费者、ISA 条件，不扩大数值近似范围。
被消除 tensor 不再分配；final output 从融合开始时就必须有有效空间。

中间 tensor 若是 graph output、存在第二消费者或诊断要求 materialize，创建时退回完整语义计划。
调试 materialization 也在创建时选择，不能运行中临时访问已不存在的中间值。
紧凑计划执行时不得临时 fallback 到被删除的逐节点路径；运行中不变量破坏应明确报错。

验收：

- generic/fused 计划可分别创建、运行、销毁。
- AVX2 与非 AVX2 构建通过；WASM/ARM 不因 unused helper 而 `-Werror` 失败。
- 分支、graph-output 中间值、非标准常量等负例确实走 generic。
- profile 保留原节点编号，不把 skipped 节点当成漏执行。
- 融合算术不变时，与原 fused 路径逐元素一致。

## 8. M3：CTC greedy 专用内存计划

当前工作树已实现 M3 的第一步和分块 scratch 路径，开关为
`LW_EXPERIMENTAL_CTC_TILED=ON`（同时要求 `LW_EXPERIMENTAL_WORKSPACE_PLANNER_V2=ON`）。
recognizer 会在确认 terminal、last-axis Softmax 的 greedy CTC 结构后，以内部 plan flags
重建 session；generic `lw_session_create` 仍保留完整 graph-output contract。物理计划只将
不会被 greedy executor 写出的 terminal Softmax output 标记为 skipped，`workspace-report-driver
... --ctc-greedy` 会门禁该标记。具备 AVX2 packed 投影时，executor 进一步按 8 行分块写入
有界 logits scratch；非 AVX2 后端不分配这块 scratch，并继续使用正确的 generic fallback。

Tiny REC 在本地 320/960 报告中该 tensor 被安全省略，但峰值 workspace 仍没有下降，因为
其 semantic live peak 发生在更早节点，且当前 matmul/add 投影输出仍需保留给 packed kernel。
这项结果仍有价值：它证明了“不为永不 materialize 的输出分配独立空间”的生命周期契约，
并且分块 scratch 已在完整 OCR 样本上保持文本、分数和框完全一致；但不能宣称端到端内存
收益。后续若要进一步降低峰值，必须单独评估投影输出的物理省略，不能把这一步的零收益隐藏掉。

### 8.1 不改变 generic session 输出契约

公开 session 仍能取完整 graph output。
只给内部 recognizer 增加明确 greedy 执行策略，不让 greedy session 冒充完整输出 session。
新增接口仅放内部头文件，不进入公共 export manifest。

拆成两个独立提交：

1. greedy 计划不为永不 materialize 的最终 Softmax 输出安排独立空间。
2. 最终 MatMul + bias 按时间步分块，用有界 logits scratch。

第一步先检查 fallback 实际写哪些 tensor，不能因为有 compact probabilities 就删除 logits。

### 8.2 分块策略

时间步 T、类别 V 的完整 logits 大小为 `T * V * sizeof(float)`。
候选使用 `B * V * sizeof(float)` scratch，先测 B=1/2/4/8，不预设最快值。

```text
执行 REC 前缀，保留投影输入
for 每个时间块:
    使用既有 packed 权重计算本块 logits + bias
    按原 tie-breaking 取得 best index
    跨块维护 previous index，处理 blank/repeat
    仅为发出的字符计算原 Softmax 概率
    写紧凑 index/probability 数组
执行原有 CTC 文本解码
```

投影输入与 scratch 同时存活；scratch 纳入 workspace 规划与 max_workspace 限额。
不能先分配完整 workspace 再收缩，也不能在栈上创建词典大小的巨大数组。
保持 packed 权重布局，先实现现有 AVX2 eligible 路径，其余后端保留正确 fallback。
不同时切换 FMA、修改 bias 加法或 Softmax 求和次序。
保留非有限值处理、相等值选择、Unicode 解码与重复字符规则。

测试覆盖：

- T<B、T 不整除 B、最后一块单行。
- 跨块重复、blank 分隔重复、全 blank、全相等 logits。
- 极大/极小 logits、非有限输入的既有错误语义。
- 三模型不同 V 和全部宽度。
- text、index、字符概率和最终 rec_score，不只比较文本 checksum。
- generic 完整输出仍可取得，公开 API 行为不变。

分别报告 scratch、session workspace、进程 RSS；三者不会必然同比下降。
B 太小导致投影变慢时调整 B，或只对收益明确的 shape 启用。

## 9. M4：原地执行与 Concat 消拷贝

不是 M1 默认能力，仅在报告证实占用显著后实施。
原地 elementwise 要求：输入最后使用、非外部 input/constant、无其他观察者、大小/layout 相同、kernel 明确支持 alias。
广播输入不能因单消费者就覆盖；并行块不能有跨块读写依赖。
核查 `restrict` 约定与向量读取顺序，不能只修改 output 指针。

Concat 首版只考虑可证明的连续 NCHW 拼接，例如有效 outer=1，输入能直接落在最终输出连续片段。
生产者单消费者、slice 对齐、重复输入、外部输出、分支和非连续 layout 都要明确判定。
alias group 生命周期先取保守并集，再考虑更细 slice lifetime。

当前 [scalar_layout.c](../src/kernels/scalar_layout.c) 使用 `memcpy`，不能让 input/output 同址后继续调用。
新路径要显式跳过已就位 slice，拒绝未经证明的部分重叠。
消拷贝可能延长巨大输出的生命周期，最终按完整 workspace/耗时 A/B 决定保留与否。

## 10. M5：裁剪缓冲与 worker 流水

### 10.1 最小改动

当前先串行生成整页 crop 再启动识别 worker，crop 按历史高水位保留。
长行优先和动态领取已经存在，不重复开发。
当前工作树已先实现安全的批量预留：DET 后几何尺寸全部确定时，先累加预计 crop 字节数并
一次 reserve；栅格化后若单个 quad 被 canonicalize 成更大尺寸，仍保留逐 crop 兜底扩容。
这一步只减少 realloc/copy 次数，不改变 aggregate crop 的峰值容量，也不能冒充 worker-local
streaming 的内存收益。
当前工作树还提供 `LW_EXPERIMENTAL_STREAMING_CROPS=ON`：不再建立整页 aggregate crop，
每个 worker 持有可增长的私有 crop buffer，在领取任务后直接从只读源图完成透视裁剪。
默认路径不启用该开关；它必须经过密集页、长行页、大图后小图、多 worker 失败回收和峰值
RSS A/B 后，才能决定是否进入默认构建。
可评估稳定 O(n log n) 宽度排序，但保留相同宽度时原始索引顺序。

### 10.2 每 worker 持有 crop

```text
DET + 后处理 → 不可变 crop 几何任务表
worker 领取 → 本 worker crop buffer → CLS/旋转 → REC → 按原索引写回
所有 worker join → 阅读顺序/公开结果
```

每个 worker 独占可增长 crop buffer；源图只读且有效到全部任务结束。当前实验实现保留
worker buffer 的高水位，下一次请求复用，不把 capacity 保留误判为泄漏。
CLS 原地旋转只作用本 worker crop，不修改调用者图像。
warp 插值、尺寸取整、坐标及 CLS/REC 前处理与旧路径一致。
内存从整页 crop 总容量变为各 worker crop 历史容量之和，不能声称所有输入都必然降低。
测大→小→大序列；容量保留不等于泄漏。

### 10.3 并发与失败

- 保留动态领取，原始 line index 唯一确定输出槽。
- 分配/推理失败后，不提前释放其他线程仍使用的图像或任务表。
- 保留线程创建失败的安全执行/回收路径，避免丢任务、重复任务。
- 不扩大同一 OCR handle 的并发契约。
- 验证失败后重试、close、存在时的取消路径。

首版不顺便引入永久线程池。只有建线程成本显著时，再独立设计线程池销毁、idle 内存和并发边界。

## 11. M6：跨宽度资源共享与 Prepared Execution

资源分为：

1. 模型级只读：原始权重、字典、可共享 packed weights。
2. shape 级不可变：shape、offset 计划、kernel 选择、绑定描述。
3. 实例级可变：workspace、input、输出、scratch、错误状态。

先修宽度切换重复准备同一 packed arena，不先默认常驻五个完整 session。
共享 key 包含模型/权重身份、dtype、pack kind、实际 ISA、影响布局的 kernel 参数。
若 packing eligibility/layout 依赖 shape，将相关条件纳入 key，不能假设所有宽度可共享。
初始 clone 避免先生成昂贵副本再替换为共享引用造成瞬时峰值。
引用计数/失败回收覆盖 owner 先销毁、clone 继续运行。

后续可把多个 immutable shape plan 绑定到同一 worker 高水位 workspace，但该 worker 不能同时执行两个 plan。
新 shape 创建失败时旧 plan 仍可用；不同 worker 绝不共享可变 workspace。
先保持当前两宽度缓存策略；resident-five 是另一实验，不混入共享收益。

Prepared Execution 复用现有内部表，仅绑定已稳定的 plan。
Conv1x1/Conv3x3/MatMul 是否默认开启分别以 dispatch A/B 决定，不借内存修改顺便推广 FMA。

### 11.1 当前实现

当前工作树已在 `lw_session` 内实现兼容性检查和引用计数共享：

1. 新 REC 宽度 session 先完成模型、ISA、节点数和 prepared 表建立，但只计算布局签名，不立即分配 packed arena。
2. 只有模型指针、实际 SIMD 等级、节点 kind、packed offset/count 以及 arena 大小全部一致时，才调用 `lw_session_share_prepared_constants`；不兼容时才为候选 session 分配并 pack 自己的 arena。
3. 共享只读 packed arena；workspace、输入、输出、CTC scratch、错误状态仍保持 session 私有。
4. 不兼容时保留新 session 独立准备的资源，不改变识别路径；任一共享失败都会释放候选 session 并返回原错误。
5. owner/clone 释放由 arena 引用计数管理，先释放任一 session 不会悬空另一 session 的 packed 指针。

这条路径没有新增公开 C ABI，也没有把五个宽度 session 常驻化。它只消除实际发生的重复 packed 权重准备，并避免兼容宽度切换时的瞬时双 arena；shape plan 和可变 workspace 仍按 session 独立管理。

### 11.2 当前证据与限制

Tiny 模型在本地 x64 报告中，`packed_weight_bytes` 为 **4,207,680 bytes**，与 REC 宽度 320/960 无关；因此每个额外兼容 session 理论上可避免约 4.01 MiB 的重复 packed arena（4,207,680 / 1024²）。这只是组件级上限，不等于完整 OCR RSS 节省。

同一台 Windows x64/AVX2 机器上，现有 Small/Medium 转换产物也显示相同的宽度不变量：

| 模型 | packed arena | workspace 320 | workspace 960 | prepared nodes |
| --- | ---: | ---: | ---: | ---: |
| Tiny | 4,207,680 | 2,948,160 | 8,844,480 | 22 |
| Small | 19,349,568 | 6,680,000 | 20,040,000 | 32 |
| Medium | 69,499,648 | 8,847,360 | 26,542,080 | 34 |

这些是 `workspace-report-driver` 的组件级结果，不是完整 OCR 的 RSS/Peak Working Set。它们证明 M6 的共享候选在三种模型上都有实际内存意义：若多个兼容宽度 session 同时常驻，理论上分别可避免约 4.01、18.45、66.28 MiB 的重复 packed arena；最终是否值得保留 resident 多宽度，还必须以 1/4 worker、连续图片和全进程峰值测量为准。

`tests/test_session.c` 已覆盖 320/960 prepared-source 创建、兼容性、同一 packed 指针和引用计数释放；`workspace-report-driver` 已输出上述字段。还需要在 Small/Medium、1/4 worker、连续宽度切换和全进程 RSS 场景复测，才能决定是否扩展 plan cache 或调整默认 session 缓存策略。

### 11.3 端到端 AB/BA（本机证据）

使用 `compare_rec_runtime_profiles.py --paired-rounds 5`，在同一 Windows x64/AVX2 主机、同一 960 宽度和同一模型资产下，每轮启动独立进程并交替 compact-first/performance-first。每个条目的 `rounds × iterations` 已列出；两边文本行数与 FNV-1a checksum 全部一致：

| 模型 | workers | rounds × iterations | 延迟变化（resident 相对 compact） | Peak RSS 变化 | checksum |
| --- | ---: | ---: | ---: | ---: | --- |
| Tiny | 1 | 5 × 2 | -2.71% | -0.48 MiB | 一致 |
| Tiny | 4 | 3 × 2 | -10.69% | -18.34 MiB | 一致 |
| Small | 1 | 5 × 2 | -1.93% | -6.43 MiB | 一致 |
| Small | 4 | 5 × 2 | -14.73% | -94.02 MiB | 一致 |
| Medium | 1 | 5 × 1 | -0.77% | -51.25 MiB | 一致 |
| Medium | 4 | 5 × 1 | -5.61% | -399.66 MiB | 一致 |

这组结果支持 M6 的主要结论：它消除了 resident 多宽度的重复 packed arena，并在 4-worker 下同时降低了 RSS 和延迟；1-worker 延迟收益接近噪声。它仍是本机证据，不是跨机器发布承诺；resident 继续保持实验性，100 图集和跨平台复核完成前不改变 compact 默认路径。

### 11.4 三模型 100 图质量基线

使用项目自有生成器生成的 100 图核心数据集（seed `20260907`，manifest
SHA-256 `c51474cb3761515c8c9b07c0afb1846303aba1b2304ef9a87043c9d8c282159d`，
614 条参考线），在同一 Native x64 构建、REC 最大宽度 960、Tiny CLS 和对应
字典下重新评估 Tiny/Small/Medium。生成图片和报告位于被 Git 忽略的
`build-local-data/`，不进入源码或发布包。

| 模型 | Detection F1 | Exact reference line rate | CER on matched lines |
| --- | ---: | ---: | ---: |
| Tiny | 99.35% | 58.14% | 3.65% |
| Small | 99.92% | 78.66% | 2.01% |
| Medium | 100.00% | 80.62% | 1.26% |

这组结果与 `docs/full-ocr-golden-corpus.md` 的项目基线一致，说明当前
M6 改动没有改变三模型的质量口径；它是质量/回归证据，不是性能或真实用户
场景的发布承诺。下一步仍需在同一数据集上增加每图耗时、RSS 和 resident/compact
AB/BA 配对，才能决定是否改变默认 session 缓存策略。

### 11.5 同进程多图 resident/compact 采样

新增 `full-ocr-dataset-benchmark`，读取 PPM image list，在同一个 `lw_ocr`
实例中完成 warm-up 和完整数据集迭代，避免逐图重启进程掩盖跨宽度共享的
常驻成本。它输出数据集平均/P95 OCR 时间、总行数、聚合 checksum、warm-up
后 RSS、逐图 RSS 采样范围和 OS peak RSS；compact 与 resident 只改变构建开关，模型、图片顺序、
REC 960 和 worker 数保持一致。

先把项目数据集转换为 benchmark-only PPM 列表（输出目录应放在被 Git 忽略的
`build-local-data/`）：

```powershell
python tools/prepare_ocr_dataset_ppm.py `
  --dataset build-local-data/lw-generated-ocr `
  --output build-local-data/lw-generated-ocr-ppm `
  --force

build-memory-base/Release/full-ocr-dataset-benchmark.exe `
  build-memory-base/models/det.lwm build-memory-base/models/cls.lwm `
  build-memory-base/models/rec.lwm models/ppocrv6-tiny/ppocr_keys.txt `
  build-local-data/lw-generated-ocr-ppm/images.txt 1 2 4 960
```

使用交替顺序执行 compact/resident，并让比较器校验 manifest、worker、宽度、
行数和 checksum：

```powershell
python tools/compare_ocr_dataset_runtime.py `
  --compact-driver build-memory-base/Release/full-ocr-dataset-benchmark.exe `
  --resident-driver build-performance-vs/Release/full-ocr-dataset-benchmark.exe `
  --det build-memory-base/models/det.lwm `
  --cls build-memory-base/models/cls.lwm `
  --rec build-memory-base/models/rec.lwm `
  --dictionary models/ppocrv6-tiny/ppocr_keys.txt `
  --image-list build-local-data/lw-generated-ocr-ppm/images.txt `
  --benchmark-manifest build-local-data/lw-generated-ocr-ppm/benchmark-manifest.json `
  --warmup 1 --iterations 2 --workers 4 --target-width 960 `
  --paired-rounds 3 `
  --json-output build-local-data/tiny-resident-vs-compact.json `
  --markdown-output build-local-data/tiny-resident-vs-compact.md
```

当前 Windows x64/AVX2 采样如下（resident 相对 compact；图片为同一 100 图
manifest；Small/Medium 为单次完整数据集迭代，属于方向性证据）：

| 模型 | workers | iterations | 延迟变化 | P95 变化 | Peak RSS 变化 | checksum |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Tiny | 1 | 2 | +4.34% | +6.91% | +6.54 MiB | 一致 |
| Tiny | 4 | 2 | -3.72% | -3.79% | +19.29 MiB | 一致 |
| Small | 1 | 1 | -0.98% | -0.29% | +9.85 MiB | 一致 |
| Small | 4 | 1 | +34.50% | +38.94% | +36.97 MiB | 一致 |
| Medium | 1 | 1 | +4.40% | +4.07% | +23.93 MiB | 一致 |
| Medium | 4 | 1 | +0.21% | +2.37% | +60.40 MiB | 一致 |

这组结果说明 resident 多宽度缓存的收益强依赖模型和并发度：它可能改善
Tiny 4-worker 的吞吐，但会稳定增加峰值 RSS；Small 4-worker 的单轮结果出现
明显回退，Medium 4-worker 延迟接近持平但 RSS 增加约 60 MiB。因此 resident
仍不应进入默认构建；下一步需要重复 Small/Medium 采样，并把结论与多图 RSS
采样一起交给 CI 复核。

对同一 manifest 的前 10 张图又执行了 3 轮交替 AB/BA（1 worker），用于估计
跨轮次噪声：

| 模型 | rounds | 延迟变化（resident 相对 compact） | Peak RSS 变化 | checksum |
| --- | ---: | ---: | ---: | --- |
| Small | 3 | +8.96% | +11.33 MiB | 一致 |
| Medium | 3 | +2.69% | +17.56 MiB | 一致 |

Medium 的第一轮存在明显冷启动/调度离群值，后两轮接近持平；这正是采用
AB/BA 和中位数、而不是单轮绝对值的原因。10 图结果只用于噪声分析，不能替代
100 图性能验收；当前证据仍支持 resident 保持实验性。

### 11.6 M3/M5 实验组合的多图复核

在 `LW_EXPERIMENTAL_WORKSPACE_PLANNER_V2=ON`、`LW_EXPERIMENTAL_CTC_TILED=ON`
和 `LW_EXPERIMENTAL_STREAMING_CROPS=ON` 的实验构建中，使用同一 Tiny 100 图
数据集与 2 次完整迭代复核默认 compact 构建。文本 checksum、620 条输出行均一致：

| workers | 延迟变化（实验相对默认） | P95 变化 | Peak RSS 变化 |
| ---: | ---: | ---: | ---: |
| 1 | -2.31% | -4.11% | -12.37 MiB |
| 4 | -7.32% | -7.33% | -10.52 MiB |

这只证明 M3/M5 在 Tiny 的完整多图样本上没有出现单图特例回退。Small 也已
完成 100 图、1/4-worker 的同迭代比较；Medium 目前完成同一 10 图子集的
1-worker smoke，输出均为 56 行且 checksum 与默认构建一致。实验开关继续保持
默认关闭。

Small 100 图的实验相对默认结果为：1 worker 延迟 **-2.34%**、P95 **-5.18%**、
Peak RSS **-21.15 MiB**；4 worker 延迟 **-2.35%**、P95 **-5.69%**、Peak RSS
**-25.68 MiB**。这组结果支持继续对 M3/M5 做多模型验证，但还不能据此默认开启。

Medium 100 图、1 worker 的实验相对默认结果为：延迟 **-1.24%**、P95 **-0.90%**、
Peak RSS **-50.97 MiB**，checksum 和 614 行输出一致。至此 M3/M5 已在三模型
完整 100 图配置上完成复核；Small 和 Medium 均已覆盖 1/4 worker。

Medium 100 图、4 worker 的实验相对默认结果为：延迟 **-2.13%**、P95 **-3.11%**、
Peak RSS **-49.79 MiB**，checksum 和 614 行输出一致。

### 11.7 连续多图 RSS 采样

full-ocr-dataset-benchmark 现在在每张 warm-up/timed 图片完成并释放输入后采样一次当前 RSS，报告：

- rss_sample_count：有效采样数；
- min_sampled_rss_bytes / max_sampled_rss_bytes：连续请求中的当前 RSS 范围；
- peak_rss_bytes：平台进程峰值（Windows Working Set peak、Linux/macOS 的 rusage 口径）。

peak_rss_bytes 仍是发布比较的主口径；max_sampled_rss_bytes 用来检查宽度切换或多 worker 请求中是否出现只在中间阶段发生的瞬时双份 arena。两者不能互相替代，也不能把 RSS 采样直接解释成泄漏结论。

本地 Windows x64、AVX2、Tiny、10 图、1 warm-up + 1 timed iteration、1 worker 的 smoke：

| 配置 | checksum | 采样数 | max sampled RSS | process peak RSS |
|---|---|---:|---:|---:|
| compact | f243c4064db65317 | 20 | 99.17 MiB | 104.51 MiB |
| resident | f243c4064db65317 | 20 | 102.85 MiB | 107.99 MiB |

该 smoke 仅确认观测字段和结果一致性；10 图样本不足以决定 resident 默认策略。4-worker 重复采样和未覆盖平台仍按 checklist 执行。

### 11.8 Small 100 图三轮 AB/BA

在同一 Windows x64/AVX2 主机、同一 100 图 manifest、REC 960、1 worker 下，
Small 完成 3 轮 compact-first / resident-first 交替。每个进程为 1 次 warm-up
和 1 次 timed iteration，单次包含 100 张图片；每张 warm-up/timed 图片均记录
RSS 样本，三轮 checksum 和 613 条输出行一致。

| 配置 | OCR mean (ms/image) | OCR P95 (ms/image) | Peak RSS | Max sampled RSS |
|---|---:|---:|---:|---:|
| compact | 791.034 | 1059.494 | 203.438 MiB | 187.699 MiB |
| resident | 816.886 | 1089.804 | 213.270 MiB | 197.484 MiB |

resident 相对 compact 的中位数变化为：mean **+3.27%**、P95 **+2.86%**、
Peak RSS **+9.83 MiB**、max sampled RSS **+9.79 MiB**。三轮输出 checksum
均为 8705f04d72739195。这组完整数据集证据不支持把 resident 多宽度缓存
设为 Small 的默认构建；它继续保持实验性。

### 11.9 Medium 100 图三轮 AB/BA

同一 Windows x64/AVX2 主机、同一 100 图 manifest、REC 960、1 worker 下，Medium 也完成 3 轮 compact-first / resident-first 交替。每轮包含 1 次 warm-up 和 1 次 timed iteration；逐图 RSS 采样覆盖 warm-up 与 timed 图片，三轮输出 checksum 和 614 条结果行一致。

| 配置 | OCR mean (ms/image) | OCR P95 (ms/image) | Peak RSS | Max sampled RSS |
|---|---:|---:|---:|---:|
| compact | 3886.069 | 5345.429 | 608.566 MiB | 548.766 MiB |
| resident | 4094.914 | 5562.578 | 632.551 MiB | 572.758 MiB |

resident 相对 compact 的变化为：mean **+5.37%**、P95 **+4.06%**、Peak RSS **+23.98 MiB**、max sampled RSS **+23.99 MiB**。三轮输出 checksum 均为 f9331b391e95a309。该结果与 Small 的完整数据集结论一致：在单 worker 连续多图场景下，resident 仍带来稳定内存增加且没有延迟收益，因此不应进入默认构建；后续只在 4-worker 与跨平台数据支持时再评估是否值得扩展。
### 11.10 宽度切换序列与 M6 首轮多模型证据

为避免连续 benchmark 只按原始图片顺序运行，新增
`tools/build_ocr_width_switch_manifest.py`。它读取同一数据集的 `metadata.json`
和 `prepare_ocr_dataset_ppm.py` 生成的 `benchmark-manifest.json`，校验
`source_manifest_sha256`，按每张图的中位 `natural_width_at_height_48` 将请求归入
192/320/480/640/960 代表 bucket，再按低/高交替顺序重复输出 PPM 路径。该工具
只改变请求顺序，不修改图片、Runtime 或 C ABI；Runtime 仍对每个检测 crop 独立
选择 REC 宽度。输出列表和报告必须放在被 Git 忽略的 `build-local-data/`。

生成序列：

```powershell
python tools/build_ocr_width_switch_manifest.py `
  --dataset build-local-data/lw-generated-ocr/metadata.json `
  --ppm-manifest build-local-data/lw-generated-ocr-ppm/benchmark-manifest.json `
  --output-list build-local-data/lw-generated-ocr-ppm/width-switch-images.txt `
  --output-manifest build-local-data/lw-generated-ocr-ppm/width-switch-manifest.json `
  --cycles 3
```

在同一 Windows x64/AVX2 主机上，使用该 12 项交替序列、1 次 warm-up、2 次
连续迭代、REC 上限 960，结果 checksum 均保持一致：

| 模型 | workers | 构建 | mean ms/image | P95 ms/image | peak RSS | checksum |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Tiny | 4 | compact | 130.047 | 172.138 | 123.69 MiB | b5ad6ed0938a51ed |
| Tiny | 4 | resident | 152.891 | 221.300 | 132.27 MiB | b5ad6ed0938a51ed |
| Tiny | 1 | compact | 203.592 | 256.578 | 104.19 MiB | b5ad6ed0938a51ed |
| Tiny | 1 | resident | 202.622 | 256.464 | 109.54 MiB | b5ad6ed0938a51ed |

同一序列的 Small/Medium 4-worker 首轮 smoke（1 次 warm-up、1 次 timed
iteration）如下：

| 模型 | compact mean | resident mean | 延迟变化 | compact peak RSS | resident peak RSS | RSS 变化 | checksum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Small | 544.623 ms | 371.888 ms | -31.70% | 241.46 MiB | 256.06 MiB | +14.59 MiB | c1e9bce211d3bcc7 |
| Medium | 2847.182 ms | 2762.864 ms | -2.96% | 668.47 MiB | 708.46 MiB | +39.97 MiB | d17c033d412db12f |

这些 Small/Medium 数据只有单轮，不能替代 3 轮 AB/BA，也不能证明 resident
适合默认构建。它们只证明宽度切换序列能够触发真实的连续请求压力，并显示
resident 的典型代价是增加常驻/峰值工作集；后续必须补齐独立进程的多轮配对、
1-worker、跨平台和大图→小图场景，再决定是否保留该实验开关。

`tests/test_build_ocr_width_switch_manifest.py` 覆盖 manifest 身份校验、bucket
排序、重复周期和缺失/不匹配输入；生成的图片、列表、报告均不进入 Git。
### 11.11 Small/Medium 4-worker 多轮配对结果

使用 `tools/compare_ocr_dataset_runtime.py` 对同一 12 项宽度切换序列执行 3 轮
交替顺序（compact-first、resident-first、compact-first），每轮 1 次 warm-up、
1 次 timed iteration、4 workers、REC 上限 960。该工具现在同时接受普通
`images` manifest 和 `build_ocr_width_switch_manifest.py` 生成的重复 `entries`
manifest，并要求每轮输出 checksum、行数和 RSS 字段一致。

| 模型 | compact mean | resident mean | 延迟变化 | compact peak RSS | resident peak RSS | RSS 变化 | checksum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Small | 444.641 ms | 447.554 ms | +0.66% | 241.25 MiB | 256.14 MiB | +14.89 MiB | `c1e9bce211d3bcc7` |
| Medium | 2163.499 ms | 2027.281 ms | -6.30% | 671.75 MiB | 711.90 MiB | +40.15 MiB | `d17c033d412db12f` |

Small 的多轮结果基本持平但增加约 14.9 MiB 峰值工作集；Medium 的多轮结果
显示约 6.3% 平均延迟下降，但增加约 40.2 MiB 峰值工作集，且单轮差异仍较大。
因此 resident-width 继续保持实验开关，不能因为 Medium 的局部收益就改成默认。
上述报告位于被 Git 忽略的 `build-local-data/runtime-small-fma/` 和
`build-local-data/runtime-medium-fma/`，可用以下命令复现：

```powershell
python tools/compare_ocr_dataset_runtime.py `
  --compact-driver build-memory-base/Release/full-ocr-dataset-benchmark.exe `
  --resident-driver build-memory-resident/Release/full-ocr-dataset-benchmark.exe `
  --det <model-dir>/det.lwm --cls <model-dir>/cls.lwm `
  --rec <model-dir>/rec.lwm --dictionary <model-dir>/ppocr_keys.txt `
  --image-list build-local-data/lw-generated-ocr-ppm/width-switch-images.txt `
  --benchmark-manifest build-local-data/lw-generated-ocr-ppm/width-switch-manifest.json `
  --warmup 1 --iterations 1 --workers 4 --target-width 960 --paired-rounds 3 `
  --json-output <model-dir>/width-switch-4w-paired.json `
  --markdown-output <model-dir>/width-switch-4w-paired.md
```
## 12. M7：可选的局部 NHWC 实验

仅当前述完成而 profile 仍显示 layout/通用卷积为主热点时启动。
从一个连续热点 segment 开始，不整体迁移 Runtime，也不改变 LWM 逻辑 shape。

必须包括边界/layout 元数据、分支/输出转换、pack/scratch/create 成本、实际 AVX2/NEON kernel、generic fallback、三模型与全 OCR 1/4 worker 验证。
只有 kernel 快而完整 OCR 不快即判不通过；不能恢复旧负优化分支当成起点。

### 12.1 M7.1：layout eligibility 与转换遥测

当前工作树已在内部 execution profile 增加分析字段，并由 full-ocr-profile-driver 输出 layout 对象：

- candidate_nodes：满足 batch-one、rank-4、group-one Conv 保守条件的候选节点数；
- selected_nodes：实际选择 NHWC 的节点数，当前固定为 0；
- fallback_nodes：候选但仍使用 NCHW 的节点数，当前应等于 candidate_nodes；
- transform_nanoseconds、transform_invocations、transform_bytes：实际 Transpose 节点的时间、次数和输出字节数。

这一步只建立可复现的布局成本基线，不修改 LWM shape、C ABI 或默认 kernel。后续 NHWC 实验必须让 selected_nodes 非零，并同时满足文本 checksum、score、box、确定性、workspace、RSS 与完整 OCR latency 门禁；否则只保留 NCHW fallback。报告中的 candidate 是保守启发式，不等于已经证明 NHWC 一定更快。
### 12.2 M7.2：REC 预处理 NHWC parity 原型

本轮完成了一个仅用于分析的 `lw_rec_preprocess_bgr_u8_nhwc` 原型。它复用现有 NCHW 预处理的尺寸计算、双线性插值、右侧 mid-gray padding 和归一化规则，只改变输出写入布局为 `[1,H,W,C]`。生产 recognizer 仍调用 NCHW 入口，LWM 图、session shape、C ABI 和默认 kernel 调度均未改变。

新增 `rec-preprocess-layout-driver` 对合成 BGR 输入在 REC 320 和 960 两个宽度执行 NCHW/NHWC 对照，并把 NCHW plane 重排到 NHWC 后逐元素比较。Windows Release 两个独立构建目录的结果均为：

```text
{"target_width":320,"resized_width":90,"max_abs_difference":0}
{"target_width":960,"resized_width":90,"max_abs_difference":0}
```

`rec_preprocess_layout` CTest 已在 `build-memory-base` 与 `build-performance-vs` 通过。该结果只证明预处理布局变换保持数值一致，不证明 NHWC 图段更快或更省内存。下一步若继续 M7，应在一个明确的 Conv/MatMul 连续段实现同一权重的 NHWC kernel 与 NCHW fallback，先做单算子 A/B，再进入完整 OCR 1/4-worker 门禁；在此之前不得把该原型接入默认路径。
### 12.3 M7.3：NHWC Conv1x1 局部 A/B 结果

在不改动 Runtime 调度的前提下，新增 
hwc-conv1x1-driver` 做一个局部候选实验：权重从同一份 OI canonical 数据分别打包为现有 NCHW `[OC/4][IC][4]` 和实验 NHWC `[IC][OC/8]`，输入/输出分别使用 NCHW 与 NHWC；x64 上用 AVX2 8-lane 输出 kernel，其他平台自动使用同一 scalar fallback。测试先把 NCHW 结果按布局映射后比较，确保数值语义一致，再测 40 次热循环。

Windows x64 AVX2、两个构建配置的 parity 均为零误差：

```text
scalar_max_abs_difference = 0
selected_max_abs_difference = 0
```

在 `IC=96, OC=192, H=12, W=32` 的代表性 shape 上，现有 NCHW packed kernel 明显更快。`build-memory-base` 重复 7 次的 `NHWC/NCHW` 比值为 **2.125x–3.000x**；`build-performance-vs` 单次为 **3.000x**。输出 checksum 均为 `-10.3748732`。因此该 `[IC][OC/8]` 局部 NHWC 候选判定为负优化，不接入 production dispatch，也不继续扩展为全图转换；驱动和结果保留用于防止重复尝试同一失败布局。

这次 A/B 也说明：仅把输出通道向量化并不能抵消当前 NCHW kernel 已经具备的 spatial blocking、packed 权重布局和 AVX2/FMA 调度优势。下一轮若继续 M7，必须改变可验证的瓶颈假设（例如融合前后 layout conversion、跨多个连续节点复用 NHWC），并先用 profile 证明转换成本值得承担；否则优先回到现有 NCHW 热点。


### 12.4：真实 REC Conv1x1 shape profile

M7.3 的局部候选使用单一合成 shape；为避免据此选择错误 kernel，新增
`tools/collect_rec_shape_profile.py`。它包装现有 `rec-profile-driver`，逐一执行
192/320/480/640/960 五档 REC 宽度，读取每个 Conv 节点解析后的 input/weight/output
shape，只保留 group=1、kernel=1×1 的节点，并按实际 shape 汇总耗时。

用法：

```powershell
python -X utf8 tools/collect_rec_shape_profile.py `
  --driver build-memory-base/Release/rec-profile-driver.exe `
  --model build-memory-base/models/rec.lwm `
  --iterations 3 `
  --json-output build-local-data/rec-shape-profile-tiny.json `
  --markdown-output build-local-data/rec-shape-profile-tiny.md
```

该工具只生成诊断报告，不修改 dispatch、LWM、C ABI 或默认布局。当前 Tiny
Windows x64 报告中，累计热点首先集中在 `160→320, h=3`、`320→160, h=3`、
`96→192, h=6`、`192→96, h=6` 等真实 REC shape；同一通道 shape 在不同宽度下
按独立样本记录。下一步 AVX2 A/B 必须优先覆盖这些 shape，并用 100 图端到端结果
决定是否晋级，不能再只依据单一 synthetic shape。

## 13. 实验开关与文件清单

以下为实验开关。M1/M2 已在当前工作树实现，但仍默认关闭；CMake 对未知 `-D` 变量可能只给 warning，必须检查实际使用与命中记录。

| 拟新增开关 | 默认 | 作用 |
| --- | --- | --- |
| `LW_EXPERIMENTAL_WORKSPACE_PLANNER_V2` | OFF | M1 布局 |
| `LW_EXPERIMENTAL_FUSION_MEMORY` | OFF | M2 物理计划，依赖 planner v2 |
| `LW_EXPERIMENTAL_CTC_TILED` | OFF | M3 greedy 物理计划；省略 terminal Softmax 输出并在 AVX2 packed 路径使用有界 scratch |
| `LW_EXPERIMENTAL_STREAMING_CROPS` | OFF | M5 worker crop |

统一 static/shared 开关；不适用 ISA 明确 fallback 并报告实际路径。
helper 定义与调用使用一致条件编译，不通过关闭 `-Werror` 解决 WASM/macOS 错误。

| 文件/区域 | 改动 |
| --- | --- |
| `src/runtime/memory.c` | M1 布局，后续消费物理生命周期 |
| `session.c`、`session_internal.h` | 内部策略、创建顺序、失败回收与共享 |
| 拟新增 `src/runtime/execution_plan.c/.h` | 物理计划、融合元数据、验证 |
| `executor.c`、`prepared_execution.c` | 消费统一计划，保留 generic |
| `src/ppocr/recognizer.c` | greedy 策略、缓存和资源绑定 |
| `src/simd/*matmul*`、CTC helper | 分块投影，概率语义不变 |
| `src/ppocr/ocr.c` | crop 生命周期与 worker 所有权 |
| `tests/test_session.c`、拟新增 planner/CTC 测试 | 布局、边界和结果契约 |
| `tools/compare_ocr_dispatch_profiles.py` | 身份、配对原始样本、内存报告 |
| `.github/workflows/x64-performance.yml` | 共用 prepared assets，单变量 A/B |
| `.github/workflows/arm64-performance.yml` | Native ARM64 正确性与性能 |

公开 `include/` ABI、LWM 格式、HTTP/SDK JSON 字段不在本轮改动范围。

## 14. 测试与 benchmark 协议

### 14.1 矩阵

| 维度 | 内容 |
| --- | --- |
| 模型 | Tiny/Small/Medium，正确字典与共享 CLS |
| worker | 1/4，显式指定 DET threads，避免两边默认策略不同 |
| 宽度 | 主测 960 上限；五档宽度边界与切换序列 |
| 图像 | bundled sample、100 图自有集、长行、密集、空白、单行、大图 |
| 生命周期 | create/first/warm、连续请求、大→小→大、重建、失败重试 |
| ISA | AVX2 对齐；scalar/SSE2 回归；ARM64 NEON；其他平台 smoke |
| 接口 | C full OCR/REC-only/generic session、HTTP、Java/Android、现代/Legacy WASM |

基线和候选共用同一 prepared assets 字节，不混入模型重新转换、版本、decoder 或字体变化。

### 14.2 计时与内存

- 同机器/CPU 配额、独立进程、串行运行两组，不同时争抢资源。
- 建议 warmup=3、iterations=10，至少 5 个交替 AB/BA 配对轮次。
- 保存原始样本，轮内计算 candidate/baseline 后汇总；M0 扩展现有脚本，不能只比无配对的两组总体中位数。
- cold/create/first 单列，不计入 warm latency，也不隐藏。
- 100 图保存每图耗时与结果；全数据集耗时和每图分位数分开。
- `evaluate_ocr_dataset.py` 是正确性工具，不是持久 handle 吞吐基准。缺少连续数据集 driver 时由 M0 增加内部 harness。
- peak RSS/WS 用全新进程测，不能在同进程读取 baseline 留下的历史峰值当 candidate。
- 记录 after-create/first/warm/large/small 容量；周期采样可能漏峰值，能取 OS peak counter 时优先使用。
- workspace 是确定性内部量；RSS 还受 allocator、触页、栈、运行库和文件映射影响。
- 不把 capacity 总和命名为 RSS，不仅凭 close 后 RSS 未立即下降认定泄漏。

### 14.3 正确性与晋级

只改变存储、不改算术时，同 ISA 逐图 text、box、顺序、CLS 元数据与 score 必须一致。
M3 首版保持算术次序；若某 kernel 改累计顺序，独立声明并提前定义 tensor/score 误差界，不能失败后放宽。
CER/Exact Line Rate 是附加指标，不替代逐字段比较；相同 CER 可能来自不同错误。

安全、ABI、正确性、内存限额和确定性是硬门禁。
M1 workspace 不大于旧布局；其他阶段验证各自实际收益。
可把代表 workload workspace 降低 10% 或稳定端到端延迟降低 3% 作为筛选目标，不作收益承诺。
小于噪声的差异不宣称收益，必要时加配对次数/置信区间。
性能 CI 先 informational，不设置全平台统一 1% 耗时门禁。
负优化模型/shape 缩小 eligibility 或保持 OFF，不能用另一模型收益掩盖回退。
默认开启前要求独立性能机复核、跨平台 CI、完整报告及 fallback 范围。

## 15. 本地具体操作

以下由开发者在仓库根目录执行。文档交付本身不执行构建、安装依赖、提交或推送。
需要已有 CMake、工具链和转换/测试依赖；先检查，不强制安装新的工具链。
Ninja/MSVC 需在已有 Developer PowerShell 中运行，确认 `cl` 可见。
不要硬编码某代 Visual Studio generator；两份构建必须采用相同 generator/compiler。

### 15.1 编译基线（修改 Runtime 之前）

```powershell
git status --short
git rev-parse HEAD
cmake --version
python --version

$reportRoot = "build-local-data/runtime-memory"
New-Item -ItemType Directory -Force $reportRoot | Out-Null

cmake -S . -B build-memory-base `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_TESTING=ON `
  -DLW_BUILD_HTTP_DEMO=OFF `
  -DLW_BUILD_CSHARP_DEMOS=OFF `
  -DLW_EXPERIMENTAL_PREPARED_EXECUTION=OFF `
  -DLW_EXPERIMENTAL_AVX2_FMA_DISPATCH=OFF `
  -DLW_REC_RESIDENT_WIDTHS=OFF
if ($LASTEXITCODE -ne 0) { throw "Baseline configure failed" }
cmake --build build-memory-base --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "Baseline build failed" }
ctest --test-dir build-memory-base -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "Baseline tests failed" }
```

现有实验开关关闭，发行版本来启用的其他优化不变。
记录完整 CMakeCache 和编译参数，不能把上述列表当成完整身份。
多配置 generator 用 `--config Release`，单配置用 `CMAKE_BUILD_TYPE`。
基线完成后不再在该 build 目录重编修改后的源码。

### 15.2 实现 M1 后编译候选

新开关须先按第 13 节实现；否则不能运行后误认为已启用优化。

```powershell
cmake -S . -B build-memory-candidate `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_TESTING=ON `
  -DLW_BUILD_HTTP_DEMO=OFF `
  -DLW_BUILD_CSHARP_DEMOS=OFF `
  -DLW_EXPERIMENTAL_PREPARED_EXECUTION=OFF `
  -DLW_EXPERIMENTAL_AVX2_FMA_DISPATCH=OFF `
  -DLW_REC_RESIDENT_WIDTHS=OFF `
  -DLW_EXPERIMENTAL_WORKSPACE_PLANNER_V2=ON
if ($LASTEXITCODE -ne 0) { throw "Candidate configure failed" }
cmake --build build-memory-candidate --config Release --parallel
if ($LASTEXITCODE -ne 0) { throw "Candidate build failed" }
ctest --test-dir build-memory-candidate -C Release -N
ctest --test-dir build-memory-candidate -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "Candidate tests failed" }
```

后续各阶段使用独立 candidate 目录，只打开该轮实验开关。
确认 CTest 预期测试已注册，不能把正则没有匹配视为通过。
关键现有测试：`session_planner`、`c_abi_contract`、`c_abi_prefix_compatibility`、`abi_layout_contract`、`rec_graph_reference`、`full_ocr_pipeline_reference`、`full_ocr_golden_corpus`。

### 15.3 现有脚本：Tiny 单图配对 A/B

```powershell
function Find-ReleaseExe([string]$BuildDir, [string]$Name) {
    foreach ($relative in @("Release/$Name.exe", "$Name.exe")) {
        $candidatePath = Join-Path $BuildDir $relative
        if (Test-Path -LiteralPath $candidatePath) {
            return (Resolve-Path -LiteralPath $candidatePath).Path
        }
    }
    throw "Missing executable: $BuildDir / $Name"
}

$baseDriver = Find-ReleaseExe "build-memory-base" "full-ocr-intra-benchmark"
$candidateDriver = Find-ReleaseExe "build-memory-candidate" "full-ocr-intra-benchmark"

foreach ($workers in @(1, 4)) {
    python -X utf8 tools/compare_ocr_dispatch_profiles.py `
      --baseline-driver $baseDriver --candidate-driver $candidateDriver `
      --det build-memory-base/models/det.lwm `
      --cls build-memory-base/models/cls.lwm `
      --rec build-memory-base/models/rec.lwm `
      --dictionary models/ppocrv6-tiny/ppocr_keys.txt `
      --image build-memory-base/models/sample.ppm `
      --warmup 3 --iterations 10 --repeats 5 `
      --workers $workers --target-width 960 --det-threads 1 `
      --baseline-label v1-baseline --candidate-label workspace-v2 `
      --json-output "$reportRoot/tiny-w$workers.json" `
      --markdown-output "$reportRoot/tiny-w$workers.md"
    if ($LASTEXITCODE -ne 0) { throw "Paired benchmark failed" }
}
```

固定 DET threads=1 先隔离 REC worker 变化，再补实际部署 DET threading 策略，两边一致。
该脚本当前主要验证 line count/checksum，不替代 score/box 回归与内存分解。
Small/Medium 使用现有 validation/model-pack 流程生成对应 LWM，替换资产与字典路径，不能误用 Tiny LWM。
M0 应产出每种模型实际资产路径及 SHA 清单，不虚构当前通用目录。

### 15.4 自有数据集

只生成一次，两组共用；已有符合条件的数据集直接复用，不覆盖。

```powershell
python -X utf8 tools/generate_ocr_dataset.py `
  --output build-local-data/runtime-memory-dataset `
  --count 100 --seed 20260907 --format png
if ($LASTEXITCODE -ne 0) { throw "Dataset generation failed" }

$baseOcr = Find-ReleaseExe "build-memory-base" "lw-ocr-ppm"
$candidateOcr = Find-ReleaseExe "build-memory-candidate" "lw-ocr-ppm"

foreach ($entry in @(@("baseline", $baseOcr), @("candidate", $candidateOcr))) {
    python -X utf8 tools/evaluate_ocr_dataset.py `
      --dataset build-local-data/runtime-memory-dataset `
      --driver $entry[1] `
      --detector build-memory-base/models/det.lwm `
      --classifier build-memory-base/models/cls.lwm `
      --recognizer build-memory-base/models/rec.lwm `
      --dictionary models/ppocrv6-tiny/ppocr_keys.txt `
      --model-name tiny --rec-max-width 960 `
      --output "$reportRoot/$($entry[0])-accuracy.json"
    if ($LASTEXITCODE -ne 0) { throw "Dataset evaluation failed" }
}

python -X utf8 tools/check_ocr_dataset_regression.py `
  --baseline "$reportRoot/baseline-accuracy.json" `
  --candidate "$reportRoot/candidate-accuracy.json" `
  --max-f1-drop 0 --max-exact-drop 0 --max-cer-increase 0 `
  --output "$reportRoot/accuracy-gate.json"
if ($LASTEXITCODE -ne 0) { throw "Accuracy regression" }
```

记录实际字体、生成器版本、manifest/image SHA；相同 seed 不能消除跨机器字体差异。
以上是已有 aggregate gate，还必须运行 M0 增加的逐字段 comparator。
图片、临时模型、原始报告放已忽略的 `build-*` / `build-local-data`，不进 Git。
可提交生成器、测试、schema、精简结论与复现说明。

## 16. CI 接入

1. 保留默认构建，增加 planner v2 ON 配置，不能让旧路径失去覆盖。
2. GCC/Clang/MSVC 严格告警不变，关注 32 位溢出、EOF 换行、非目标 ISA unused helper。
3. x64 performance 共用 prepared assets，先同源码 OFF/ON，再固定版本与候选对比。
4. Native ARM64 做正确性与性能；LoongArch 模拟器仅功能验证，不输出硬件性能结论。
5. 现代 WASM SIMD 和 Legacy scalar 都跑文本/score/内存回归，保留有限预热后收敛原则。
6. JNI、Android、HTTP staged package 验证实际加载和调用；Android 实机性能单列。
7. ASan/UBSan 检查 alias、scratch、crop 与释放顺序；支持时补多 worker 线程检查。
8. 失败也上传诊断，缺关键报告/样本不能当成功。

不增加本地 SDK 安装要求，也不扩展为签名、发布或工作流平台迁移。

## 12.5 Conv1x1-only FMA 实验结果

为避免 terminal MatMul FMA 的结果影响 Conv1x1 判断，CMake 现在提供两个独立的实验开关：

```text
LW_EXPERIMENTAL_AVX2_FMA_CONV1X1_DISPATCH
LW_EXPERIMENTAL_AVX2_FMA_MATMUL_DISPATCH
```

旧的 `LW_EXPERIMENTAL_AVX2_FMA_DISPATCH` 仍作为聚合兼容开关保留，但三个开关默认均为关闭。生产默认 dispatch 没有改变。

在本机 Windows x64 上使用项目生成的 100 图 Tiny PPM 集合、REC 宽度 960、1 次预热、3 轮配对测量，Conv1x1-only FMA 的结果为：

| workers | 默认中位数 | Conv1x1-only FMA 中位数 | 延迟变化 | 峰值工作集变化 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 224.692 ms/图 | 217.902 ms/图 | -3.02% | +0.03 MiB |
| 4 | 172.307 ms/图 | 169.578 ms/图 | -1.58% | -0.11 MiB |

两种构建的输出 checksum 均为 `ea94c481ced30cbe`，识别行数均为 620。该结果只代表一个 Windows x64 主机和 Tiny 数据集；4-worker 未达到现有完整 OCR 的 2% 推进门槛，因此不能据此默认启用。Small 100 图单轮约快 1.38%，Medium 10 图三轮中位数约快 1.82%（4-worker 约快 1.00%），但仍缺少完整 Medium/Small 语料和 4-worker CI 复核。
## 12.6 REC shape telemetry contract

`full_ocr_profile_driver` 现在为每个实际执行的 REC `Conv`/`ConvTranspose` 节点输出：

- `input`、`weights`、`output` 四维 tensor shape；
- `group`；
- `kernel`、`strides`、`dilations`、`pads` 参数。

非卷积节点不会伪造这些字段。`tests/test_full_ocr_profile.py` 已验证字段存在性、rank、正维度、参数长度和至少一个卷积节点覆盖；现有 `det_convolution_nodes` 同样保留 shape contract。该 telemetry 只用于 profile 和 shape 选择，不进入公共 C ABI，也不改变默认 dispatch。

本机 `build-memory-base` 的完整 CTest 已通过 **56/56**，包括 `full_ocr_operator_profile`、REC/DET/完整 OCR reference、Golden、workspace、SIMD benchmark smoke 和 staged package。
## 12.7 DET shape profile 基线

新增 `tools/collect_det_shape_profile.py`，包装 `full-ocr-profile-driver`，按 worker/REC width 聚合真实 DET `Conv` 与 `ConvTranspose` 的输入、权重、输出、group、kernel、stride、dilation、padding、调用次数和耗时。`tests/test_collect_det_shape_profile.py` 覆盖 Conv/ConvTranspose 分组、几何 key 和 Markdown 输出，并已接入 `model_analysis` CTest。

Tiny 的 960 profile（同一 Windows x64，1/4 worker，DET threads=1）显示，当前 DET 主要耗时集中在 `32->16, h256, w256, 3x3, stride2`、`64->16, h128, w128, 3x3` 和 `64->64, 5x5`；`ConvTranspose 16->16, h128, w128, 2x2, stride2` 是独立的真实节点。后续 regular 3x3/ConvTranspose 优化必须从该工具输出的真实 shape 开始，并用完整 OCR checksum/box/score/内存门禁复核。

当前工具只建立基线，不改变 dispatch；生成的 JSON/Markdown 留在被 Git 忽略的 `build-local-data`。
## 12.8 DET ConvTranspose 直接基线

新增 `tests/convtranspose_benchmark_driver.c` 和 `tests/test_convtranspose_benchmark.py`，并接入 `convtranspose_benchmark_smoke`。默认测试覆盖 DET profile 的真实形状：输入 `1x16x128x128`、权重 `16x16x2x2`、输出 `1x16x256x256`；驱动按运行时检测选择 AVX2/SSE2/NEON，非支持平台回到 Scalar，且始终与 Scalar reference 做误差和 checksum 校验。

Windows x64 Release、AVX2、本地 5 次迭代的直接算子结果为：

| backend | scalar | SIMD | speedup | max abs error | checksum |
|---|---:|---:|---:|---:|---|
| AVX2 | 4.706 ms | 1.278 ms | 3.682x | 0 | `0xaf53e520663f7755` |

该数字是合成输入上的单算子证据，不代表完整 OCR 延迟，也不表示已经需要改默认 dispatch。下一步只有在 output-channel blocking 候选完成同一驱动的 parity、单算子 A/B、Tiny/Small/Medium 完整 OCR 和 Peak WS 门禁后，才评估是否合入。

## 12.9 DET regular 3x3 stride-1 直接基线

新增 `tests/conv3x3_benchmark_driver.c` 和 `tests/test_conv3x3_benchmark.py`，并接入 `conv3x3_benchmark_smoke`。默认覆盖 profile 中的 `Conv 64->16, h128, w128, k3x3, stride1`，输入和输出均为 `1x*.128x128`；驱动按运行时检测选择 AVX2/SSE2/NEON，始终与 Scalar reference 做误差和 checksum 校验。

Windows x64 Release、AVX2、本地 5 次迭代的直接算子结果为：

| backend | scalar | SIMD | speedup | max abs error | checksum |
|---|---:|---:|---:|---:|---|
| AVX2 | 19.845 ms | 6.802 ms | 2.918x | 0 | `0x03aeef64730efbfe` |

该结果确认 regular 3x3 的真实热点已有稳定的现有 SIMD 路径；后续 output-channel blocking 必须证明能超过这条基线，并通过完整 DET/Full OCR 门禁后才可考虑默认 dispatch。

## 17. 提交与交付物

按实际完成拆分，不提前合并全部阶段。当前工作树已覆盖 M0、M1、M2、M3、M5，并包含
M6 的跨宽度 prepared 常量共享；后续提交仍应按阶段拆分：

```text
test(runtime): add memory accounting and reproducible baselines
feat(runtime): add experimental offline workspace planner
test(runtime): cover workspace interval and allocation boundaries
feat(runtime): plan fused GELU physical lifetimes
feat(rec): add internal greedy execution memory policy
perf(rec): tile CTC projection scratch without changing scores
perf(ocr): reuse worker-local crop buffers
perf(runtime): share eligible prepared constants across REC widths
docs(perf): publish paired memory and latency results
```

原地执行/Concat/NHWC 只有通过准入条件才另开提交，不是必须凑齐的功能。
每阶段交付说明需要：

- 改动范围、默认状态、支持 shape/ISA、fallback。
- 源码/二进制/模型/数据身份。
- 测试命令、结果、未执行平台明确待 CI。
- 1/4 worker × 三模型 memory/latency 摘要，缺项写未测，不填推测数字。
- 文本、框、score 比较和所有数值偏差解释。
- cold/create/warm、retained/peak、workspace/进程内存分项变化。
- 回退方式、是否达到默认开启条件。

## 18. 给下一位开发者的起步清单

当前 M0 + M1 + M2 + M3、M5 的 crop 预留/streaming 候选，以及 M6 的兼容性检查和
prepared arena 引用计数已在本地 Windows x64 验证；同进程多图 benchmark 已完成三模型
首轮采样，Small/Medium 100 图 1-worker 三轮 AB/BA 也已完成，4-worker 仍待补齐。
下一位开发者从 M6 Medium 完整 100 图多轮 AB/BA、4-worker 重复、宽度切换 RSS
和跨平台 CI 开始，不要重复实现已有 planner、streaming
生命周期或跨宽度共享。

- [x] 确认 HEAD 与工作树，保留用户改动。
- [x] 增加内部 session memory report，不改公共 ABI。
- [x] 补 interval planner 测试与模型 workspace 基线。
- [x] 实现默认 OFF 的 planner v2，保留旧布局对比/回退。
- [x] 实现默认 OFF、仅 AVX2 命中的 GELU 物理生命周期计划。
- [x] 跑本地默认配置与 fusion 候选配置的完整 CTest。
- [x] 跑本地 CTC 计划的 Tiny REC 320/960 报告和完整 OCR sample；输出一致，workspace 峰值无变化。
- [x] 跑本地 CTC tiled + streaming crop 组合配置的关键 OCR/staged 回归；Tiny 输出 checksum 一致。
- [x] 跑本地 Tiny 4-worker/960 的 aggregate 与 streaming crop 初步 A/B；结果一致，streaming 路径仍需多样本 RSS 复测。
- [x] 验证 320/960 prepared constants 兼容性、共享指针和引用计数释放；报告 packed arena 字段。
- [x] 生成 Tiny/Small/Medium REC 320/960 的 workspace/packed arena 组件基线；三者 packed arena 均与宽度无关。
- [x] 完成 Tiny/Small/Medium resident smoke A/B；输出 checksum 一致，Medium 4-worker 的单次延迟回退保留为风险记录。
- [x] 完成 Tiny/Small/Medium 1/4 worker、960 的严格 AB/BA 配对基线；Tiny 4-worker 为 3 轮，其余组合为 5 轮，均保持 checksum/16 行一致。
- [x] 用同一 manifest 完成 Tiny/Small/Medium 100 图质量基线；三模型结果与现有 golden corpus 一致。
- [x] 新增同进程多图 benchmark，并完成 Tiny/Small/Medium 1/4-worker 的首轮采样；所有采样 checksum 一致。
- [x] 在 Tiny 100 图、1/4-worker 上复核 M3/M5 候选；checksum 一致且未见多图回退。
- [x] 在 Small/Medium 100 图 1/4-worker 上完成 M3/M5 复核；checksum 一致。
- [x] 扩展 benchmark 以逐图采样 RSS，并完成 Tiny 10 图 1-worker smoke；Small/Medium 100 图 1-worker 已完成，4-worker 的完整场景仍待补齐。
- [x] 增加真实 REC Conv1x1 shape profile 工具，并完成 Tiny 五档宽度报告。
- [x] 拆分 Conv1x1 与 MatMul FMA 实验开关，并完成 Tiny 100 图 1/4-worker 三轮配对验证。
- [x] 完成 Conv1x1-only FMA 的 Small 100 图单轮和 Medium 10 图三轮配对；checksum 一致，收益低于默认推广门槛，未扩大 allowlist。
- [x] 为 REC Conv/ConvTranspose profile 增加 shape/parameter telemetry，并通过完整 56 项 CTest。
- [x] 为 DET ConvTranspose 真实形状增加 Scalar/SIMD benchmark、parity smoke 和 machine-readable 输出。
- [x] 为 DET regular 3x3 stride-1 真实形状增加 Scalar/SIMD benchmark、parity smoke 和 machine-readable 输出。
- [x] 新增 `build_ocr_width_switch_manifest.py`，并完成 Tiny/Small/Medium 宽度切换序列首轮 4-worker smoke；结果写入 11.10，未将单轮结果当作默认启用依据。
- [ ] 扩展 Small/Medium 的更多宽度切换与连续请求场景。
- [ ] 在连续宽度切换和多 worker 进程中测量 M6 的实际 RSS/Peak WS，确认没有瞬时双份 arena 峰值。
- [x] 对三模型在 100 图集上增加 resident/compact 的每图耗时与 RSS 首轮采样；单轮结果仅作方向性证据。
- [x] 补齐 Medium 4-worker 的首轮 100 图采样；checksum 一致，延迟接近持平但 Peak RSS 增加约 60 MiB。
- [x] 对 Small/Medium 的 10 图子集完成 3 轮 1-worker AB/BA；checksum 一致，结果写入 build-local-data。
- [x] Small 100 图、1-worker 完成 3 个交替轮次；checksum 一致，resident 延迟和 RSS 均上升。
- [x] Medium 100 图完成 3 个交替轮次；checksum 一致，resident 延迟和 RSS 均上升。
- [x] 补齐 Small/Medium 的 4-worker 重复采样；结果写入 11.11，resident 仍保持实验开关。
- [ ] 接入 CI，等待跨平台结果，不宣称未跑平台已通过。
- [ ] 汇报 workspace、RSS 与配对耗时，再决定 M4、M6 的后续 plan-cache 扩展是否值得实施。

M0 若发现主要内存来自 crop、重复 packed arena 或宿主加载，依据报告调整 M3/M5/M6 顺序。
调整顺序不等于取消正确性门禁，也不应扩大成全 Runtime 重写。

## 18.1 P1 Streaming Crop formalization

P1 已将实验性的 worker-local crop 生命周期抽成内部 `lw_crop_buffer` 模块：

- `src/ppocr/crop_buffer.c` 负责按需增长、跨请求复用和释放；不改变公共 C ABI。
- `src/ppocr/ocr.c` 的 `LW_EXPERIMENTAL_STREAMING_CROPS` 路径只保存 buffer 对象，保持原有
  aggregate crop 路径和默认关闭状态不变。
- 默认保留上限为 4 MiB。单次请求超过上限时允许临时增长，请求完成后释放超出上限的容量；
  4 MiB 以内的 buffer 跨请求保留，以减少 allocator 抖动。
- `tests/test_crop_buffer.c` 覆盖增长/复用、超限 trim 和 `retained_limit == 0` 的不 trim 语义。

验证命令：

```text
cmake -S . -B build-crop-streaming -DLW_EXPERIMENTAL_STREAMING_CROPS=ON -DBUILD_TESTING=ON
cmake --build build-crop-streaming --config Release --parallel
ctest --test-dir build-crop-streaming -C Release -R "^(crop_buffer|full_ocr_golden_corpus)$" --output-on-failure
```

Streaming crop 仍是实验开关。在完成默认/Streaming 完整 CTest、Golden 以及 Tiny/Small/Medium
100 图 1/4 worker 的 AB/BA 多轮结果前，不将它改为默认路径，也不把单轮延迟或 RSS 结果写成
产品承诺。
## 18.2 远端长基准

100 图 AB/BA 会占用较长的本机时间，不再作为每次本地开发的必跑步骤。仓库新增
`.github/workflows/runtime-memory-benchmark.yml`，在 Windows x64 runner 上完成：

- compact 与 `LW_EXPERIMENTAL_STREAMING_CROPS=ON` 两套 Release 构建；
- Tiny、Small、Medium 的临时 LWM 模型准备；
- 固定 seed `20260907` 生成的 100 图 PPM 数据集；
- 1/4 worker、REC 960、warm-up 1、paired AB/BA 多轮；
- 每图 OCR 耗时、P95、Peak RSS、最大采样 RSS、文本 checksum。

该 workflow 默认通过 GitHub Actions 的 **Run workflow** 手动启动，也在每月 1 日 04:00 UTC 执行一次。推送代码后，
可用 GitHub CLI 触发完整矩阵：

```powershell
gh workflow run runtime-memory-benchmark.yml `
  --ref main `
  -f model=all `
  -f workers=both `
  -f paired_rounds=3 `
  -f image_count=100
```

若只想快速检查某个模型，可将 `model` 设为 `tiny`、`small` 或 `medium`，将 `workers` 设为
`1` 或 `4`。结果会同时写入 Job Summary，并上传为保留 30 天的 artifact；未下载报告前不要把
远端耗时或 RSS 解读成跨 runner 的绝对基线。
## 18.3 Quick/full matrix profile

长基准现在拆成 `prepare` 和 `model × workers` 两阶段。Prepare runner 只构建一次
Compact/Streaming、生成一次数据集并上传公共 package；随后 Tiny/Small/Medium × W1/W4
在最多 6 个 Windows x64 runner 上并行执行。每个 matrix case 内的 Compact 与 Streaming
仍在同一 runner 上交替执行，避免跨主机 CPU 差异污染 AB/BA 结果。

- `quick`：30 张图、1 个 paired round，适合开发中手动检查。
- `full`：100 张图、3 个 paired round，适合发布前或性能确认。
- 周期任务强制使用 `full`。
- matrix runner 下载 package 后重新生成 `images-runner.txt`；不能复用 Prepare runner
  生成的绝对路径列表。
- 每个子进程默认 3600 秒超时，每 60 秒输出 heartbeat；报告 schema、warm-up=1 和
  REC 960 口径保持不变。

手动触发示例：

```powershell
gh workflow run runtime-memory-benchmark.yml `
  --ref main `
  -f profile=quick `
  -f model=tiny `
  -f workers=1
```

正式确认使用 `-f profile=full -f model=all -f workers=both`。单个 matrix job 的报告会
单独上传，便于先查看 Medium/W1 等慢 case，不必等待全部 case 串行结束。
## 19. 参考链接

- [SimdPaddleOCR README（当前主线）](https://github.com/sdcb/SimdPaddleOCR)
- [SimdPaddleOCR LayoutPlanner.cs](https://github.com/sdcb/SimdPaddleOCR/blob/main/src/Sdcb.SimdPaddleOCR/OnnxSharp/LayoutPlanner.cs)
- [SimdPaddleOCR MatMul.ArgMax.cs](https://github.com/sdcb/SimdPaddleOCR/blob/main/src/Sdcb.SimdPaddleOCR/Kernels/MatMul.ArgMax.cs)
- [SimdPaddleOCR 性能记录](https://github.com/sdcb/SimdPaddleOCR/blob/main/docs/perf.md)
