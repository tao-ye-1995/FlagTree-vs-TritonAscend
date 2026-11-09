# FlagTree vs Triton-Ascend:昇腾用户 API 对比文档集

本目录对比 **FlagTree** 与 **Triton-Ascend** 两个仓库在**华为昇腾(Ascend NPU)**后端上、面向**用户编程层**暴露的 API 差异。

- **Triton-Ascend**:`/workspace/triton-ascend`(Triton 3.6.0,昇腾专用 fork)
- **FlagTree**:`/workspace/FlagTree`(Triton 3.5.1,多后端统一框架,Ascend 为其一)
- 共享编译底座:AscendNPU-IR / BiShengIR(`/workspace/FlagTree/third_party/ascend/AscendNPU-IR`)

## 文件说明

| 文件 | 功能 | 适合场景 |
|---|---|---|
| [triton-ascend-user-apis.md](triton-ascend-user-apis.md) | **Triton-Ascend 用户 API 概要**。按在线文档侧边栏组织,列出全部命名空间(`tl.*` / `cann.libdevice` / `cann.extension` / `extension.buffer.language` / `kernels` / `tla`)的定位、要点与源码位置,附三层心智模型与导入模板。 | 想快速了解 Triton-Ascend 能用哪些 API |
| [flagtree-user-apis.md](flagtree-user-apis.md) | **FlagTree(昇腾)用户 API 概要**。与上一份结构对照,突出 FlagTree 的多后端定位、`flagtree_spec` 注入机制,以及相对 Triton-Ascend 的裁剪(无 `tla`、无 `dot`/`conv2d` builtin、libdevice 精简)。 | 想了解 FlagTree 昇腾侧能用哪些 API |
| [user-apis-flagtree-vs-triton-ascend.md](user-apis-flagtree-vs-triton-ascend.md) | **逐 API 对照表**(核心产物)。表 1:昇腾专属扩展 API(接口名 / 适用硬件 / TA 支持 / FT 支持 / 功能,分 6 类)。表 2:算子名 / TA 源码路径 / FT 源码路径。表 3:通用 `tl.*` / `triton` / `triton.testing`(逐一列出,两仓一致)。表 4:FlagTree TLE 扩展原语 → Triton-Ascend 对应情况(相同/类似/无)。 | 查某个具体 API 两仓是否支持、代码在哪 |
| [triton-ascend-vs-flagtree.md](triton-ascend-vs-flagtree.md) | **两仓昇腾差异概要**(架构级,**以 Triton-Ascend 为基准**)。涵盖定位、命名空间、libdevice 实现策略、TLA、编译管线(lowering passes)、AscendNPU-IR 底座、各自的胶水机制。 | 想理解两仓架构与编译链路差异 |
| README.md | 本文件,文档集索引。 | 入口导航 |

## 阅读建议

- **只想对照某个 API** → 直接查 [user-apis-flagtree-vs-triton-ascend.md](user-apis-flagtree-vs-triton-ascend.md)。
- **想先建立整体认知** → 先读 [triton-ascend-vs-flagtree.md](triton-ascend-vs-flagtree.md)(差异概要),再按需翻两份 user-apis 概要。
- **要查 API 具体实现** → 两份 user-apis 概要和对照表的表 2 都带源码相对路径,可在编辑器点击跳转。

## 核心结论(速览)

- **TA 独有**:整套 `tla` 显存层级范式、`al.dot` / `al.conv2d` cube builtin、全量 `libdevice`(~180)、`SyncEventSlot` / `AddressSpace` / `DataLayout` / `is_register_based_arch`。
- **FT 独有**:`al.custom_object_semantic`;**整套 TLE 扩展原语**(`triton.experimental.tle`,含 `dsa.ascend` 的流水调度器、分布式通信族;对应 TA 的 `tla`,逐项见对照表表 4)。
- **共有且一致**:绝大多数 `al.*` builtin、`extension.buffer.language.*`、`kernels.gather_2d_simd`、`do_bench_npu`、通用 `tl.*`。
- **功能可用但实现不同**:FT 部分 `libdevice` 函数 rebind 到 upstream `triton.language.math.*`,而非 cann 自实现。
- **共性**:两仓最终都经 **bishengir-compile / AscendNPU-IR** 完成 lowering;FlagTree 侧 AscendNPU-IR 为 in-tree 完整,Triton-Ascend 侧为 git submodule(当前 workspace 未 checkout)。

> 说明:硬件列中 **A2/A3** = Atlas A2 / A3(910B / 910C),**950** = Ascend 950PR&950DT(910_95 / A5 register-based)。
