# FlagTree vs Triton-Ascend:昇腾用户 API 对比文档集

本目录对比 **FlagTree** 与 **Triton-Ascend** 两个仓库在**华为昇腾(Ascend NPU)**后端上、面向**用户编程层**暴露的 API 差异。

- **Triton-Ascend**:`/workspace/triton-ascend`(Triton 3.6.0,昇腾专用 fork)
- **FlagTree**:`/workspace/FlagTree`(Triton 3.5.1,多后端统一框架,Ascend 为其一)
- 共享编译底座:AscendNPU-IR / BiShengIR(`/workspace/FlagTree/third_party/ascend/AscendNPU-IR`)

## 文件说明

| 文件 | 功能 | 适合场景 |
|---|---|---|
| [triton-ascend-user-apis.md](triton-ascend-user-apis.md) | **Triton-Ascend 用户 API 概要**。按在线文档侧边栏组织,列出全部命名空间(`tl.*` / `cann.libdevice` / `cann.extension` / `extension.buffer.language` / `kernels` / `tla`)的定位、要点与源码位置,附三层心智模型与导入模板。 | 想快速了解 Triton-Ascend 能用哪些 API |
| [flagtree-user-apis.md](flagtree-user-apis.md) | **FlagTree(昇腾)用户 API 概要**。与上一份结构对照,突出 FlagTree 的多后端定位、`flagtree_spec` 注入机制,以及相对 Triton-Ascend 的历史快照差异(无 `dot`/`conv2d` builtin、libdevice 保留精简 + rebind 形态)和 TLE/TLA 路线差异,不将这些差异归因于主动裁剪。 | 想了解 FlagTree 昇腾侧能用哪些 API |
| [user-apis-flagtree-vs-triton-ascend.md](user-apis-flagtree-vs-triton-ascend.md) | **逐 API 对照表**(核心产物)。表 1:昇腾专属扩展 API(接口名 / 适用硬件 / TA 支持 / FT 支持 / 功能,分 6 类)。表 2:算子名 / TA 源码路径 / FT 源码路径。表 3:通用 `tl.*` / `triton` / `triton.testing`(逐一列出,两仓一致)。表 4:FlagTree TLE 扩展原语 → Triton-Ascend 对应情况(相同/类似/无)。 | 查某个具体 API 两仓是否支持、代码在哪 |
| [triton-ascend-vs-flagtree.md](triton-ascend-vs-flagtree.md) | **两仓昇腾差异概要**(架构级,**以 Triton-Ascend 为基准**)。涵盖定位、命名空间、libdevice 实现策略、TLA、编译管线(lowering passes)、AscendNPU-IR 底座、各自的胶水机制。 | 想理解两仓架构与编译链路差异 |
| [feature-timeline-git.md](feature-timeline-git.md) | **特性引入时间线**(基于 git 考据)。用 pickaxe 定位每个 API 符号/文件首次出现的 commit 与日期,给出 TA 与 FlagTree TLE 各自的演进时间轴、对应特性的先后对照,并考证出「FlagTree 缺 `dot`/`conv2d` 源于 2026-05-15 同步基线早于 TA 提交」等结论,并区分具体接口首见与较早 `al`/buffer 对应能力、作者日期与提交者日期;不把时间先后直接解释为原创归属。 | 想知道某特性何时引入、两仓谁先谁后 |
| README.md | 本文件,文档集索引。 | 入口导航 |

## 阅读建议

- **只想对照某个 API** → 直接查 [user-apis-flagtree-vs-triton-ascend.md](user-apis-flagtree-vs-triton-ascend.md)。
- **想先建立整体认知** → 先读 [triton-ascend-vs-flagtree.md](triton-ascend-vs-flagtree.md)(差异概要),再按需翻两份 user-apis 概要。
- **要查 API 具体实现** → 两份 user-apis 概要和对照表的表 2 都带源码相对路径,可在编辑器点击跳转。

## 核心结论(速览)

1. **cann 层主要是整体同步,不是逐特性独立开发。** FlagTree 在 **2026-06-08** 的 `8c30adb6c` 中同步 TA,基线是 **2026-05-15** 的 `c85527413`。目录见 [FlagTree cann/](../FlagTree/third_party/ascend/language/cann/)。
2. **al/libdevice 的主要差异首先是同步版本差。** FlagTree 缺少 `al.dot`、`al.conv2d`,因为 TA 在同步基线之后才引入;libdevice 的“精简 + rebind 到 `math.*`”也是 TA 当时的原貌,不是 FlagTree 主动裁剪。TA 此后扩展为全量 libdevice;当前识别到的 FlagTree 独有 cann 扩展是 `custom_object_semantic`。
3. **显式内存层是 TLA 与 TLE 两条路线。** TA 的 `triton.language.extra.tla` 更贴近昇腾 DMA/SIMD/Cube 指令;FlagTree 的 `triton.experimental.tle` 提供跨后端 buffer/tile/流水及昇腾扩展,还包含流水调度器和分布式通信接口。相同或类似能力不代表接口、编译行为和性能等价。
4. **API 暴露不代表三代硬件全支持。** `al.fixpipe`/`al.copy`/`al.copy_from_ub_to_l1` 及 `cast` 的 fp8/fp64 存在 **950 专属限制**;还需区分 dtype 与 SIMT 模式限制。源码无架构守卫不等于所有组合已运行验证,对照表硬件列是源码分析结论,不是全组合实测认证。
5. **原语先后必须分口径。** 当前 HEAD 历史中,TA 的早期 `al`/buffer 对应 API 多数更早;TLE 的 tile 接口(2026-08-27)早于具体 TLA 接口(2026-10~11 月)。`tile_load/store` 与视图族需另算,无直接对应者不强行比较;旧分支记录不当作当前主线首见。完整逐项表与 A/C 日期见 [时间线 §3](feature-timeline-git.md#3-逐原语首见与先后对照)。
6. **共享编译体系不代表编译器版本相同。** 两仓都使用 AscendNPU-IR / BiShengIR 体系,差异主要集中在 API 快照、TLA/TLE、扩展注入和前端 lowering。FlagTree 的底座源码完整可查,TA 子模块当前未检出;具体版本及实际编译器需另行核对。

**一句话**:FlagTree 昇腾侧 = **TA 历史后端快照 + 多后端适配机制 + 独立演进的 TLE**,而不是重新逐项实现或主动删减 TA API。

完整总结见 [triton-ascend-vs-flagtree.md](triton-ascend-vs-flagtree.md) §0;历史依据见 [feature-timeline-git.md](feature-timeline-git.md)。

> 说明:硬件列中 **A2/A3** = Atlas A2 / A3(`Ascend910B`/`910_93`,LEGACY 代),**950** = Ascend 950PR&950DT(`Ascend910_95`/`Ascend950`,GEN5 / register-based)。**「仅 950」= A2/A3 调用会报错**。
