# 特性引入时间线：Triton-Ascend vs FlagTree TLE

> 方法：在完整 Git 历史上用路径限定 `git log -S/-G` 找候选，再检查真实新增定义/导出、迁移前路径与 HEAD 祖先关系。**作者时间 A、提交者时间 C 都不自动等于 PR 合入日期。** 整批导入只能证明仓内可追溯首见，不能证明原创研发起点。
> 基准：TA HEAD `ae07d3d13917a53300bb74127604242f7bf02f2e`；FlagTree HEAD `394d50493a21dc99bdbfc527b9625dbbb5f327da`；均非浅克隆。
> 仓库：[triton-ascend/](../triton-ascend/) 与 [FlagTree/](../FlagTree/)。以下正文以**当前 HEAD 祖先历史中的定义与用户导出首见**为主；不在这条历史中的旧分支单列。

## 0. 先看结论与比较口径

**没有统一的“哪边所有原语都更早”。需要同时看两列：**

1. **TA 较早的相同/相似能力**：`al.*`、`extension.buffer.language.*` 和继承 upstream 的 `tl.*`。
2. **TA 后来的具体 TLA 接口**：`tla.alloc_buffer`、`tla.subview`、`tla.sync.*`、`tla.cube.mmad` 等。

按当前 HEAD 历史：
- 跨核同步、子向量核、编译提示、空间/PIPE、基础 buffer、切片/转换：**TA 的早期对应 API 更早**；TLE 的 DSA/Ascend 导出集中于 **2026-07-01**。
- TLE 的 tile 分配/拷贝/子视图、流水 flag/barrier、cube launch：**2026-08-27**，早于具体 TLA 对应接口的 **2026-10-26～11-03**。但早期 buffer 分配/子视图、受限搬运或高层矩阵乘早已存在，不能因此说 TA 在 8 月前没有这些能力。
- `SyncSpec` 对应的 TA `SyncEventSlot`：**TA 更早**，但描述对象的粒度不同。
- TLE `tile_load/store`：**2026-10-28**，早于 TA 命名 DMA 族的 **2026-10-31**，仅约三天，不能套用“tile 都早两个月”。
- TLE 视图族：**2026-10-28**；TA `tla.view` 在 **10-26** 更早，而 `tla.subview` 到 **11-03** 才出现。应逐构造器说明，不能整族判同一先后。
- 专门 GM→L1 gather、sort32/mrgsort/unpack_sort、int4 转换、workspace、独立 cube_wait：没有全部一一对应的 TA API；不能拿通用算子日期替代专门原语日期。

下面表中显示日期采用 **C 的日历日期**，精确 A/C 与时区见 §5。对有跨时区的记录不以字符串大小判断先后。先后只说明当前可追溯历史，不说明原创归属、借鉴关系、性能或全硬件支持。

## 1. 两仓关键时间轴

| 事件 | C 日期 | commit | 核查说明 |
|---|---|---|---|
| TA fork 整批导入 cann | 2025-12-29 | `313dccecf` | 含跨核同步三项、compile_hint/multibuffer、切片/元素、一般 sort、GM→UB gather；不代表原始研发起点 |
| TLE 初始设计 | 2026-01-05 | `70bd76c8d` | 早期 GPU storage scope/alloc/copy/pipeline；不是后来 DSA 或 core_mode scope 的首见 |
| FlagTree 当前路线同步 TA cann/buffer | 2026-06-08 | `8c30adb6c` | 明确同步基线 TA `c85527413`（2026-05-15）；**不是 TLE 包装导出日期** |
| FlagTree 当前路线 DSA/Ascend 导出 | 2026-07-01 | `b34efe490` | 基础 DSA、Ascend 常量与 cann re-export |
| FlagTree tile 九项与 core_mode scope | 2026-08-27 | `f56cd1bdb` | 定义与用户导出同批进入当前历史 |
| FlagTree gather 两项与 unpack_sort | 2026-09-04 | `bd382d749` | 专门 custom-op class、导出与 C++ 实现 |
| FlagTree sort32/mrgsort | 2026-09-17 | `895e4ea41` | 不能归并到 9 月 4 日 |
| FlagTree int4→fp16 | 2026-09-23 | `2b5b81320` | 专用转换，不是通用 cast |
| FlagTree SyncSpec/workspace/Ascend pipe 调度器 | 2026-10-10 | `bb9b00998` | 不等于基础 DSA 的首见 |
| TA 首批 TLA | 2026-10-26 | `b6dda4fa1` | alloc/alloc_buffer/copy/view、scope、部分 SIMD；作者时间为 10-24，导入不证明原创日期 |
| FlagTree TensorView 与 tile_load/store | 2026-10-28 | `aac54bb19` | 注意 load/tensor_view 的实际命名空间 |
| TA cube.mmad | 2026-10-30 | `690924d08` | scope 迁移不是 scope 首见 |
| TA set_flag/wait_flag/barrier | 2026-10-30 | `f810ca587` | 段内流水同步，与跨核 sync_block 分开 |
| TA DMA 族与更多 SIMD | 2026-10-31 | `e76cc20ec` | 11 个命名 DMA 接口；vsub/vmax/vmin 与 sync.PIPE |
| TA tla.subview | 2026-11-03 | `a81c089dc` | 当前 main 历史首见；不能用首批 view 的日期替代 |

## 2. 源码与导出证据索引

| 简称 | 实际路径 | 定义/导出证据 |
|---|---|---|
| TA al | [cann/extension/](../triton-ascend/third_party/ascend/language/cann/extension/) | core.py、aux_ops.py、vec_ops.py、mem_ops.py、custom_op.py、scope.py；检查同批 __init__.py |
| TA buffer | [buffer/language/core.py](../triton-ascend/python/triton/extension/buffer/language/core.py) | alloc、to_tensor、to_buffer、subview 与成员绑定/导出 |
| TA TLA | [language/tla/](../triton-ascend/third_party/ascend/language/tla/) | alloc/copy/view/vf→scope/simd/cube/sync/dma；检查 __init__.py |
| TA upstream | [language/core.py](../triton-ascend/python/triton/language/core.py)、[math.py](../triton-ascend/python/triton/language/math.py) | 还追踪历史 python/triton/core.py、language.py；继承 upstream 不算昇腾原创 |
| FT DSA | [dsa/core.py](../FlagTree/python/triton/experimental/tle/language/dsa/core.py)、[dsa/__init__.py](../FlagTree/python/triton/experimental/tle/language/dsa/__init__.py) | 基础函数与 tile 定义、用户导出分开核实 |
| FT Ascend | [ascend/core.py](../FlagTree/python/triton/experimental/tle/language/dsa/ascend/core.py)、[ascend/__init__.py](../FlagTree/python/triton/experimental/tle/language/dsa/ascend/__init__.py) | cann import/re-export、space/PIPE alias、SyncSpec |
| FT custom ops | [registry.py](../FlagTree/python/triton/experimental/tle/language/dsa/ascend/custom_ops/registry.py)、[custom_ops/](../FlagTree/python/triton/experimental/tle/language/dsa/ascend/custom_ops/) | class、__init__.py 导出与相应 C++ entrypoint |
| FT 通用 TLE | [language/core.py](../FlagTree/python/triton/experimental/tle/language/core.py)、[tile.py](../FlagTree/python/triton/experimental/tle/language/tile.py)、[tle/__init__.py](../FlagTree/python/triton/experimental/tle/__init__.py) | 检查 language 导出、顶层导出及 backends 动态分发 |

**两处命名空间纠错**：当前是 `tle.language.load` 和 `tle.language.tensor_view`，不是 `tle.load`、`tle.tensor_view`。顶层显式导出三个 make_* 与 tile_load/store；`__getattr__` 的 dsa/pipe 特判及 backends op 表不补出 load/tensor_view。

## 3. 逐原语首见与先后对照

覆盖 [API 对照表 4](user-apis-flagtree-vs-triton-ascend.md#表-4--flagtree-tle-扩展原语--triton-ascend-对应情况) 原有全部 33 个“相同/类似”关系行（合并行中的符号全部列名）；workspace 与 int4 专项核查后改为无直接对应。下表编号保持原始覆盖范围，不把后续分类纠错误当漏查。

### 3.1 跨核同步、编译提示与空间

| # | TLE 具体接口 | FT 首见：C 日期 / commit | TA 较早对应：C 日期 / commit | 判定与边界 |
|---|---|---|---|---|
| 1 | dsa.ascend.sub_vec_id | 07-01 `b34efe490` | al.sub_vec_id：01-12 `fa46fac7b` | **TA 早**；FT TLE 为 cann re-export |
| 2 | dsa.ascend.sub_vec_num | 07-01 `b34efe490` | al.sub_vec_num：01-21 `04caff591` | **TA 早**；不能把它与 id 都回填 fork 日期 |
| 3 | dsa.ascend.sync_block_set | 07-01 `b34efe490` | 2025-12-29 `313dccecf` | **TA 早**；aux_ops 已定义并导出；01-13 `2fdb3cab7` 是迁移及 pipe 参数增强 |
| 4 | dsa.ascend.sync_block_wait | 07-01 `b34efe490` | 2025-12-29 `313dccecf` | **TA 早**；同上，不以 core.py 首见替代能力首见 |
| 5 | dsa.ascend.sync_block_all | 07-01 `b34efe490` | 2025-12-29 `313dccecf` | **TA 早**；01-15 `3393b4416` 只是从 aux_ops 迁移 |
| 6 | dsa.ascend.compile_hint | 07-01 `b34efe490` | 2025-12-29 `313dccecf` | **TA 早**；FT TLE re-export 与 06-08 cann 导入分开 |
| 7 | dsa.ascend.multibuffer | 07-01 `b34efe490` | 2025-12-29 `313dccecf` | **TA 早**；同上 |
| 8 | dsa.ascend.UB/L1/L0A/L0B/L0C | 五项均 07-01 `b34efe490` | ascend_address_space 五项：01-09 `97b5ed7c7` | **TA 早**；TLA 字符串 space：10-26 `b6dda4fa1`，为 ub/l1/l0a/l0b/l0c，不是单一 l0 |
| 9 | dsa.ascend.PIPE | 07-01 `b34efe490` | al.PIPE：01-16 `395ca4c6d` | **TA 早**；tla.sync.PIPE：10-31 `e76cc20ec` 较晚 |
| 10 | dsa.ascend.SyncSpec | 10-10 `bb9b00998` | al.SyncEventSlot：06-24 `e707856bc` | **TA 相似描述对象早**；custom-op 事件槽与流水 SyncSpec 不等价 |

> 除显式 2025 日期外，本节及后续表中的 MM-DD 均为 **2026 年**。

### 3.2 基础 buffer、切片、算术

| # | TLE 具体接口 | FT 首见：C 日期 / commit | TA 较早能力 | TA 具体 TLA 接口 / 判定 |
|---|---|---|---|---|
| 11 | dsa.alloc | 07-01 `b34efe490` | buffer.alloc：01-09 `97b5ed7c7` | tla.alloc/alloc_buffer：10-26 `b6dda4fa1`；**早期 TA buffer 早，FT DSA 早于 TLA** |
| 12 | dsa.copy | 07-01 `b34efe490` | al.copy_from_ub_to_l1：01-13 `8d06e51ce`；al.copy：02-12 `b35b91bbe` | tla.copy：10-26 `b6dda4fa1`；**早期受限搬运 TA 早，FT DSA 早于 TLA copy**；不是同一搬运矩阵 |
| 13 | dsa.subview | 07-01 `b34efe490` | buffer.subview：01-13 `1d542d4a3` | tla.subview：11-03 `a81c089dc`；**TA 早期子视图早，FT 早于具体 TLA 子视图** |
| 14 | dsa.extract_slice / insert_slice | 两项均 07-01 `b34efe490` | al 两项：2025-12-29 `313dccecf` | **TA 早** |
| 15 | dsa.extract_element | 07-01 `b34efe490` | al.get_element：2025-12-29 `313dccecf` | **TA 对应取元素早**；名字不同 |
| 16 | dsa.to_tensor / to_buffer | 两项均 07-01 `b34efe490` | buffer.to_tensor：01-10 `d8c03adf6`；to_buffer：01-12 `9bdc70c2f` | **TA 早**；两转换不能混为一个首见日 |
| 17 | dsa.add/sub/mul/div/max/min | 六项均 07-01 `b34efe490` | 继承 upstream 算术/maximum/minimum：U1；fdiv：U2；显式 add/sub/mul：U4 | tla vadd/vmul/vdiv：10-26 `b6dda4fa1`；vsub/vmax/vmin：10-31 `e76cc20ec`；**upstream 能力更早，FT DSA 早于具体 TLA 指令** |

算术口径：没有 `tl.div`；高层除法看 `/` 或 `tl.fdiv` 的具体语义。逐元素 max/min 对应 `tl.maximum/minimum`，不是归约 `tl.max/min`。upstream 历史是两仓共同继承的通用能力，不能归为 TA 昇腾侧原创。

### 3.3 tile、scope 与流水

| # | TLE 具体接口 | FT 首见：C 日期 / commit | TA 对应日期 / commit | 判定与边界 |
|---|---|---|---|---|
| 18 | dsa.tile_alloc | 08-27 `f56cd1bdb` | tla.alloc_buffer：10-26 `b6dda4fa1`；早期 buffer.alloc：01-09 `97b5ed7c7` | **FT 具体 tile 接口早于 TLA；TA 早期分配能力更早** |
| 19 | dsa.tile_copy | 08-27 `f56cd1bdb` | tla.copy：10-26 `b6dda4fa1`；DMA：10-31 `e76cc20ec`；受限 al 搬运见 #12 | **FT 早于具体 TLA copy/DMA；不代表全部搬运路线 TA 更晚** |
| 20 | dsa.tile_subview | 08-27 `f56cd1bdb` | tla.subview：11-03 `a81c089dc`；buffer.subview：01-13 `1d542d4a3` | **FT 早于 TLA，TA 早期子视图更早** |
| 21 | dsa.tile_set_flag / tile_wait_flag | 两项均 08-27 `f56cd1bdb` | tla.sync.set_flag/wait_flag：10-30 `f810ca587` | **FT 具体流水 flag 接口早**；不能用跨核 sync_block 日期代替段内同步 |
| 22 | dsa.tile_pipe_barrier | 08-27 `f56cd1bdb` | tla.sync.barrier：10-30 `f810ca587` | **FT 早**；pipe barrier 与跨核 all 不同 |
| 23 | dsa.tile_cube_launch / tile_cube_wait | 两项均 08-27 `f56cd1bdb` | tla.cube.mmad：10-30 `690924d08`；scope.cube：10-26 `b6dda4fa1` | **FT launch 早于具体 TLA Cube 接口**；TA 无独立等价 cube_wait；早期 tl.dot（U1）仅高层能力，不能等同启动/等待 |
| 24 | dsa.tile_gm_offset | 08-27 `f56cd1bdb` | tla.view offsets：10-26 `b6dda4fa1`；早期指针算术 U1 / block_ptr U3 | **FT 具体 offset 原语早于 TLA view**；旧指针能力仅概念对应 |
| 25 | dsa.pipeline / parallel / hint | 三项均 07-01 `b34efe490` | al.parallel/compile_hint/multibuffer：2025-12-29 `313dccecf` | **TA 相似标注早**；非同名 pipeline，不证明自动调度能力等价；DSA marker 不是直接 re-export |
| 26 | dsa.workspace | 10-10 `bb9b00998` | **无独立对应 TA DSL API** | 可在 host 分配后传 kernel，但不能给组合实现编造接口首见日期 |
| 27 | tle.scope(core_mode=) | 08-27 `f56cd1bdb` | al.scope：01-07 `5e693bb3f`；tla.scope.simd/cube：10-26 `b6dda4fa1` | **TA al 核作用域更早，FT core_mode scope 早于 TLA scope**；01-05 TLE GPU storage scope 不算当前语义首见 |

### 3.4 专门算子、通用加载与视图

| # | TLE 具体接口 | FT 首见：C 日期 / commit | TA 对应日期 / commit | 判定与边界 |
|---|---|---|---|---|
| 28 | custom_ops.gather_gm_to_ub / gather_gm_to_l1 | 两项均 09-04 `bd382d749` | al.gather_out_to_ub：2025-12-29 `313dccecf` | **TA GM→UB gather 功能早**；L1 专门接口无直接对应，需组合；FT 是 class+C++，不是简单 re-export |
| 29 | custom_ops.sort32 / mrgsort / unpack_sort | sort32/mrgsort：09-17 `895e4ea41`；unpack_sort：09-04 `bd382d749` | 一般 al.sort：2025-12-29 `313dccecf` | **一般排序 TA 早**；无三个专门指令的一一对应 TA API，不能由一般 sort 推导指令首见 |
| 30 | custom_ops.cast_int4_to_fp16 | 09-23 `2b5b81320` | **无直接对应** | 通用 al.cast 不提供这个专门 packed int4 转换；撤销此前类似 cast 的配对 |
| 31 | tle.language.load（非顶层 tle.load） | 01-05 `2b19cd5e4` | tl.load：U1（2021） | **共同 upstream 高层加载更早**；TA 无等价 Hopper 异步 SMEM 语义，不能判该异步能力 TA 更早 |
| 32 | tle.tile_load / tile_store | 两项均 10-28 `aac54bb19` | tla.dma.load_gm_to_ub/store_ub_to_gm 等：10-31 `e76cc20ec` | **FT 比具体命名 DMA 接口早约三天**；不是 08-27 DSA tile_copy 首见日；高层 tl.load/store 更早但粒度不同 |
| 33 | tle.language.tensor_view；tle.make_partition_view / make_strided_view / make_gather_scatter_view | 四项均 10-28 `aac54bb19` | tla.view：10-26 `b6dda4fa1`；tla.subview：11-03 `a81c089dc`；buffer.subview/U3 更早 | **TA view 早两天，FT 早于 tla.subview**；partition/gather-scatter 没有完全同构的 TA 构造器，不能整族强行判早晚 |

TA 命名 DMA 族同次新增 11 项：`load_gm_to_ub`、`store_ub_to_gm`、`load_gm_to_l1`、`copy_ub_to_l1`、`copy_l1_to_ub`、`copy_l1_to_bt`、`copy_l1_to_fb`、`copy_l1_to_l0`、`store_l0c_to_gm`、`copy_l0c_to_ub`、`copy_l0c_to_l1`。存在接口不等于所有硬件/dtype/layout 组合都支持。

## 4. 旧分支证据：为什么不直接拿最早作者日期排序

以下提交已核实真实定义/导出，但**不是当前 HEAD 祖先**。它们说明本地历史有更早开发线，不能悄悄回填成当前主线首见。

| 仓库 / commit | A | C | 记录与限制 |
|---|---|---|---|
| FT `f12689016` | 2025-10-23T09:23:53+00:00 | 同 A | core_ext 已有 sync_block 三项、compile_hint/multibuffer、切片/get_element；**早于 TA 当前 fork 导入记录**，但非 TLE、非当前 FT HEAD；不足以判原创起点 |
| FT `4a05c2a15` | 2025-10-23T09:23:53+00:00 | 2025-12-15T20:01:59+08:00 | 同套底层代码重放；作者时间保留，非另一首创 |
| FT `db585cdd0` | 2026-02-11T18:44:15+08:00 | 同 A | 旧路线 cann/buffer 导入；不同于当前 06-08 导入 |
| FT `837fe3ee1` | 2026-03-16T14:29:08+08:00 | 同 A | 旧 DSA 基础定义/导出、五项空间 alias |
| FT `db7e7ffc3` | 2026-05-20T16:38:52+08:00 | 同 A | 旧 TLE sub_vec/sync/PIPE re-export；沿用旧 DSA 分支 |
| FT `951af6322` | 2026-06-11T15:11:31+08:00 | 同 A | 旧 TLE compile_hint/multibuffer re-export；标题含 DOC，实际 diff 有新增导出 |
| FT `acdcfa173` | 2026-06-18T02:43:41+00:00 | 2026-07-07T09:23:42+00:00 | tile 七项仅定义，尚未在 DSA __init__ 导出 |
| FT `7f2f48438` | 2026-06-25T22:48:11+08:00 | 2026-07-07T09:32:08+00:00 | 上述七项首次导出；新增 cube_launch/wait 并导出 |
| FT `3affbd1c9` | 2026-07-01T17:28:53+08:00 | 2026-07-07T09:42:29+00:00 | 旧 core_mode scope 定义；不是 GPU storage scope |
| TA `9e6fdfcfc` | 2026-10-24T17:17:51+08:00 | 同 A | 另一路线首批 TLA；当前历史用 b6dda4fa1 |
| TA `0988a7ed2` | 2026-11-02T19:58:23+08:00 | 同 A | 另一路线 tla.subview；当前历史用 11-03 的 a81c089dc |

因此，“TA 对应早期 API 更早”必须附带**当前 HEAD 历史口径**。若把本地所有 refs 都纳入，FT 在 2025-10 已有部分旧底层记录，早于 TA 在 2025-12 的整批导入首见；这恰好说明仓内导入时间不能当原创日期。

迁移排除：TA sync 三项原在 aux_ops；scope 原在 TLA vf.py；FT buffer 当前路线由 backend/spec 迁往 python/triton/spec/ascend（`85aeabbf5`，2026-06-23），均不算能力新增。subview 签名增强也不是首见。

## 5. 正文提交字典：精确 A/C

以下全部为当前 HEAD 祖先。为减少重复，正文只写短 hash；短 hash 在相应仓库可唯一解析。`同 A` 指完整 ISO 时间完全相同。

### 5.1 TA

| commit | A | C | 本次使用的证据 |
|---|---|---|---|
| `313dccecf` | 2025-12-29T10:33:06+08:00 | 同 A | cann 整批导入；sync 三项定义/导出、hint/multibuffer/parallel、切片/元素/sort/gather |
| `5e693bb3f` | 2026-01-07T10:07:57+08:00 | 同 A | al.scope |
| `97b5ed7c7` | 2026-01-09T17:45:49+08:00 | 同 A | buffer.alloc/address_space；ascend_address_space 五空间 |
| `d8c03adf6` | 2026-01-10T16:42:28+08:00 | 同 A | buffer.to_tensor |
| `9bdc70c2f` | 2026-01-12T11:13:15+08:00 | 同 A | buffer.to_buffer |
| `fa46fac7b` | 2026-01-12T15:38:45+08:00 | 同 A | sub_vec_id |
| `8d06e51ce` | 2026-01-13T22:30:56+08:00 | 同 A | copy_from_ub_to_l1 |
| `1d542d4a3` | 2026-01-13T22:31:40+08:00 | 同 A | buffer.subview |
| `2fdb3cab7` | 2026-01-13T23:43:20+08:00 | 同 A | sync set/wait 迁移/pipe 参数扩展，非首见 |
| `3393b4416` | 2026-01-15T11:42:03+08:00 | 同 A | sync all 迁移，非首见 |
| `395ca4c6d` | 2026-01-16T10:53:18+08:00 | 同 A | al.PIPE |
| `04caff591` | 2026-01-21T09:56:34+08:00 | 同 A | sub_vec_num |
| `b35b91bbe` | 2026-02-12T14:17:44+08:00 | 同 A | al.copy，受限 UB→UB/L1 |
| `e707856bc` | 2026-06-23T18:10:11+08:00 | 2026-06-24T22:08:22+08:00 | SyncEventSlot；HEAD 祖先但非 main first-parent |
| `b6dda4fa1` | 2026-10-24T17:17:51+08:00 | 2026-10-26T21:25:32+08:00 | 首批 TLA alloc/copy/view/scope 与部分 SIMD |
| `690924d08` | 2026-10-27T21:14:17+08:00 | 2026-10-30T16:44:31+08:00 | cube.mmad；reshape；scope 文件迁移 |
| `f810ca587` | 2026-10-28T21:15:08+08:00 | 2026-10-30T16:44:31+08:00 | set_flag/wait_flag/barrier |
| `e76cc20ec` | 2026-10-30T20:44:17+08:00 | 2026-10-31T19:24:44+08:00 | DMA 11 项；vsub/vmax/vmin；sync.PIPE |
| `a81c089dc` | 2026-11-03T19:13:09+08:00 | 同 A | tla.subview 定义与导出 |

继承 upstream 的提交（**不是 TA 昇腾原创**）：

| 代号 / commit | A | C | 对应 |
|---|---|---|---|
| U1 `39f473030` | 2021-04-20T22:29:40-04:00 | 2021-07-27T12:38:49-07:00 | Python tensor 算术、maximum/minimum、load/dot；历史 core.py |
| U2 `bef76b142` | 2022-01-29T18:29:29-08:00 | 同 A | fdiv；HEAD 祖先非 first-parent |
| U3 `72b071253` | 2023-03-28T07:46:49+08:00 | 2023-03-27T16:46:49-07:00 | make_block_ptr/advance；A/C 只是时区写法差异 |
| U4 `c54f9882a` | 2024-10-05T23:09:41-07:00 | 同 A | 显式 tl.add/sub/mul；算术能力不是此时才出现 |

### 5.2 FlagTree

| commit | A | C | 本次使用的证据 |
|---|---|---|---|
| `70bd76c8d` | 2026-01-05T10:23:41+08:00 | 同 A | 初始 TLE GPU 设计，不替代后来 DSA 首见 |
| `2b19cd5e4` | 2026-01-05T17:03:35+08:00 | 同 A | language.load 定义与 language 导出，无顶层 load |
| `8c30adb6c` | 2026-06-08T20:41:25+08:00 | 同 A | 当前路线 TA cann/buffer 导入 |
| `b34efe490` | 2026-07-01T19:29:56+08:00 | 同 A | 基础 DSA 与 Ascend 包装/导出 |
| `f56cd1bdb` | 2026-08-27T19:22:25+08:00 | 同 A | tile 九项定义/导出、core_mode scope |
| `bd382d749` | 2026-09-04T15:47:28+08:00 | 同 A | gather 两项、unpack_sort：class、导出、C++ |
| `895e4ea41` | 2026-09-17T14:41:10+08:00 | 同 A | sort32/mrgsort：class、导出、C++ |
| `2b5b81320` | 2026-09-22T22:08:20-04:00 | 2026-09-23T10:08:20+08:00 | 专门 int4→fp16；A/C 是同一时刻的时区表示 |
| `bb9b00998` | 2026-10-10T19:38:50+08:00 | 同 A | workspace、SyncSpec 定义/导出；Ascend pipe 调度器 |
| `aac54bb19` | 2026-10-28T10:24:20+08:00 | 同 A | tensor_view/make_*、tile_load/store；逐层核实导出 |

## 6. 来源与方法学结论

- **cann 当前路线确有 TA→FlagTree 同步证据**：`8c30adb6c` 明确给出 TA 基线 `c85527413`。dot/conv2d 晚于该基线，可解释缺失；libdevice 的精简+rebind 是当时 TA 原貌，不是 FT 主动裁剪。
- **TLE 不能全部解释成 cann 重导出**：Ascend 子核/同步/提示等是 re-export 或 alias；普通 DSA copy/算术/marker、tile 和 custom-op 各有实现。相似语义不是代码同源证明。
- **TLA 命名空间比 TLE 晚，不代表每项同类能力都晚**。早期 al/buffer 与后来具体 TLA 必须分列。反过来，TA 一般能力更早，也不代表 FT 专门指令或调度能力被 TA 覆盖。
- **本次未验证精确 PR 合入时刻，也未运行硬件测试或性能测试**。接口暴露、历史先后、数值/架构支持和性能是不同问题。

## 附：可复现 Git 命令

```bash
# 在相应仓库执行；路径限定避免命中 upstream 的同名符号
# 默认只查 HEAD 历史；--all 可找旧分支，但必须单列可达性
git log HEAD --reverse -G'^def subview\(' --format='%H|%aI|%cI|%s' -- python/triton/extension/buffer/language/core.py

# 真实新增/导出及迁移，不仅看提交标题
git show --find-renames --unified=3 <hash> -- path/to/file.py
git show <hash>:path/to/file.py
git log --follow --format='%H|%aI|%cI|%s' -- path/to/file.py

# 日期、HEAD 可达性；后者不提供精确合入时间
git show -s --format='%H|%aI|%cI|%s' <hash>
git merge-base --is-ancestor <hash> HEAD
# 是否属于 first-parent 历史，还需在以下输出中核对 hash
git rev-list --first-parent HEAD

# 同名同步原语需追踪旧文件，不能只限 core.py
git show 313dccecf:third_party/ascend/language/cann/extension/aux_ops.py
git show 313dccecf:third_party/ascend/language/cann/extension/__init__.py
```
