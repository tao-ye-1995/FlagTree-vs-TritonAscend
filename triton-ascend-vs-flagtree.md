# Triton-Ascend vs FlagTree:昇腾后端 API 与架构对比

> **以 Triton-Ascend 为基准**,看 FlagTree 在昇腾上与之的异同。
>
> 对比对象:
> - **Triton-Ascend**(基准):`/workspace/triton-ascend`(Triton 3.6.0,昇腾专用 fork)
> - **FlagTree**(对比方):`/workspace/FlagTree`(Triton 3.5.1,多后端统一框架,Ascend 为其一)
>
> AscendNPU-IR 版本路径:`/workspace/FlagTree/third_party/ascend/AscendNPU-IR`(FlagTree 侧 in-tree 完整)。

---

## 0. 结论速览(TL;DR)

| 维度 | Triton-Ascend(基准) | FlagTree(对比) |
|---|---|---|
| 定位 | **昇腾专用** fork | **多后端统一框架**(amd/nvidia/enflame/iluvatar/mthreads/xpu/tsingmicro/**ascend**/tle) |
| Triton 版本 | 3.6.0 | 3.5.1 |
| 后端版本 | ascend backend 3.6.0 | — |
| 用户 API 命名空间 | `cann.libdevice` + `cann.extension` + `extension.buffer.language` + `kernels` + **`tla`** | **少 `tla`**,其余对应 |
| **TLA 范式** | ✅ 有(`triton.language.extra.tla`,独立显存层级编程) | ❌ 无 |
| libdevice 实现 | **全量自实现**(~180 def / 4182 行) | **精简版**(39 def / 1004 行,大量 rebind 到 upstream `math.*`) |
| `cann.extension.dot` / `conv2d` | ✅ 有 | ❌ 无 |
| AscendNPU-IR | git submodule(**当前 workspace 未 checkout,空目录**) | **in-tree 完整**(submodule 已落地) |
| Triton→IR lowering passes | `TritonToLinalg/Structured/Unstructure` + `LibEntry` + `TritonMemoryAccess` 等(更全) | `CommonIRToHIVM` / `TritonAffinityOpt`(精简) |
| 后端编译器 | **bishengir-compile / bishengir-opt** | 同 |
| 特有机制 | backend monkey-patch(`_apply_ascend_patch`) | `flagtree_spec`(多后端符号注入 + spec_path) |

### 当前结论总结

1. **FlagTree 的 cann 层主要来自 Triton-Ascend 整体同步,而非逐个特性独立开发。** Git 记录显示,FlagTree 在 **2026-06-08** 的提交 `8c30adb6c` 中同步 TA,基线为 **2026-05-15** 的 `c85527413`。FlagTree 的源码目录是 [third_party/ascend/language/cann/](../FlagTree/third_party/ascend/language/cann/),包含 `libdevice.py` 和 `extension/`(`al`)。
2. **当前 al/libdevice 的主要差异首先是同步版本差,不能归因于 FlagTree 主动裁剪。** FlagTree 缺少 `al.dot`、`al.conv2d`,因为 TA 在同步基线之后才引入它们。FlagTree 的 libdevice“少量独立实现 + rebind 到 `math.*`”也是 TA 当时已有的形态;TA 此后扩展为全量 libdevice,FlagTree 当前仍主要保留历史形态。当前识别到的 FlagTree 独有 cann 扩展是 `custom_object_semantic`。历史依据见 [feature-timeline-git.md](feature-timeline-git.md)。
3. **显式内存编程层是两条不同路线:TA 的 TLA 与 FlagTree 的 TLE。** TA 使用 `triton.language.extra.tla`,更贴近昇腾 DMA、SIMD、Cube 指令;FlagTree 使用 `triton.experimental.tle`,包含跨后端的 buffer、tile、流水与昇腾扩展。不少能力相同或类似,但不能据此认定接口、编译行为和性能等价。TLE 还提供流水调度器和分布式通信接口,TA 当前未找到直接对应接口。
4. **API 已暴露不等于 A2/A3/950 全部支持。** 已确认 `al.fixpipe`、`al.copy`、`al.copy_from_ub_to_l1` 以及 `cast` 的 fp8/fp64 转换存在 950 专属限制。需区分硬件限制、dtype 限制和 SIMT 模式限制;源码没有架构守卫,也不等于所有组合都已运行验证。逐 API 对照表中的硬件标注是源码分析结论,不是全组合实测认证。
5. **两仓共享 AscendNPU-IR / BiShengIR 编译体系,但不代表编译器版本完全相同。** 用户层差异主要集中在 API 快照、TLA/TLE 编程方式、扩展注入机制和前端 lowering。当前 FlagTree 的 AscendNPU-IR 源码完整可查,TA 对应子模块在工作区尚未检出;底座版本是否一致需另行核对两仓引用的 commit 和实际使用的编译器。

6. **提交先后需要分具体接口与较早能力两种口径。** 当前 HEAD 历史中,TA 的早期 `al`/buffer API 多数先于 TLE DSA/Ascend 的 2026-07-01 导出;TLE tile 接口(2026-08-27)先于具体 TLA 接口(2026-10~11 月)。通用 tile_load/store 与视图族另有日期,不能套用整族结论;旧分支与整批导入也不能作为原创归属证据。逐项对照与作者/提交者日期见 [时间线 §3](feature-timeline-git.md#3-逐原语首见与先后对照)。

**一句话**:FlagTree 昇腾侧更适合理解为“TA 历史后端快照 + 多后端适配机制 + 独立演进的 TLE”,而不是“重新逐项实现或主动删减 TA API”。

---

## 1. 架构定位差异

### Triton-Ascend —— 昇腾专用 fork(基准)
```
triton-ascend/
├── python/triton/            ← upstream Triton 原版 + extension/buffer
└── third_party/ascend/       ← 唯一后端,昇腾全栈
    ├── AscendNPU-IR/         ← git submodule(.gitmodules 指向 gitcode Ascend/AscendNPU-IR)
    ├── language/{cann, tla, kernels}   ← 三套用户 API(tla 为独有)
    ├── include/ lib/         ← Triton→HIVM 全套 passes
    └── backend/              ← compiler/driver/launcher
```
- 专注单一后端,通过 `_apply_ascend_patch()` monkey-patch 把昇腾特性注入 upstream Triton。
- `python/triton/extension/buffer/language/` 直接内置通用 buffer 抽象。

### FlagTree —— 多后端统一框架(对比方)
```
FlagTree/third_party/
├── amd/  nvidia/  enflame/  iluvatar/  mthreads/  xpu/  tsingmicro/   ← 各家 NPU/GPU
├── ascend/            ← 昇腾后端(本次对比重点)
│   ├── AscendNPU-IR/  ← in-tree 完整 MLIR 编译器(BiShengIR)
│   ├── backend/  include/  lib/  language/  python/  runtime/
│   └── backend/spec/  ← flagtree_spec 适配层
└── tle/               ← Tensor-memory Language Extension(Hopper SMEM/TMA,见 tle.md)
```
- 统一入口 `python/triton/`,通过 `flagtree_spec.py` 把各后端的 language 扩展**注入** `tl.*` 命名空间([python/triton/flagtree_spec.py:64](../FlagTree/python/triton/flagtree_spec.py#L64) `bind_language_extension_symbols_to_tl`)。
- 昇腾专属的 `extension.buffer.language` 放在 `python/triton/spec/ascend/`(与 Triton-Ascend 对应实现**逐函数一致**)。

---

## 2. 用户可编程 API 对比(以 Triton-Ascend 为基准)

### 2.1 命名空间总表

| 命名空间 | Triton-Ascend(基准) | FlagTree | 在线文档收录 |
|---|---|---|---|
| `triton` / `triton.language` / `triton.testing` | ✅ 通用 | ✅ 通用(一致) | ✅ |
| `triton.language.extra.cann.libdevice` | ✅ 全量 | ✅ **精简** | ✅ |
| `triton.language.extra.cann.extension`(`al`) | ✅ 全 | ✅ **少 dot/conv2d** | ✅ |
| `triton.language.extra.cann.kernels`(`gather_2d_simd`) | ✅ | ✅ 一致 | ❌(文档外) |
| `triton.extension.buffer.language` | ✅ | ✅(spec/ascend 下) | ✅ |
| **`triton.language.extra.tla`** | **✅ 有** | **❌ 无**(近义能力在 TLE) | ❌(文档外) |
| **`triton.experimental.tle`**(TLE 扩展原语) | ❌ 无 | **✅ 有**(含 `dsa.ascend`) | ❌(文档外) |

> `kernels.gather_2d_simd` 两边源码**完全一致**。逐 API 行级对照见 [user-apis-flagtree-vs-triton-ascend.md](user-apis-flagtree-vs-triton-ascend.md)(TLE↔TA 对应见其表 4)。
>
> **TLE vs tla**:两者是各自的显式内存/流水编程层。TLE(`triton.experimental.tle`,跨 GPU/NPU)含完整昇腾实现 `dsa.ascend`,多数原语与 TA `tla`/`al`「相同或类似」(跨核同步、内存层级、tile 搬运、scope),但 TLE 额外带**流水调度器**(`pipe_scheduler`)与**分布式通信**(`device_mesh`/`remote`)能力,TA `tla` 无;反之 TA `tla.simd.v*`/`tla.cube.mmad` 的向量/cube 指令粒度在 TLE 里以 `tile_*` 抽象表达。

### 2.2 `cann.extension`(`al`)导出差异

以 Triton-Ascend 为基准,FlagTree 的增减:
- FlagTree **缺少**(Triton-Ascend 有):
  - `dot`、`conv2d` —— **cube 矩阵乘 / 2D 卷积 builtin**(重要功能差异)
  - `SyncEventSlot`、`EVENT_ID`、`SYNC_HINT`、`AddressSpace`、`DataLayout`、`is_register_based_arch`
- FlagTree **额外多出**:`custom_object_semantic`(自定义 op 的对象语义变体)

共有(两边一致):`copy`/`copy_from_ub_to_l1`/`fixpipe`/`scope`/`parallel`/`compile_hint`/`multibuffer`/`sync_block_*`/`conv1d`/`sub_vec_id`/`sub_vec_num`/`insert_slice`/`extract_slice`/`get_element`/`sort`/`flip`/`cast`/`index_put`/`gather_out_to_ub`/`scatter_ub_to_out`/`index_select_simd`/`custom`/`custom_semantic`/`register_custom_op`/`atan2`/`isfinited`/`finitef`/ affine 系列 / 枚举 `PIPE/MODE/CORE/IteratorType/Fixpipe*`。

> 源码:[triton-ascend extension/__init__.py](../triton-ascend/third_party/ascend/language/cann/extension/__init__.py) 与 [FlagTree extension/__init__.py](../FlagTree/third_party/ascend/language/cann/extension/__init__.py)。

> **架构限制(两仓一致)**:并非所有 `al.*` 都三代通用。经逐个核查,**`fixpipe` / `copy` / `copy_from_ub_to_l1` 是 950(register-based,`Ascend910_95`/`Ascend960`)专属**——A2/A3 调用会 `RuntimeError`(这些是 L0C/L1 register-based 片上搬运);**`cast` 的 fp8/fp64 转换也仅 950**(A2/A3 `ValueError`),基础 dtype 全支持。其余 `dot`/`conv1d`/`conv2d`/`sync_block_*`/`sub_vec_*`/`sort`/`flip`/`get_element`/`insert_slice`/`extract_slice`/`index_put`/`gather_out_to_ub`/`scatter_ub_to_out`/`index_select_simd` 等**未发现显式架构禁用守卫**;这不等于全部硬件/dtype/shape 组合已通过运行验证。逐行见对照表 [表 1.1](user-apis-flagtree-vs-triton-ascend.md)。

### 2.3 `cann.libdevice` 差异(实现策略不同)

| | Triton-Ascend(基准) | FlagTree |
|---|---|---|
| 行数 / `def` 数 | 4182 行 / ~180 | 1004 行 / 39 |
| 策略 | **全量自实现** `@core.extern` | **大量 rebind 到 upstream** `triton.language.math.*` |

Triton-Ascend 维护全套 libdevice(含 `add_rn/mul_rz/float2int_ru`、Bessel `j0/j1/jn`、`gamma/erfinv/normcdf` 等)。FlagTree 在 [cann/__init__.py](../FlagTree/third_party/ascend/language/cann/__init__.py) 里把 ~20 个函数指回 upstream math:
```python
libdevice.exp  = math.exp;   libdevice.log = math.log;   libdevice.sqrt = math.sqrt
libdevice.cos  = math.cos;   libdevice.sin = math.sin;   libdevice.rsqrt = math.rsqrt
libdevice.abs  = math.abs;   libdevice.fma = math.fma;   libdevice.erf  = math.erf  ...
```

> 含义:Triton-Ascend 把 libdevice 作为独立全集维护;FlagTree 更依赖 upstream 数学实现 + 后端 lowering,接口暴露面更小。两边都 export `reciprocal/log1p/relu/isinf/tan/atan/tanh/ilogb/ldexp/pow/isnan` 等核心 extern。

> **架构维度**:libdevice 全部函数在 **A2/A3/950 硬件均可用**,无「某代禁用」。差异在于:大量带舍入/类型转换/特殊函数/位运算的函数带 `if SIMT: static_assert(False)` 守卫——**SIMT 是「950 + 环境开关 `TRITON_ENABLE_LIBDEVICE_SIMT`」才进入的模式,非芯片限制**,普通模式三代全支持;另有 `mulhi/hadd/rhadd` 在 950 走不同实现、`exp/sqrt/reciprocal` 等的 fp32 在 950/A6 走优化路径。两仓机制一致(FlagTree 用 `triton_enable_libdevice_simt() and is_compile_on_910_95` 作同样守卫)。

#### "rebind" 到底是什么含义

同样写 `libdevice.exp(x)`,两仓指向的**不是同一个函数对象**:

- **Triton-Ascend**:`libdevice.py` 里有独立的 `def exp`(`@core.extern`,[libdevice.py:3865](../triton-ascend/third_party/ascend/language/cann/libdevice.py#L3865)),自己定义调用哪个底层符号、支持哪些 dtype。
- **FlagTree**:`libdevice.py` 里**根本没有** `def exp`;而是在 [cann/__init__.py](../FlagTree/third_party/ascend/language/cann/__init__.py) 执行时,用 `libdevice.exp = math.exp` **把一个已存在于别处的实现贴到 `libdevice.exp` 这个名字上**——这就是 rebind(重新绑定模块属性)。

因此 FlagTree 的 `libdevice` 是个**聚合壳**,三类来源混合(注:这套形态搬自 TA 2026-05-15 快照,非 FlagTree 自创,见下方澄清):

| 来源 | 数量 | 例子 | 说明 |
|---|---|---|---|
| **自实现** `@core.extern` | 39 | `tanh/atan/pow/erfinv/gamma/hypot` | 昇腾侧 extern(TA 当时写的,随搬入带来) |
| rebind 自 **upstream `math.*`** | ~17 | `exp/exp2/log/log2/sin/cos/sqrt/sqrt_rn/rsqrt/div_rn/erf/floor/ceil/fdiv/fma/abs/umulhi` | 复用 Triton 官方通用数学实现(各后端共享) |
| rebind 自 **`cann.extension`** | 4 | `flip`([vec_ops.py](../FlagTree/third_party/ascend/language/cann/extension/vec_ops.py)) / `isfinited` / `finitef` / `atan2`([math_ops.py](../FlagTree/third_party/ascend/language/cann/extension/math_ops.py)) | 指向昇腾自己的 builtin 扩展 |

对用户的影响:

1. **可用性相同、无感知**:这些函数照常 `libdevice.xxx(x)` 调用,结果正确。
2. **底层实现来源不同**:rebind 的函数走 upstream 通用 lowering(或 extension builtin),而非 TA 式的昇腾定制 extern。多数标准函数数值一致,但**边界/精度/特殊 dtype 下行为可能有细微差异**。
3. **部分 rebind 带开关**:`atan2` 是**条件性**的——[cann/__init__.py](../FlagTree/third_party/ascend/language/cann/__init__.py) `if not triton_enable_libdevice_simt()` 才指向 extension 版本(SIMT 模式下不用)。
4. **存在反向 rebind**:`math.tanh = libdevice.tanh`——把昇腾的 `tanh` 贴回 upstream `math`,使 `tl.tanh` 等上层调用也走昇腾实现(此行两仓一致)。

> 重要澄清(git 考据,见 [feature-timeline-git.md](feature-timeline-git.md)):**FlagTree 的 cann 层不是自己逐特性开发的,而是 2026-06-08 从 TA 整体搬入**(同步基线 = TA commit `c85527413` @ 2026-05-15),之后 cann 目录在 FlagTree 几乎零改动。所谓"精简 + rebind"形态,**正是 TA 在 2026-05-15 那个时点的原貌**(当时 TA libdevice 就只有 37 函数/1044 行、`__init__.py` 里就有 `libdevice.exp=math.exp` 等 rebind)。**是 TA 自己在基线之后把 libdevice 膨胀到了全量 183 函数/4182 行并移除 rebind**(全量 SIMT 版 `7c6f063ae` 等在基线后才并入 TA 主线)。因此"FlagTree 精简、TA 全量"的差异,本质是**两仓在 libdevice 演进时间轴上的快照差**,不是 FlagTree 主动裁剪。

### 2.4 TLA(Triton-Ascend 独有,FlagTree 无)

`triton.language.extra.tla` 是 Triton-Ascend 的显存层级显式编程范式:`@tla.jit` + `alloc_buffer`(UB/L1/L0)+ DMA(`load_gm_to_ub`/`store_l0c_to_gm`…)+ `tla.simd.v*` 向量指令 + `tla.cube.mmad` + `tla.scope.simd/cube` + `tla.sync.*`。FlagTree **完全不存在**此命名空间。

> 详见 [triton-ascend-user-apis.md](triton-ascend-user-apis.md)。
> 这也解释了 Triton-Ascend compiler 里的 `enable_tla_mode` 分支(`make_tla_ttir`/`tla_to_ttadapter`,见 [compiler.py:1683](../triton-ascend/third_party/ascend/backend/compiler.py#L1683)),FlagTree 无此分支。

---

## 3. 编译管线(lowering)对比

### 3.1 后端 pipeline stages

**Triton-Ascend**(基准,[compiler.py:1681 add_stages](../triton-ascend/third_party/ascend/backend/compiler.py#L1681)):
```
       ┌─ enable_tla_mode: ttir(make_tla_ttir) ─▶ ttadapter(tla_to_ttadapter)
ttir ──┤
       └─ 普通:  ttir(make_ttir) ─▶ [is_pure_simt: npubin(ttir_to_npubin)]
                                   └▶ ttadapter(ttir_to_linalg)
                       ▼
          910_95:  linalg_to_bin_enable_npu_compile_910_95
          A2/A3 :  linalg_to_bin_enable_npu_compile_A2_A3   (均 → bishengir-compile)
```

**FlagTree**([compiler.py:1125 add_stages](../FlagTree/third_party/ascend/backend/compiler.py#L1125)):
```
ttir ──make_ttir──▶ ttadapter(linalg, ttir_to_linalg, named_ops=True)
                       │  (可选 mlirbc: linalg_to_bc_by_triton_mlir_opt)
                       │  (可选 bcmlir: bc_to_linalg_by_bishengir_opt)
                       ▼
                    npubin (_compile_linalg_to_npu_bin → bishengir-compile)
     └─(SIMT 纯标量路径) ttir ──▶ npubin (ttir_to_npubin)
```

**共性**:都走 `ttir → ttadapter(linalg) → npubin`,终点都是 **`bishengir-compile` / `bishengir-opt`**(AscendNPU-IR 的二进制工具)。两边都有 `_check_bishengir_api_change / _check_bishengir_is_regbased`、910_95 与 A2/A3 分芯片分支。

**差异(以 Triton-Ascend 为基准)**:
- Triton-Ascend 多 **TLA 前端分支**,FlagTree 无。
- Triton-Ascend 的 SIMT 纯标量路径用 `is_pure_simt`,FlagTree 用 `triton_enable_libdevice_simt()` 开关(体现在 libdevice rebind 条件里)。

### 3.2 Triton-side MLIR passes 差异

(`include/` 下,`diff` 结果)

| Pass 目录 | Triton-Ascend(基准) | FlagTree |
|---|---|---|
| `TritonToHIVM` / `TritonToHFusion` / `TritonToGraph` / `TritonToAnnotation` / `TritonToLLVM` / `DynamicCVPipeline` / `AutoBlockify` / `Dialect` | ✅ | ✅ |
| `TritonToLinalg` / `TritonToStructured` / `TritonToUnstructure` | ✅ 独有 | ❌ |
| `LibEntry` / `TritonControlFlowOpt` / `TritonMemoryAccess` / `DiscreteMaskAccessConversion` / `Utils` | ✅ 独有 | ❌ |
| `CommonIRToHIVM` | ❌ | ✅ 独有 |
| `TritonAffinityOpt` | ❌ | ✅ 独有 |

> 解读:Triton-Ascend 的前端转换链路更细(显式的 Structured/Unstructure/MemoryAccess/DiscreteMask 等 pass),与它更强的"直接控内存访问 / fractal / mask"能力一致(呼应 `cann.extension.dot` 的 fractal 参数)。FlagTree 把一部分合并进 `CommonIRToHIVM` + `TritonAffinityOpt`,保持通用框架的简洁。

---

## 4. AscendNPU-IR(BiShengIR)—— 共享的编译底座

路径:[FlagTree/third_party/ascend/AscendNPU-IR/bishengir](../FlagTree/third_party/ascend/AscendNPU-IR/bishengir)

- 这是 **MLIR-for-Ascend** 的开源编译器(GitCode `Ascend/AscendNPU-IR`),提供从硬件无关表达到达芬奇指令的编译优化。
- **两个仓库共享这一编译体系,不代表使用完全相同的版本**:Triton-Ascend 以 git submodule 引入(`.gitmodules` 指向同一 gitcode 仓),**但当前 workspace 未 checkout(空目录)**;FlagTree 则是完整 in-tree。当前可用 FlagTree 侧源码研究底座实现,但不能直接视为 TA 对应版本的实现;版本一致性需核对两仓引用的 commit 和实际编译器。

### Dialect 层(`include/bishengir/Dialect/`)
```
Analysis  Annotation  Bufferization  HACC  HFusion  HIVM
Linalg  MathExt  MemRef  MemRefExt  SCF  Scope  Symbol  Tensor  Torch  Utils
```
- **HIVM**:核心硬件抽象(Huawei Intermediate Vector Machine),Triton 各后端最终汇聚到此。
- **HACC**:target/架构描述(`NPUTargetSpec.td`,如 `hacc.target<"Ascend910B">`)—— 正是 backend monkey-patch 注入的 `hacc.target` 属性所在。
- **HFusion**:融合算子表达。
- **MemRefExt / Scope / Annotation / Symbol**:内存扩展、作用域、标注、符号。

工具:`bishengir/tools`(含 `bishengir-compile`、`bishengir-opt`),两边 compiler 都调用它完成 linalg→npubin。

---

## 5. 特有机制

### Triton-Ascend:backend monkey-patch(基准)
- [backend/__init__.py](../triton-ascend/third_party/ascend/backend/__init__.py) `_apply_ascend_patch()`:注入 `hacc.target`、改写 `tl.dot`(tf32→hf32、禁 imprecise acc)、dtype guard、IR parse 扩展(支持 `npubin`/`ttadapter`)。
- `install_ascend_knobs()`:注册昇腾专属编译选项/环境变量。

### FlagTree:`flagtree_spec`
- [python/triton/flagtree_spec.py](../FlagTree/python/triton/flagtree_spec.py)
- `bind_language_extension_symbols_to_tl(extension)`:把 `cann.extension.__all__` 里 upstream `tl` 尚无的符号**动态注入** `tl.*`,实现"扩展算子像内置一样用"。
- `spec_path(__path__)`:让 `triton.extension` 指向各后端的 spec 目录(多后端共存的命名空间分发)。

---

## 6. 做对比研究时的建议路径

1. **用户 API 面**:以 [triton-ascend-user-apis.md](triton-ascend-user-apis.md) 为 Triton-Ascend 基线,FlagTree 侧按 §2 的差异裁剪(去掉 tla、dot/conv2d,libdevice 按 rebind 表折算);逐行对照见 [user-apis-flagtree-vs-triton-ascend.md](user-apis-flagtree-vs-triton-ascend.md)。
2. **编译器底座**:两边终点一致(bishengir),差异集中在 **Triton-side passes(§3.2)** 与 **TLA 前端**。看 IR 细节用 FlagTree 的 in-tree AscendNPU-IR(§4)。
3. **若要跑通 Triton-Ascend 的 AscendNPU-IR**:需先 `git submodule update --init third_party/ascend/AscendNPU-IR`(当前为空)。

---

## 附:关键源码位置索引

| 主题 | Triton-Ascend(基准) | FlagTree |
|---|---|---|
| cann 注册 | [language/cann/__init__.py](../triton-ascend/third_party/ascend/language/cann/__init__.py) | [language/cann/__init__.py](../FlagTree/third_party/ascend/language/cann/__init__.py) |
| extension 导出 | [cann/extension/__init__.py](../triton-ascend/third_party/ascend/language/cann/extension/__init__.py) | [cann/extension/__init__.py](../FlagTree/third_party/ascend/language/cann/extension/__init__.py) |
| libdevice | [cann/libdevice.py](../triton-ascend/third_party/ascend/language/cann/libdevice.py) | [cann/libdevice.py](../FlagTree/third_party/ascend/language/cann/libdevice.py) |
| TLA | [language/tla/](../triton-ascend/third_party/ascend/language/tla/) | — |
| buffer.language | [extension/buffer/language/](../triton-ascend/python/triton/extension/buffer/language/) | [spec/ascend/extension/buffer/language/](../FlagTree/python/triton/spec/ascend/extension/buffer/language/) |
| compiler | [backend/compiler.py](../triton-ascend/third_party/ascend/backend/compiler.py) | [backend/compiler.py](../FlagTree/third_party/ascend/backend/compiler.py) |
| Triton passes | [include/](../triton-ascend/third_party/ascend/include/) | [include/](../FlagTree/third_party/ascend/include/) |
| AscendNPU-IR | (submodule 未 checkout) | [AscendNPU-IR/bishengir/](../FlagTree/third_party/ascend/AscendNPU-IR/bishengir/) |
| 特有机制 | [backend/__init__.py](../triton-ascend/third_party/ascend/backend/__init__.py) | [flagtree_spec.py](../FlagTree/python/triton/flagtree_spec.py) |
