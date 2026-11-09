# FlagTree vs Triton-Ascend:昇腾后端 API 与架构对比

> 对比对象:
> - **FlagTree**:`/workspace/FlagTree`(Triton 3.5.1,多后端统一框架,Ascend 为其一)
> - **Triton-Ascend**:`/workspace/triton-ascend`(Triton 3.6.0,昇腾专用 fork)
>
> AscendNPU-IR 版本路径:`/workspace/FlagTree/third_party/ascend/AscendNPU-IR`(FlagTree 侧 in-tree 完整)。

---

## 0. 结论速览(TL;DR)

| 维度 | FlagTree | Triton-Ascend |
|---|---|---|
| 定位 | **多后端统一框架**(amd/nvidia/enflame/iluvatar/mthreads/xpu/tsingmicro/**ascend**/tle) | **昇腾专用** fork |
| Triton 版本 | 3.5.1 | 3.6.0 |
| 后端版本 | — | ascend backend 3.6.0 |
| 用户 API 命名空间 | `cann.libdevice` + `cann.extension` + `extension.buffer.language` + `kernels` | 同上 **再加 `tla`** |
| **TLA 范式** | ❌ 无 | ✅ 有(`triton.language.extra.tla`,独立显存层级编程) |
| libdevice 实现 | **精简版**(39 def / 1004 行,大量 rebind 到 upstream `math.*`) | **全量自实现**(~180 def / 4182 行) |
| `cann.extension.dot` / `conv2d` | ❌ 无 | ✅ 有 |
| AscendNPU-IR | **in-tree 完整**(submodule 已落地) | git submodule(**当前 workspace 未 checkout,空目录**) |
| Triton→IR lowering passes | `CommonIRToHIVM` / `TritonAffinityOpt`(精简) | `TritonToLinalg/Structured/Unstructure` + `LibEntry` + `TritonMemoryAccess` 等(更全) |
| 后端编译器 | 均经 **bishengir-compile / bishengir-opt** | 同 |
| 特有机制 | `flagtree_spec`(多后端符号注入 + spec_path) | backend monkey-patch(`_apply_ascend_patch`) |

**一句话**:两者共享同一套昇腾编译底座(AscendNPU-IR / BiShengIR)和大部分 `cann.*` 用户 API,但 **Triton-Ascend 功能更全(多出 TLA 范式、全量 libdevice、dot/conv2d)**,而 **FlagTree 以"多后端通用、昇腾精简接入"为目标**,把能复用 upstream 的部分(math 函数等)尽量 rebind、不重复造。

---

## 1. 架构定位差异

### FlagTree —— 多后端统一框架
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
- `python/triton/spec/ascend/` 存放昇腾专属的 `extension.buffer.language` 实现(与 Triton-Ascend 的 `python/triton/extension/buffer/language/` 源码**逐函数一致**)。

### Triton-Ascend —— 昇腾专用 fork
```
triton-ascend/
├── python/triton/            ← upstream Triton 原版 + extension/buffer
└── third_party/ascend/       ← 唯一后端,昇腾全栈
    ├── AscendNPU-IR/         ← git submodule(.gitmodules 指向 gitcode Ascend/AscendNPU-IR)
    ├── language/{cann, tla, kernels}   ← 三套用户 API(tla 为独有)
    ├── include/ lib/         ← Triton→HIVM 全套 passes
    └── backend/              ← compiler/driver/launcher
```

---

## 2. 用户可编程 API 对比

### 2.1 命名空间总表

| 命名空间 | FlagTree | Triton-Ascend | 在线文档收录 |
|---|---|---|---|
| `triton` / `triton.language` / `triton.testing` | ✅ 通用 | ✅ 通用 | ✅ |
| `triton.language.extra.cann.libdevice` | ✅ 精简 | ✅ 全量 | ✅ |
| `triton.language.extra.cann.extension`(`al`) | ✅ 少 dot/conv2d | ✅ 全 | ✅ |
| `triton.language.extra.cann.kernels`(`gather_2d_simd`) | ✅ 一致 | ✅ | ❌(文档外) |
| `triton.extension.buffer.language` | ✅(spec/ascend 下) | ✅ | ✅ |
| **`triton.language.extra.tla`** | **❌ 无** | **✅ 有** | ❌(文档外) |

> `kernels.gather_2d_simd` 两边源码**完全一致**。

### 2.2 `cann.extension`(`al`)导出差异

FlagTree **缺少**(Triton-Ascend 独有):
- `dot`、`conv2d` —— **cube 矩阵乘 / 2D 卷积 builtin**(重要功能差异)
- `SyncEventSlot`、`EVENT_ID`、`SYNC_HINT`、`AddressSpace`、`DataLayout`、`is_register_based_arch`

FlagTree **独有**:
- `custom_object_semantic`(自定义 op 的对象语义变体)

共有(两边一致):`copy`/`copy_from_ub_to_l1`/`fixpipe`/`scope`/`parallel`/`compile_hint`/`multibuffer`/`sync_block_*`/`conv1d`/`sub_vec_id`/`sub_vec_num`/`insert_slice`/`extract_slice`/`get_element`/`sort`/`flip`/`cast`/`index_put`/`gather_out_to_ub`/`scatter_ub_to_out`/`index_select_simd`/`custom`/`custom_semantic`/`register_custom_op`/`atan2`/`isfinited`/`finitef`/ affine 系列 / 枚举 `PIPE/MODE/CORE/IteratorType/Fixpipe*`。

> 对比命令:`diff <(FlagTree __all__) <(triton-ascend __all__)`,源码分别在
> [FlagTree extension/__init__.py](../FlagTree/third_party/ascend/language/cann/extension/__init__.py) 与
> [triton-ascend extension/__init__.py](../triton-ascend/third_party/ascend/language/cann/extension/__init__.py)。

### 2.3 `cann.libdevice` 差异(实现策略不同)

| | FlagTree | Triton-Ascend |
|---|---|---|
| 行数 / `def` 数 | 1004 行 / 39 | 4182 行 / ~180 |
| 策略 | **大量 rebind 到 upstream** `triton.language.math.*` | **全量自实现** `@core.extern` |

FlagTree 在 [cann/__init__.py](../FlagTree/third_party/ascend/language/cann/__init__.py) 里显式把 ~20 个函数指回 upstream math:
```python
libdevice.exp  = math.exp;   libdevice.log = math.log;   libdevice.sqrt = math.sqrt
libdevice.cos  = math.cos;   libdevice.sin = math.sin;   libdevice.rsqrt = math.rsqrt
libdevice.abs  = math.abs;   libdevice.fma = math.fma;   libdevice.erf  = math.erf  ...
```
而 Triton-Ascend 的这些函数都有独立的 `@core.extern` 定义(`add_rn/mul_rz/float2int_ru`、Bessel `j0/j1/jn`、`gamma/erfinv/normcdf` 等全套)。

> 含义:**FlagTree 更依赖 upstream 的数学实现 + 后端 lowering**,接口暴露面更小;Triton-Ascend 把 libdevice 作为独立全集维护。两边都 export `reciprocal/log1p/relu/isinf/tan/atan/tanh/ilogb/ldexp/pow/isnan` 等核心 extern。

### 2.4 TLA(Triton-Ascend 独有)

`triton.language.extra.tla` 在 FlagTree **完全不存在**。它是 Triton-Ascend 的显存层级显式编程范式:`@tla.jit` + `alloc_buffer`(UB/L1/L0)+ DMA(`load_gm_to_ub`/`store_l0c_to_gm`…)+ `tla.simd.v*` 向量指令 + `tla.cube.mmad` + `tla.scope.simd/cube` + `tla.sync.*`。

> 详见同目录 [triton-ascend-user-apis.md](triton-ascend-user-apis.md) 第 7 类。
> 这也解释了 Triton-Ascend compiler 里的 `enable_tla_mode` 分支(`make_tla_ttir`/`tla_to_ttadapter`,见 [compiler.py:1683](../triton-ascend/third_party/ascend/backend/compiler.py#L1683)),FlagTree 无此分支。

---

## 3. 编译管线(lowering)对比

### 3.1 后端 pipeline stages

**FlagTree**([compiler.py:1125 add_stages](../FlagTree/third_party/ascend/backend/compiler.py#L1125)):
```
ttir ──make_ttir──▶ ttadapter(linalg, ttir_to_linalg, named_ops=True)
                       │  (可选 mlirbc: linalg_to_bc_by_triton_mlir_opt)
                       │  (可选 bcmlir: bc_to_linalg_by_bishengir_opt)
                       ▼
                    npubin (_compile_linalg_to_npu_bin → bishengir-compile)
     └─(SIMT 纯标量路径) ttir ──▶ npubin (ttir_to_npubin)
```

**Triton-Ascend**([compiler.py:1681 add_stages](../triton-ascend/third_party/ascend/backend/compiler.py#L1681)):
```
       ┌─ enable_tla_mode: ttir(make_tla_ttir) ─▶ ttadapter(tla_to_ttadapter)
ttir ──┤
       └─ 普通:  ttir(make_ttir) ─▶ [is_pure_simt: npubin(ttir_to_npubin)]
                                   └▶ ttadapter(ttir_to_linalg)
                       ▼
          910_95:  linalg_to_bin_enable_npu_compile_910_95
          A2/A3 :  linalg_to_bin_enable_npu_compile_A2_A3   (均 → bishengir-compile)
```

**共性**:都走 `ttir → ttadapter(linalg) → npubin`,终点都是 **`bishengir-compile` / `bishengir-opt`**(AscendNPU-IR 的二进制工具)。两边都有 `_check_bishengir_api_change / _check_bishengir_is_regbased`、910_95 与 A2/A3 分芯片分支。

**差异**:
- Triton-Ascend 多 **TLA 前端分支**。
- Triton-Ascend 的 SIMT 纯标量路径用 `is_pure_simt`,FlagTree 用 `triton_enable_libdevice_simt()` 开关(体现在 libdevice rebind 条件里)。

### 3.2 Triton-side MLIR passes 差异

(`include/` 下,`diff` 结果)

| Pass 目录 | FlagTree | Triton-Ascend |
|---|---|---|
| `TritonToHIVM` / `TritonToHFusion` / `TritonToGraph` / `TritonToAnnotation` / `TritonToLLVM` / `DynamicCVPipeline` / `AutoBlockify` / `Dialect` | ✅ | ✅ |
| `CommonIRToHIVM` | ✅ 独有 | ❌ |
| `TritonAffinityOpt` | ✅ 独有 | ❌ |
| `TritonToLinalg` / `TritonToStructured` / `TritonToUnstructure` | ❌ | ✅ 独有 |
| `LibEntry` / `TritonControlFlowOpt` / `TritonMemoryAccess` / `DiscreteMaskAccessConversion` / `Utils` | ❌ | ✅ 独有 |

> 解读:Triton-Ascend 的前端转换链路更细(显式的 Structured/Unstructure/MemoryAccess/DiscreteMask 等 pass),与它更强的"直接控内存访问 / fractal / mask"能力一致(呼应 `cann.extension.dot` 的 fractal 参数)。FlagTree 把一部分合并进 `CommonIRToHIVM` + `TritonAffinityOpt`,保持通用框架的简洁。

---

## 4. AscendNPU-IR(BiShengIR)—— 共享的编译底座

路径:[FlagTree/third_party/ascend/AscendNPU-IR/bishengir](../FlagTree/third_party/ascend/AscendNPU-IR/bishengir)

- 这是 **MLIR-for-Ascend** 的开源编译器(GitCode `Ascend/AscendNPU-IR`),提供从硬件无关表达到达芬奇指令的编译优化。
- **两个仓库用的是同一个它**:Triton-Ascend 以 git submodule 引入(`.gitmodules` 指向同一 gitcode 仓),**但当前 workspace 未 checkout(空目录)**;FlagTree 则是完整 in-tree。所以要看 AscendNPU-IR 实现细节,**应以 FlagTree 侧为准**。

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

### FlagTree:`flagtree_spec`
- [python/triton/flagtree_spec.py](../FlagTree/python/triton/flagtree_spec.py)
- `bind_language_extension_symbols_to_tl(extension)`:把 `cann.extension.__all__` 里 upstream `tl` 尚无的符号**动态注入** `tl.*`,实现"扩展算子像内置一样用"。
- `spec_path(__path__)`:让 `triton.extension` 指向各后端的 spec 目录(多后端共存的命名空间分发)。

### Triton-Ascend:backend monkey-patch
- [backend/__init__.py](../triton-ascend/third_party/ascend/backend/__init__.py) `_apply_ascend_patch()`:注入 `hacc.target`、改写 `tl.dot`(tf32→hf32、禁 imprecise acc)、dtype guard、IR parse 扩展(支持 `npubin`/`ttadapter`)。
- `install_ascend_knobs()`:注册昇腾专属编译选项/环境变量。

---

## 6. 做对比研究时的建议路径

1. **用户 API 面**:以本目录 [triton-ascend-user-apis.md](triton-ascend-user-apis.md) 为 Triton-Ascend 基线,FlagTree 侧按 §2 的差异裁剪(去掉 tla、dot/conv2d,libdevice 按 rebind 表折算)。
2. **编译器底座**:两边终点一致(bishengir),差异集中在 **Triton-side passes(§3.2)** 与 **TLA 前端**。看 IR 细节用 FlagTree 的 in-tree AscendNPU-IR(§4)。
3. **若要跑通 Triton-Ascend 的 AscendNPU-IR**:需先 `git submodule update --init third_party/ascend/AscendNPU-IR`(当前为空)。

---

## 附:关键源码位置索引

| 主题 | FlagTree | Triton-Ascend |
|---|---|---|
| cann 注册 | [language/cann/__init__.py](../FlagTree/third_party/ascend/language/cann/__init__.py) | [language/cann/__init__.py](../triton-ascend/third_party/ascend/language/cann/__init__.py) |
| extension 导出 | [cann/extension/__init__.py](../FlagTree/third_party/ascend/language/cann/extension/__init__.py) | [cann/extension/__init__.py](../triton-ascend/third_party/ascend/language/cann/extension/__init__.py) |
| libdevice | [cann/libdevice.py](../FlagTree/third_party/ascend/language/cann/libdevice.py) | [cann/libdevice.py](../triton-ascend/third_party/ascend/language/cann/libdevice.py) |
| TLA | — | [language/tla/](../triton-ascend/third_party/ascend/language/tla/) |
| buffer.language | [spec/ascend/extension/buffer/language/](../FlagTree/python/triton/spec/ascend/extension/buffer/language/) | [extension/buffer/language/](../triton-ascend/python/triton/extension/buffer/language/) |
| compiler | [backend/compiler.py](../FlagTree/third_party/ascend/backend/compiler.py) | [backend/compiler.py](../triton-ascend/third_party/ascend/backend/compiler.py) |
| Triton passes | [include/](../FlagTree/third_party/ascend/include/) | [include/](../triton-ascend/third_party/ascend/include/) |
| AscendNPU-IR | [AscendNPU-IR/bishengir/](../FlagTree/third_party/ascend/AscendNPU-IR/bishengir/) | (submodule 未 checkout) |
| 特有机制 | [flagtree_spec.py](../FlagTree/python/triton/flagtree_spec.py) | [backend/__init__.py](../triton-ascend/third_party/ascend/backend/__init__.py) |
