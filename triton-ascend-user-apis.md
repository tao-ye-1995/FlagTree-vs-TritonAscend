# Triton-Ascend 用户可编程 API 概要

> 仓库 `/workspace/triton-ascend`(Triton 3.6.0 / ascend backend 3.6.0)。
> 组织对齐在线文档「API 参考」侧边栏:<https://triton-ascend.readthedocs.io/zh-cn/latest/>
> 逐 API 的两仓对照见 [user-apis-flagtree-vs-triton-ascend.md](user-apis-flagtree-vs-triton-ascend.md)。

## 命名空间全景

| 命名空间 | 定位 | 文档收录 | 源码 |
|---|---|---|---|
| `triton` | 入口装饰器/配置 | ✅ | [python/triton/runtime/](../triton-ascend/python/triton/runtime/) |
| `triton.language`(`tl`) | 核心 DSL | ✅ | [python/triton/language/](../triton-ascend/python/triton/language/) |
| `triton.testing` | 性能测试(含 `do_bench_npu`) | ✅ | [testing.py](../triton-ascend/python/triton/testing.py) + [backend/testing.py](../triton-ascend/third_party/ascend/backend/testing.py) |
| `triton.language.extra.cann.libdevice` | 昇腾数学库(~180,全量自实现) | ✅ | [cann/libdevice.py](../triton-ascend/third_party/ascend/language/cann/libdevice.py) |
| `triton.language.extra.cann.extension`(`al`) | 昇腾底层 builtin | ✅ | [cann/extension/](../triton-ascend/third_party/ascend/language/cann/extension/) |
| `triton.extension.buffer.language` | 通用 buffer 抽象 | ✅ | [extension/buffer/language/](../triton-ascend/python/triton/extension/buffer/language/) |
| `triton.language.extra.cann.kernels` | 现成高层 kernel(`gather_2d_simd`) | ❌ | [kernels/gather.py](../triton-ascend/third_party/ascend/language/kernels/gather.py) |
| **`triton.language.extra.tla`** ⚠️ | **TLA:显存层级显式编程范式** | ❌ | [language/tla/](../triton-ascend/third_party/ascend/language/tla/) |

> 打包:`third_party/ascend/language/<X>/` → `triton.language.extra.<X>`(见 [setup.py:639](../triton-ascend/setup.py#L639))。

## 三层心智模型

1. **像写 CUDA Triton**:`tl.*` + `cann.libdevice.*`,编译器自动 tiling/lowering —— 大多数用户。
2. **手动控硬件、仍在 Triton 语义内**:加 `cann.extension`(`al`)的 builtin —— `dot`/`fixpipe`/`copy`/跨核同步/`custom` 自定义 op。
3. **完全显式控制内存层级与流水**:用 `tla.*`(文档外)—— `@tla.jit` + alloc UB/L1/L0 + DMA + SIMD 指令 + cube `mmad` + 同步。

---

## 各命名空间要点

### `triton` / `triton.language` / `triton.testing`(通用,继承 upstream)
- `triton`:`jit`、`autotune`、`heuristics`、`Config`。
- `triton.language`:17 个分组(Creation / Shape / LinearAlgebra(`dot`,`dot_scaled`)/ Memory-Pointer / Indexing / Math / Logical / Comparison / Reduction / Scan-Sort / Atomic / Random / Iterators / CompilerHint / Debug / InlineAssembly)。
- `triton.testing`:`do_bench` / `perf_report` / `assert_close` / **`do_bench_npu`**(昇腾专属)。

### `cann.libdevice`(昇腾数学库)
~180 个 `@core.extern`,功能归类:基础数学、三角/双曲、**带舍入模式**(`add_rn/mul_rz/div_rd/float2int_ru`…)、类型转换、快速近似(`fast_*`)、特殊函数(`erf/gamma/j0/jn/normcdf`…)、判定/位运算。**全量自实现**是与 FlagTree 的主要差异点。

### `cann.extension`(`al`,昇腾底层 builtin)
按文件分组:
- [core.py](../triton-ascend/third_party/ascend/language/cann/extension/core.py):`copy`/`fixpipe`/**`dot`**/**`conv1d`/`conv2d`**/`debug_barrier`/`sub_vec_id/num`/`sync_block_*`;枚举 `PIPE/MODE/CORE/IteratorType/Fixpipe*/AddressSpace/DataLayout`。
- [custom_op.py](../triton-ascend/third_party/ascend/language/cann/extension/custom_op.py):`custom`/`custom_semantic`/`register_custom_op`/`SyncEventSlot`。
- [vec_ops.py](../triton-ascend/third_party/ascend/language/cann/extension/vec_ops.py):`insert_slice`/`extract_slice`/`get_element`/`sort`/`flip`/`cast`。
- [mem_ops.py](../triton-ascend/third_party/ascend/language/cann/extension/mem_ops.py):`index_put`/`gather_out_to_ub`/`scatter_ub_to_out`/`index_select_simd`。
- [aux_ops.py](../triton-ascend/third_party/ascend/language/cann/extension/aux_ops.py):`parallel`/`compile_hint`/`multibuffer`。
- [math_ops.py](../triton-ascend/third_party/ascend/language/cann/extension/math_ops.py):`atan2`/`isfinited`/`finitef`。
- affine(MLIR 透传):`affine_map`/`affine_expr`/…

### `extension.buffer.language`(通用 buffer)
`buffer`/`buffer_type`/`address_space` + `alloc`/`to_buffer`/`to_tensor`/`subview`([core.py](../triton-ascend/python/triton/extension/buffer/language/core.py))。

### `tla`(⚠️ 文档外,Triton-Ascend 独有)
`@tla.jit` 装饰器;`alloc_buffer/alloc/buffer/vector`;`view/subview/reshape`;DMA 搬运族 `load_gm_to_ub`/`store_ub_to_gm`/`copy_l1_to_l0`/`store_l0c_to_gm`… 及参数类/枚举;`tla.simd.*`(~60 向量指令);`tla.cube.mmad`;`tla.scope.simd/cube`;`tla.sync.*`;`where`/`compensated_rstd`/`device_print`。详见 [flagtree-vs-triton-ascend.md](flagtree-vs-triton-ascend.md) §2.4。

---

## 典型导入模板

```python
import triton
import triton.language as tl
from triton.backends.ascend import _apply_ascend_patch          # 必备:激活昇腾补丁
import triton.language.extra.cann.libdevice as libdevice        # 数学函数
import triton.language.extra.cann.extension as al               # 底层 builtin
import triton.language.extra.tla as tla                         # TLA 范式(文档外)
from triton._C.libtriton.ascend import ir as ascend_ir          # MLIR 绑定
```

## 小结
- **文档口径**暴露 6 页;**源码口径**再加 `tla` + `kernels`。
- 选择:常规算子 `tl.*`+`libdevice`;需控硬件加 `al`;追极致性能用 `tla`。
