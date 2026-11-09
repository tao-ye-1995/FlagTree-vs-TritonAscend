# FlagTree 用户可编程 API 概要(昇腾后端)

> 仓库 `/workspace/FlagTree`(Triton 3.5.1,多后端统一框架,Ascend 为其一)。
> 本文只覆盖 **昇腾后端** 用户可见的 API,结构对齐 [triton-ascend-user-apis.md](triton-ascend-user-apis.md) 以便对照。
> 逐 API 的两仓对照见 [user-apis-flagtree-vs-triton-ascend.md](user-apis-flagtree-vs-triton-ascend.md)。

## 定位差异(与 Triton-Ascend 的最大不同)

FlagTree 是 **多后端统一框架**(`third_party/` 下有 amd / nvidia / enflame / iluvatar / mthreads / xpu / tsingmicro / **ascend** / tle),昇腾只是其中一个后端。因此:
- 没有在线 API 参考文档专门面向昇腾(不像 Triton-Ascend 有 readthedocs 的 6 页)。
- 昇腾扩展通过 **`flagtree_spec`** 机制注入,而非专用 fork 的 monkey-patch。
- 能复用 upstream 的部分(数学函数等)尽量 rebind,不重复实现 —— 接口暴露面更小。

## 命名空间全景

| 命名空间 | 定位 | 对比 TA | 源码 |
|---|---|---|---|
| `triton` | 入口装饰器/配置 | 一致 | [python/triton/runtime/](../FlagTree/python/triton/runtime/) |
| `triton.language`(`tl`) | 核心 DSL | 一致(3.5.1) | [python/triton/language/](../FlagTree/python/triton/language/) |
| `triton.testing` | 性能测试 | 一致(含 `do_bench_npu`) | [testing.py](../FlagTree/python/triton/testing.py) |
| `triton.language.extra.cann.libdevice` | 昇腾数学库 | **精简**(39 自实现 + 多处 rebind upstream) | [cann/libdevice.py](../FlagTree/third_party/ascend/language/cann/libdevice.py) |
| `triton.language.extra.cann.extension`(`al`) | 昇腾底层 builtin | **少 `dot`/`conv2d`**;多 `custom_object_semantic` | [cann/extension/](../FlagTree/third_party/ascend/language/cann/extension/) |
| `triton.extension.buffer.language` | 通用 buffer 抽象 | 源码逐函数一致 | [spec/ascend/extension/buffer/language/](../FlagTree/python/triton/spec/ascend/extension/buffer/language/) |
| `triton.language.extra.cann.kernels` | 现成 kernel(`gather_2d_simd`) | **完全一致** | [kernels/gather.py](../FlagTree/third_party/ascend/language/kernels/gather.py) |
| **`triton.language.extra.tla`** | TLA 显存层级编程 | **❌ FlagTree 不存在** | — |

## 三层心智模型(FlagTree 昇腾)

1. **像写 CUDA Triton**:`tl.*` + `cann.libdevice.*` —— 大多数用户(主力路径)。
2. **手动控硬件、仍在 Triton 语义内**:加 `cann.extension`(`al`)的 builtin —— `fixpipe`/`copy`/跨核同步/`custom`。⚠️ **无 `dot`/`conv2d` builtin**(矩阵乘走 `tl.dot` + 后端 lowering)。
3. **显式内存层级编程**:**无 TLA 对应物**。`extension.buffer.language`(`alloc`/`subview`/`to_buffer`)提供有限的 buffer 级控制,但不构成 TLA 那样的完整 SIMD/cube/DMA 范式。

---

## 各命名空间要点

### `triton` / `triton.language` / `triton.testing`(通用)
与 Triton-Ascend 基本一致(基于 Triton 3.5.1)。昇腾扩展算子通过 [flagtree_spec.py:64](../FlagTree/python/triton/flagtree_spec.py#L64) `bind_language_extension_symbols_to_tl` 把 `cann.extension.__all__` 里 upstream `tl` 尚无的符号注入 `tl.*`,使扩展算子"像内置一样"可用。

### `cann.libdevice`(精简版)
- **仅 39 个** `@core.extern` 自实现:`acos/acosh/asin/asinh/atan/atan2/atanh/copysign/cosh/cyl_bessel_i0/div_rz/erfinv/expm1/fast_dividef/fast_expf/float_as_int/fmod/gamma/hypot/ilogb/isinf/isnan/ldexp/lgamma/log10/log1p/nearbyint/nextafter/pow/reciprocal/relu/round/signbit/sinh/tan/tanh/trunc`。
- **~20 个 rebind 到 upstream `triton.language.math.*`**(在 [cann/__init__.py](../FlagTree/third_party/ascend/language/cann/__init__.py)):`abs/ceil/cos/div_rn/erf/exp/exp2/fdiv/floor/fma/log/log2/rsqrt/sin/sqrt/sqrt_rn/umulhi`,以及 `atan2/isfinited/finitef/flip` 指向 extension。
- **缺少** Triton-Ascend 全量 libdevice 里的大量函数:带舍入模式的 `add_r*/mul_r*/sub_r*/fma_r*/rcp_r*/sqrt_r*`、类型转换 `float2int_*/int2float_*/float2ll_*` 系列、特殊函数 `erfc/erfcx/j0/j1/jn/y0/y1/yn/cyl_bessel_i1/norm3d/normcdf/tgamma`、位运算 `clz/popc/ffs/brev/byte_perm/mulhi/sad` 等。

### `cann.extension`(`al`,底层 builtin)
与 Triton-Ascend **高度重合**,差异:
- **缺 `dot`、`conv2d`**(TA 独有的 cube 矩阵乘 / 2D 卷积 builtin)。
- **缺** `SyncEventSlot`/`EVENT_ID`/`SYNC_HINT`/`AddressSpace`/`DataLayout`/`is_register_based_arch`。
- **独有** `custom_object_semantic`(自定义 op 对象语义变体)。
- 共有:`copy`/`copy_from_ub_to_l1`/`fixpipe`/`conv1d`/`scope`/`parallel`/`compile_hint`/`multibuffer`/`sync_block_*`/`sub_vec_id|num`/`insert_slice`/`extract_slice`/`get_element`/`sort`/`flip`/`cast`/`index_put`/`gather_out_to_ub`/`scatter_ub_to_out`/`index_select_simd`/`custom`/`custom_semantic`/`register_custom_op`/`atan2`/`isfinited`/`finitef`/ affine 系列 / 枚举。

### `extension.buffer.language`(通用 buffer)
`buffer`/`buffer_type`/`address_space` + `alloc`/`to_buffer`/`to_tensor`/`subview`。源码与 Triton-Ascend **逐函数一致**,只是放在 [spec/ascend/extension/buffer/language/](../FlagTree/python/triton/spec/ascend/extension/buffer/language/)(多后端命名空间分发,`spec_path` 机制)。

### `kernels.gather_2d_simd`
与 Triton-Ascend 源码**完全一致**。

---

## 典型导入模板

```python
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice        # 数学函数(精简)
import triton.language.extra.cann.extension as al               # 底层 builtin(无 dot/conv2d)
# 无 tla
```

## 小结
- FlagTree 昇腾用户 API ≈ Triton-Ascend **去掉 `tla`、去掉 `dot`/`conv2d` builtin、libdevice 精简**后的子集。
- 设计取向:**多后端通用 + 昇腾精简接入**,复用 upstream 数学实现,接口面小。
- 若需矩阵乘:用 `tl.dot`(经后端 lowering),而非 `al.dot` builtin。
