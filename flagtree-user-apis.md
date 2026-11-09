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
| **`triton.experimental.tle`** | TLE 显式内存/流水原语(含 `dsa.ascend`) | **FlagTree 独有**(TA 用 `tla` 表达近义能力) | [experimental/tle/](../FlagTree/python/triton/experimental/tle/) |
| **`triton.language.extra.tla`** | TLA 显存层级编程 | **❌ FlagTree 不存在**(近义能力在 TLE) | — |

## 三层心智模型(FlagTree 昇腾)

1. **像写 CUDA Triton**:`tl.*` + `cann.libdevice.*` —— 大多数用户(主力路径)。
2. **手动控硬件、仍在 Triton 语义内**:加 `cann.extension`(`al`)的 builtin —— `fixpipe`/`copy`/跨核同步/`custom`。⚠️ **无 `dot`/`conv2d` builtin**(矩阵乘走 `tl.dot` + 后端 lowering)。
3. **显式内存层级编程**:FlagTree 用 **TLE 扩展原语**(`triton.experimental.tle`,含 `dsa.ascend.*`)承担 Triton-Ascend `tla` 的角色 —— 显式 UB/L1/L0、tile 搬运、跨核同步、流水调度。`extension.buffer.language`(`alloc`/`subview`/`to_buffer`)提供更轻量的 buffer 级控制。详见下文「TLE 扩展原语」章。

---

## 各命名空间要点

### `triton` / `triton.language` / `triton.testing`(通用)
与 Triton-Ascend 基本一致(基于 Triton 3.5.1)。昇腾扩展算子通过 [flagtree_spec.py:64](../FlagTree/python/triton/flagtree_spec.py#L64) `bind_language_extension_symbols_to_tl` 把 `cann.extension.__all__` 里 upstream `tl` 尚无的符号注入 `tl.*`,使扩展算子"像内置一样"可用。

### `cann.libdevice`(精简版)
- **仅 39 个** `@core.extern` 自实现:`acos/acosh/asin/asinh/atan/atan2/atanh/copysign/cosh/cyl_bessel_i0/div_rz/erfinv/expm1/fast_dividef/fast_expf/float_as_int/fmod/gamma/hypot/ilogb/isinf/isnan/ldexp/lgamma/log10/log1p/nearbyint/nextafter/pow/reciprocal/relu/round/signbit/sinh/tan/tanh/trunc`。
- **~20 个 rebind 到 upstream `triton.language.math.*`**(在 [cann/__init__.py](../FlagTree/third_party/ascend/language/cann/__init__.py)):`abs/ceil/cos/div_rn/erf/exp/exp2/fdiv/floor/fma/log/log2/rsqrt/sin/sqrt/sqrt_rn/umulhi`,以及 `atan2/isfinited/finitef/flip` 指向 extension。
- **缺少** Triton-Ascend 全量 libdevice 里的大量函数:带舍入模式的 `add_r*/mul_r*/sub_r*/fma_r*/rcp_r*/sqrt_r*`、类型转换 `float2int_*/int2float_*/float2ll_*` 系列、特殊函数 `erfc/erfcx/j0/j1/jn/y0/y1/yn/cyl_bessel_i1/norm3d/normcdf/tgamma`、位运算 `clz/popc/ffs/brev/byte_perm/mulhi/sad` 等。

> **"rebind" 含义**:FlagTree 的 `libdevice.py` 里没有 `def exp` 等定义,这些名字是在 [cann/__init__.py](../FlagTree/third_party/ascend/language/cann/__init__.py) 执行时用 `libdevice.exp = math.exp` **把别处已有的实现贴到 `libdevice.xxx` 名字上**(重新绑定模块属性)。所以 FlagTree 的 libdevice 是个**聚合壳**:39 个自实现 + ~17 个来自 upstream `math.*` + 4 个来自 `cann.extension`。对用户调用方式和结果无感知,但**底层实现来源与 Triton-Ascend 的同名 extern 不同**,边界/精度/特殊 dtype 下可能有细微差异;其中 `atan2` 的 rebind 还受 `triton_enable_libdevice_simt()` 开关控制。完整展开见 [triton-ascend-vs-flagtree.md](triton-ascend-vs-flagtree.md) §2.3。

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

## TLE 扩展原语(`triton.experimental.tle`,FlagTree 独有)

TLE(**T**ensor-memory **L**anguage **E**xtension)是 FlagTree 在标准 `tl.*` 之上提供的一套**显式内存/流水编程原语**,独立命名空间 `triton.experimental.tle`(导入 `import triton.experimental.tle as tle`)。它最初为 NVIDIA Hopper 的 SMEM/TMA 设计(见 [tle.md](../FlagTree/tle.md)),但**含完整的昇腾(DSA/Ascend)实现**,因此与 Triton-Ascend 的 `tla` 范式在定位上高度对应。

> 源码:[python/triton/experimental/tle/](../FlagTree/python/triton/experimental/tle/)。
> 与 Triton-Ascend 的对应关系(是否存在相同/类似 API)逐条见 [user-apis-flagtree-vs-triton-ascend.md](user-apis-flagtree-vs-triton-ascend.md) 表 4。

TLE 分三层:

### 1. 通用层(`tle.*` 顶层)—— 面向 GPU SMEM/TMA 为主
- `tle.load`:带 `tt.load.async` 等属性的加载(Hopper 异步拷贝)。
- `tle.scope(core_mode="cube"|"vector")`:核模式作用域(`with tle.scope(...)`)。
- `tle.tensor_view` / `tle.tensor_view_type` / `tle.make_partition_view` / `tle.make_strided_view` / `tle.make_gather_scatter_view`:张量视图构造。
- `tle.tile_load` / `tle.tile_store`:tile 级加载/存储。
- `tle.pipe`(`capacity/scope/readers/one_shot`):流水管道原语。
- `tle.local_ptr(buffer, indices)`(文档重点):从显式索引张量物化任意形状的 SMEM 指针视图(Hopper 专属,昇腾无 SMEM 概念)。
- `tle.smem` / `tle.tmem`:共享内存 / tensor 内存作用域标记(GPU 专属)。

### 2. DSA 通用层(`tle.dsa.*`)—— 显式 buffer/pipeline/tile
- buffer/数据流:`alloc` / `copy` / `to_tensor` / `to_buffer` / `subview` / `extract_slice` / `insert_slice` / `extract_element` / `workspace`。
- 流水/并行:`pipeline` / `parallel` / `hint`。
- 算术:`add` / `sub` / `mul` / `div` / `max` / `min`。
- **tile 级原语**:`tile_alloc` / `tile_copy` / `tile_subview` / `tile_to_tensor` / `tile_set_flag` / `tile_wait_flag` / `tile_pipe_barrier` / `tile_gm_offset` / `tile_cube_launch` / `tile_cube_wait`。

### 3. DSA 昇腾层(`tle.dsa.ascend.*`)—— 昇腾专属,对应 TA 的 `al`/`tla`
- 内存层级常量:`UB` / `L1` / `L0A` / `L0B` / `L0C` / `PIPE`。
- 跨核同步:`sync_block_set` / `sync_block_wait` / `sync_block_all` / `sub_vec_id` / `sub_vec_num`。
- 编译/缓冲:`compile_hint` / `multibuffer` / `raw` / `SyncSpec`。
- **流水调度器**(TA 无直接对应):`pipe_scheduler` / `PipeScheduler` / `pipe_reader` / `pipe_writer` / `pipe_slot` / `pipe_value` / `pipe_wait_result` / `run_pipe` / `reset_pipe_event_allocator`。
- **分布式/通信**(TA 无对应):`device_mesh` / `MeshConfig` / `remote` / `shard_id` / `distributed_barrier` / `swizzle2d_Nz` / `gemm_swizzle2d_Nz`。
- **自定义高性能算子**(`tle.dsa.ascend.custom_ops`):`gather_gm_to_ub` / `gather_gm_to_l1` / `sort32` / `mrgsort` / `unpack_sort` / `cast_int4_to_fp16` / `pair_reduce_sum_continuous_mask` / `duplicate_bitwise_mask` 等。

### 与 Triton-Ascend 的关系(概要)
- **有同名/近义 API**(接口层面差异):`dsa.ascend` 的 `sub_vec_id`/`sub_vec_num`/`sync_block_*`/`compile_hint`/`multibuffer`/`UB`/`L1`/`L0*`/`PIPE` → TA 的 `al.*` 几乎同名;`dsa` 的 `alloc`/`copy`/`subview`/`tile_*` → TA 的 `tla.*`(概念对应,接口名不同,如 TLE `tile_alloc` ≈ TA `tla.alloc_buffer`)。
- **TA 有、TLE 风格不同**:TA 的 `tla.simd.v*` 向量指令、`tla.cube.mmad` → TLE 用 `tile_*` + `tile_cube_launch` 表达。
- **TLE 有、TA 无对应**:`pipe`/`pipe_scheduler` 流水调度器族、`device_mesh`/`remote` 分布式通信族、Hopper SMEM/TMA 的 `local_ptr`/`smem`/`tmem`。
- **定位差异**:TLE 是**跨 GPU/NPU 的统一显式内存语言**(一套原语多后端);TA 的 `tla` 是**纯昇腾的显式内存范式**。

---

## 典型导入模板

```python
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice        # 数学函数(精简)
import triton.language.extra.cann.extension as al               # 底层 builtin(无 dot/conv2d)
import triton.experimental.tle as tle                           # TLE 扩展原语(含 dsa.ascend)
# 无 tla(Triton-Ascend 专属)
```

## 小结
- FlagTree 昇腾用户 API ≈ Triton-Ascend **去掉 `tla`、去掉 `dot`/`conv2d` builtin、libdevice 精简**后的子集,**另加 TLE 扩展原语**(`triton.experimental.tle`)。
- 设计取向:**多后端通用 + 昇腾精简接入**,复用 upstream 数学实现,接口面小;显式内存/流水编程由跨后端的 TLE 承担(对应 TA 的 `tla`)。
- 若需矩阵乘:用 `tl.dot`(经后端 lowering),而非 `al.dot` builtin。
