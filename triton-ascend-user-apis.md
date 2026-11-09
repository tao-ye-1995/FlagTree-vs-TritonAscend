# Triton-Ascend 用户可编程 API 总览

> 基于仓库 `/workspace/triton-ascend`(版本 3.6.0)源码整理。
> 组织形式对齐在线文档「API 参考」侧边栏:<https://triton-ascend.readthedocs.io/zh-cn/latest/>
>
> 两种口径:
> - **在线文档口径** = 下面的 **页面 1~6**(官方正式暴露,侧边栏 6 项)。
> - **源码 `__all__` 口径** = 页面 1~6 **再加** 文档未收录的 **第 7 类 `tla`**(有完整导出、单测与样例,但不在侧边栏)。

---

## 分类一览

| # | 文档页 / 命名空间 | 性质 | 源码位置 |
|---|---|---|---|
| 1 | `triton` | 通用(社区原版) | [python/triton/runtime/](../triton-ascend/python/triton/runtime/) |
| 2 | `triton.language` | 通用核心 DSL | [python/triton/language/](../triton-ascend/python/triton/language/) |
| 3 | `triton.testing` | 通用 + 昇腾 `do_bench_npu` | [python/triton/testing.py](../triton-ascend/python/triton/testing.py) |
| 4 | `triton.language.extra.cann.extension` | **昇腾专属** 底层 builtin | [third_party/ascend/language/cann/extension/](../triton-ascend/third_party/ascend/language/cann/extension/) |
| 5 | `triton.language.extra.cann.libdevice` | **昇腾专属** 数学库 | [third_party/ascend/language/cann/libdevice.py](../triton-ascend/third_party/ascend/language/cann/libdevice.py) |
| 6 | `triton.extension.buffer.language` | 通用 buffer 抽象 | [python/triton/extension/buffer/language/](../triton-ascend/python/triton/extension/buffer/language/) |
| 7 | `triton.language.extra.tla` ⚠️ | **昇腾专属** 独立范式(**文档未收录**) | [third_party/ascend/language/tla/](../triton-ascend/third_party/ascend/language/tla/) |

> 打包机制:`third_party/ascend/language/<X>/` 在安装时被映射到 `triton.language.extra.<X>`(见 [setup.py:639-643](../triton-ascend/setup.py#L639)),所以磁盘路径 `cann/`、`tla/` → import 路径 `triton.language.extra.cann`、`triton.language.extra.tla`。

### 三层心智模型

1. **像写 CUDA Triton**:`tl.*`(页面 2)+ `cann.libdevice.*`(页面 5),编译器自动 tiling/lowering —— 大多数用户。
2. **手动控硬件但仍在 Triton 语义内**:加 `cann.extension`(页面 4,别名 `al`)的 builtin —— fixpipe / dot / 跨核同步 / 自定义 op。
3. **完全显式控制内存层级与流水**:用 `tla.*`(第 7 类)独立范式 —— 手动 alloc UB/L1/L0、手写 DMA、SIMD 指令、cube mmad、跨核同步。

---

## 📄 页面 1:`triton`

入口装饰器 / 配置。源码:[python/triton/runtime/jit.py](../triton-ascend/python/triton/runtime/jit.py)、[autotuner.py](../triton-ascend/python/triton/runtime/autotuner.py)

- `jit` · `autotune` · `heuristics` · `Config`

---

## 📄 页面 2:`triton.language`(核心 DSL)

源码:[python/triton/language/](../triton-ascend/python/triton/language/)(`core.py` / `standard.py` / `math.py` / `random.py`)

| 分组 | API |
|---|---|
| Programming Model | `tensor`、`tensor_descriptor`、`program_id`、`num_programs`、`map_elementwise` |
| Creation Ops | `arange`、`cat`、`full`、`zeros`、`zeros_like`、`cast` |
| Shape Manipulation | `broadcast`、`broadcast_to`、`expand_dims`、`interleave`、`join`、`permute`、`ravel`、`reshape`、`split`、`trans`、`view` |
| Linear Algebra | `dot`、`dot_scaled` |
| Memory/Pointer | `load`、`store`、`make_tensor_descriptor`、`load_tensor_descriptor`、`store_tensor_descriptor`、`make_block_ptr`、`advance` |
| Indexing | `flip`、`where`、`swizzle2d`、`gather` |
| Math | `abs`、`add`、`cdiv`、`ceil`、`clamp`、`cos`、`div`、`div_rn`、`erf`、`exp`、`exp2`、`fdiv`、`floordiv`、`floor`、`fma`、`log`、`log2`、`maximum`、`minimum`、`mod`、`mul`、`neg`、`rsqrt`、`sigmoid`、`sin`、`softmax`、`sqrt`、`sqrt_rn`、`sub`、`umulhi` |
| Logical | `and`、`or`、`xor`、`not`、`logical_and`、`logical_or`、`invert`、`lshift`、`rshift` |
| Comparison | `gt`、`ge`、`lt`、`le`、`eq`、`ne` |
| Reduction | `argmax`、`argmin`、`max`、`min`、`reduce`、`reduce_or`、`sum`、`xor_sum` |
| Scan/Sort | `associative_scan`、`bitonic_merge`、`cumprod`、`cumsum`、`histogram`、`sort`、`topk` |
| Atomic | `atomic_add`、`atomic_and`、`atomic_cas`、`atomic_max`、`atomic_min`、`atomic_or`、`atomic_xchg`、`atomic_xor` |
| Random | `randint4x`、`randint`、`rand`、`rand4x`、`randn`、`randn4x` |
| Iterators | `range`、`static_range`、`condition` |
| Compiler Hint | `assume`、`debug_barrier`、`max_constancy`、`max_contiguous`、`multiple_of` |
| Debug | `static_print`、`static_assert`、`device_print`、`device_assert` |
| Inline Assembly | `inline_asm_elementwise` |

---

## 📄 页面 3:`triton.testing`

源码:[python/triton/testing.py](../triton-ascend/python/triton/testing.py);`do_bench_npu` 来自 [third_party/ascend/backend/testing.py:347](../triton-ascend/third_party/ascend/backend/testing.py#L347)

| API | 说明 |
|---|---|
| `Benchmark`、`do_bench`、`do_bench_cudagraph`、`perf_report`、`assert_close` | 社区通用 |
| `do_bench_npu` | **昇腾专属** NPU 性能测试(另有 `do_bench_npu_profiler`、`do_bench_npu_mspti`) |

---

## 📄 页面 4:`triton.language.extra.cann.extension`(昇腾底层 builtin)

导入:`import triton.language.extra.cann.extension as al`
导出点:[cann/extension/__init__.py](../triton-ascend/third_party/ascend/language/cann/extension/__init__.py)

| 文档分组 | API | 源码文件 |
|---|---|---|
| Core Types | `scope`、`ascend_address_space`、`compile_hint`、`multibuffer`、`parallel` | `scope.py` / `core.py` / `aux_ops.py` |
| Data Movement | `copy`、`copy_from_ub_to_l1`、`fixpipe` | [core.py](../triton-ascend/third_party/ascend/language/cann/extension/core.py) |
| Synchronization | `sync_block_all`、`sync_block_set`、`sync_block_wait`、`debug_barrier` | `core.py` / `aux_ops.py` |
| Vector Operations | `sub_vec_id`、`sub_vec_num`、`conv1d`、`dot` | [core.py](../triton-ascend/third_party/ascend/language/cann/extension/core.py)(`dot`@713、`conv1d`@512) |
| Enums | `PIPE`、`MODE`、`CORE`、`SYNC_IN_VF`、`IteratorType`、`FixpipeDMAMode`、`FixpipeDualDstMode`、`FixpipePreQuantMode`、`FixpipePreReluMode` | [core.py](../triton-ascend/third_party/ascend/language/cann/extension/core.py) |
| Vector/Memory Ext | `extract_slice`、`insert_slice`、`get_element`、`sort`、`flip`、`cast`、`index_put`、`gather_out_to_ub`、`scatter_ub_to_out`、`index_select_simd` | [vec_ops.py](../triton-ascend/third_party/ascend/language/cann/extension/vec_ops.py) / [mem_ops.py](../triton-ascend/third_party/ascend/language/cann/extension/mem_ops.py) |
| Custom Ops | `custom`、`custom_semantic`、`register_custom_op` | [custom_op.py](../triton-ascend/third_party/ascend/language/cann/extension/custom_op.py) |
| IR Affine Types | `affine_expr`、`affine_constant_expr`、`affine_dim_expr`、`affine_symbol_expr`、`affine_binary_op_expr`、`affine_map` | MLIR 绑定(透传 `libtriton.ascend.ir`) |

> 源码 `__all__` 另有、但**在线文档未收录**:`conv2d`、`atan2`、`isfinited`、`finitef`、`SyncEventSlot`、`AddressSpace`、`DataLayout`、`int64`、`EVENT_ID`、`SYNC_HINT`、`builtin`、`is_builtin`、`CORE`、`DataLayout`。

---

## 📄 页面 5:`triton.language.extra.cann.libdevice`(昇腾数学库)

导入:`import triton.language.extra.cann.libdevice as libdevice`
源码:[cann/libdevice.py](../triton-ascend/third_party/ascend/language/cann/libdevice.py)。文档单组 **Libdevice Ops**(~180 个),按功能归纳:

| 子类 | 代表 API |
|---|---|
| 基础数学 | `abs`、`ceil`、`floor`、`round`、`trunc`、`fma`、`pow`、`sqrt`、`rsqrt`、`exp`、`exp2`、`exp10`、`expm1`、`log`、`log2`、`log10`、`log1p`、`logb`、`ilogb`、`cbrt`、`rcbrt`、`hypot`、`rhypot` |
| 三角 / 双曲 | `sin`、`cos`、`tan`、`asin`、`acos`、`atan`、`atan2`、`sinh`、`cosh`、`tanh`、`asinh`、`acosh`、`atanh`、`sinpi`、`cospi` |
| 带舍入模式 | `add_rn/rz/rd/ru`、`sub_*`、`mul_*`、`div_*`、`fma_*`、`rcp_*`、`sqrt_*`、`rsqrt_rn` |
| 类型转换 | `float2int_*`、`int2float_*`、`float2uint_*`、`uint2float_*`、`float2ll_*`、`ll2float_*`、`float2ull_*`、`ull2float_*`、`float2half_rn`、`half2float`、`float_as_int`、`int_as_float`、`float_as_uint`、`uint_as_float` |
| 快速近似 | `fast_expf`、`fast_exp10f`、`fast_logf`、`fast_log2f`、`fast_log10f`、`fast_powf`、`fast_sinf`、`fast_cosf`、`fast_tanf`、`fast_tanhf`、`fast_dividef` |
| 特殊函数 | `erf`、`erfc`、`erfcx`、`erfinv`、`erfcinv`、`gamma`、`tgamma`、`lgamma`、`normcdf`、`normcdfinv`、`j0`/`j1`/`jn`、`y0`/`y1`/`yn`、`cyl_bessel_i0`/`i1`、`norm3d`/`norm4d`、`rnorm3d`/`rnorm4d` |
| 判定 / 位运算 | `isinf`、`isnan`、`finitef`、`signbit`、`clz`、`popc`、`ffs`、`brev`、`byte_perm`、`mulhi`、`umulhi`、`mul24`、`umul24`、`sad`、`usad` |
| 其他 | `relu`、`reciprocal`、`max`、`min`、`copysign`、`nextafter`、`fmod`、`remainder`、`saturatef`、`ldexp`、`scalbn`、`hadd`、`rhadd`、`uhadd`、`urhadd`、`nan`、`nearbyint`、`rint`、`llrint`、`llround`、`fdim` |

---

## 📄 页面 6:`triton.extension.buffer.language`(通用 buffer 抽象,非昇腾专属)

导入:`import triton.extension.buffer.language`
源码:[python/triton/extension/buffer/language/core.py](../triton-ascend/python/triton/extension/buffer/language/core.py)

| 文档分组 | API | 源码行 |
|---|---|---|
| Core Types | `buffer`、`buffer_type`、`address_space` | [core.py:138 / 78 / 68](../triton-ascend/python/triton/extension/buffer/language/core.py#L138) |
| Memory Management | `alloc`、`to_buffer`、`to_tensor`、`subview` | [core.py:191 / 207 / 220 / 283](../triton-ascend/python/triton/extension/buffer/language/core.py#L191) |

> 另导出 helper(文档未列):`builtin`、`is_builtin`。
> ⚠️ 注意与第 7 类 `tla` 的 `buffer`/`subview`/`alloc` **同名但不同模块、不同范式**:本页是 upstream 通用抽象;`tla` 是昇腾显存层级专属。

---

## ⚠️ 第 7 类:`triton.language.extra.tla`(TLA,文档未收录)

> **TLA = Triton Language for Ascend**。独立于 `al`/`libdevice` 的另一套编程范式,直接暴露昇腾内存层级(GM/UB/L1/L0/BT/FB/L0C)与 SIMD 向量寄存器 / cube 核。
> **在线文档侧边栏没有这一页**,但源码有完整 `__all__`、单测([third_party/ascend/unittest/](../triton-ascend/third_party/ascend/unittest/))与样例大量使用。

导入:`import triton.language.extra.tla as tla`
导出点:[tla/__init__.py](../triton-ascend/third_party/ascend/language/tla/__init__.py)

| 分组 | API | 源码文件 |
|---|---|---|
| Kernel 装饰器 | `@tla.jit` | [jit.py](../triton-ascend/third_party/ascend/language/tla/jit.py) |
| Buffer / 类型 | `alloc`、`alloc_buffer`、`buffer`、`buffer_type`、`vector`、`vector_type` | [alloc.py](../triton-ascend/third_party/ascend/language/tla/alloc.py) / [core.py](../triton-ascend/third_party/ascend/language/tla/core.py) |
| View | `view`、`subview`、`reshape` | [view.py](../triton-ascend/third_party/ascend/language/tla/view.py) |
| Copy | `copy`(GM↔buffer) | [copy.py](../triton-ascend/third_party/ascend/language/tla/copy.py) |
| DMA 搬运 | `load_gm_to_ub`、`store_ub_to_gm`、`load_gm_to_l1`、`copy_ub_to_l1`、`copy_l1_to_ub`、`copy_l1_to_bt`、`copy_l1_to_fb`、`copy_l1_to_l0`、`store_l0c_to_gm`、`copy_l0c_to_ub`、`copy_l0c_to_l1`、`set_ctrl` | [dma.py](../triton-ascend/third_party/ascend/language/tla/dma.py) |
| DMA 参数类 | `LoadGM2UBParams`、`StoreUB2GMParams`、`LoadGM2L1Params`、`CopyUB2L1Params`、`CopyL12UBParams`、`CopyL12BTParams`、`CopyL12FBParams`、`CopyL1ToL0Params`、`FixpipeL0C2GMParams`、`FixpipeL0C2L1Params`、`FixpipeL0C2UBParams` | [dma.py](../triton-ascend/third_party/ascend/language/tla/dma.py) |
| DMA 枚举 | `PadMode`、`EvictionPolicy`、`AtomicKind`、`UnitFlagMode`、`FixpipeDMAMode`、`FixpipeDualDstMode`、`FixpipeSubBlock`、`FixpipePreQuantMode`、`FixpipePreReluMode` | [dma.py](../triton-ascend/third_party/ascend/language/tla/dma.py) |
| SIMD 向量指令(`tla.simd.*`) | `vlds`、`vsts`、`vsstb`、`vadd`、`vadds`、`vaddc`、`vsub`、`vmul`、`vdiv`、`vmod`、`vmax`、`vmin`、`vmaxs`、`vmins`、`vfma`、`vexp`、`vsqrt`、`vrsqrt`、`vabs`、`vneg`、`vnot`、`vrelu`、`vlrelu`、`vprelu`、`vand`/`vor`/`vxor`、`vshl`/`vshls`/`vshr`、`vsel`、`vdup`/`vdups`/`vdupv`、`vextf`、`vtruncf`、`vbitcast`、`vintlv`/`vdintlv`、`vpack`、`vbroadcast`、`vfull`、`vsplat`、`vci`、`vreduce`/`vreduce_sum`/`vreduce_max`/`vreduce_min`、`vcmax`/`vcmin`、`vcmp`、`vabsdif`/`vexpdif`、`pge`、`membar`;枚举 `MODE`/`CMP`/`POS` | [simd.py](../triton-ascend/third_party/ascend/language/tla/simd.py) |
| Cube 核(`tla.cube.*`) | `mmad`(矩阵乘,L1/L0C 路由) | [cube.py](../triton-ascend/third_party/ascend/language/tla/cube.py) |
| 作用域 | `scope`(`tla.scope.simd()` / `tla.scope.cube()`) | [scope.py](../triton-ascend/third_party/ascend/language/tla/scope.py) |
| 同步(`tla.sync.*`) | `barrier`、`set_flag`、`wait_flag`、`sync_all`、`block_set_flag`、`block_wait_flag`、`sub_block_id` | [sync.py](../triton-ascend/third_party/ascend/language/tla/sync.py) |
| 选择 | `where`(eager 标量/SIMD select) | [selection.py](../triton-ascend/third_party/ascend/language/tla/selection.py) |
| 数学 | `math`、`compensated_rstd`、`truncmod` | [math.py](../triton-ascend/third_party/ascend/language/tla/math.py) |
| 调试 | `device_assert`、`device_print` | [debug.py](../triton-ascend/third_party/ascend/language/tla/debug.py) |
| 辅助 | `annotate_buffer_versions`、`assume` | [aux_ops.py](../triton-ascend/third_party/ascend/language/tla/aux_ops.py) |
| 标量 dtype(复用 triton) | `bfloat16`、`float16`、`float32`、`int8/16/32/64`、`uint8/16/32/64`、`dtype` | 透传 `triton.language.core` |

---

## 典型导入模板

```python
import triton
import triton.language as tl
from triton.backends.ascend import _apply_ascend_patch          # 必备:激活昇腾补丁
import triton.language.extra.cann.libdevice as libdevice        # 页面 5:数学函数
import triton.language.extra.cann.extension as al               # 页面 4:底层 builtin
import triton.language.extra.tla as tla                         # 第 7 类:TLA 范式(文档外)
from triton._C.libtriton.ascend import ir as ascend_ir          # MLIR 绑定
```

---

## 小结

- **以在线文档为准**:官方暴露 = 页面 1~6(4 页通用 Triton + 2 页昇腾 `cann` 扩展),已全量覆盖。
- **以源码 `__all__` 为准**:再加文档未收录的 `tla`(第 7 类)——完整的第三套编程范式。
- 选择建议:常规算子用 `tl.*`+`libdevice`;需控硬件加 `al`;追极致性能且愿意手写内存层级/流水用 `tla`。
