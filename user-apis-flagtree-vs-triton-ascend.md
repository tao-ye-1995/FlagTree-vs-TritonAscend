# 用户层 API 对照:Triton-Ascend vs FlagTree(昇腾)

> 仓库:`/workspace/triton-ascend`(TA,Triton 3.6.0) · `/workspace/FlagTree`(FT,Triton 3.5.1)。
> 概要见 [triton-ascend-user-apis.md](triton-ascend-user-apis.md) / [flagtree-user-apis.md](flagtree-user-apis.md) / [triton-ascend-vs-flagtree.md](triton-ascend-vs-flagtree.md)。
>
> 硬件列说明:**A2/A3** = Atlas A2 / A3(`Ascend910B`/`910_93`,LEGACY 代;代码里 `_is_a3_target`,A2/A3 共享 libdevice ABI);**950** = Ascend 950PR&950DT(`Ascend910_95`/`Ascend950`,GEN5 / register-based;代码里 `_is_a5_target` / `is_register_based_arch`)。
> **「仅 950」= A2/A3 调用会 `RuntimeError`/编译期报错**(register-based 专属的 L0C/L1 片上搬运等)。标 **A2/A3 · 950** = 三者硬件均支持。
> **SIMT 注记**:libdevice 多数函数带 `if SIMT: unsupported` 守卫,但 **SIMT 是「950 + 环境开关 `TRITON_ENABLE_LIBDEVICE_SIMT`」才进入的模式,不是某代芯片限制**;默认(非 SIMT)下这些函数 A2/A3/950 全支持,故硬件列仍记 A2/A3 · 950,SIMT 限制在功能列注明。
> 支持列:✅ 支持 · ❌ 不暴露 · ⚠️ 以 rebind/间接方式提供。
> 范围:**表 1** 聚焦昇腾专属扩展 API(`cann.extension` / `cann.libdevice` / `tla` / `extension.buffer.language` / `kernels`);**表 2** 源码路径对照;**表 3** 通用 `tl.*`(两仓一致,逐一列出);**表 4** FlagTree TLE 扩展原语 → Triton-Ascend 对应情况。

---

## 表 1 · 接口支持与功能对照

### 1.1 `cann.extension`(`al`,底层 builtin)

| 接口名 | 适用硬件 | TA 支持 | FT 支持 | 功能 |
|---|---|---|---|---|
| `al.dot` | A2/A3 · 950 | ✅ | ❌ | cube 矩阵乘 `D=A*B`,支持 ND / fractal(zN)layout |
| `al.conv1d` | A2/A3 · 950 | ✅ | ✅ | 1D 卷积 builtin |
| `al.conv2d` | A2/A3 · 950 | ✅ | ❌ | 2D 卷积 builtin |
| `al.fixpipe` | **仅 950** | ✅ | ✅ | L0C→UB 固定流水搬运(含量化/ReLU 后处理)。非 register-based 架构 `RuntimeError` |
| `al.copy` | **仅 950** | ✅ | ✅ | 片上内存间拷贝(L0C/L1 register-based)。A2/A3 调用 `RuntimeError` |
| `al.copy_from_ub_to_l1` | **仅 950** | ✅ | ✅ | UB→L1 拷贝(已弃用,改用 `copy`)。A2/A3 调用 `RuntimeError` |
| `al.scope` | A2/A3 · 950 | ✅ | ✅ | 核模式作用域(vector/cube) |
| `al.parallel` | A2/A3 · 950 | ✅ | ✅ | 并行循环标注 |
| `al.compile_hint` | A2/A3 · 950 | ✅ | ✅ | 编译提示 |
| `al.multibuffer` | A2/A3 · 950 | ✅ | ✅ | 多缓冲(double-buffer)标注 |
| `al.sync_block_all` | A2/A3 · 950 | ✅ | ✅ | 跨核块同步(全) |
| `al.sync_block_set` | A2/A3 · 950 | ✅ | ✅ | 跨核同步 set flag |
| `al.sync_block_wait` | A2/A3 · 950 | ✅ | ✅ | 跨核同步 wait flag |
| `al.debug_barrier` | A2/A3 · 950 | ✅ | ✅ | 调试屏障 |
| `al.sub_vec_id` | A2/A3 · 950 | ✅ | ✅ | 子向量核 id |
| `al.sub_vec_num` | A2/A3 · 950 | ✅ | ✅ | 子向量核数量 |
| `al.insert_slice` | A2/A3 · 950 | ✅ | ✅ | 片上 tensor 插入子块 |
| `al.extract_slice` | A2/A3 · 950 | ✅ | ✅ | 片上 tensor 提取子块 |
| `al.get_element` | A2/A3 · 950 | ✅ | ✅ | 取标量元素 |
| `al.sort` | A2/A3 · 950 | ✅ | ✅ | 片上排序 |
| `al.flip` | A2/A3 · 950 | ✅ | ✅ | 维度翻转 |
| `al.cast` | A2/A3 · 950(fp8/fp64 仅 950) | ✅ | ✅ | 显式类型转换。基础 dtype 全支持;**fp8/fp64 转换仅 950**(A2/A3 `ValueError`),saturate 路径按架构不同 |
| `al.index_put` | A2/A3 · 950 | ✅ | ✅ | 索引写入 |
| `al.gather_out_to_ub` | A2/A3 · 950 | ✅ | ✅ | GM→UB gather |
| `al.scatter_ub_to_out` | A2/A3 · 950 | ✅ | ✅ | UB→GM scatter |
| `al.index_select_simd` | A2/A3 · 950 | ✅ | ✅ | SIMD 索引选择 |
| `al.custom` | A2/A3 · 950 | ✅ | ✅ | 自定义 MLIR op 调用 |
| `al.custom_semantic` | A2/A3 · 950 | ✅ | ✅ | 自定义 op 语义构造 |
| `al.custom_object_semantic` | A2/A3 · 950 | ❌ | ✅ | 自定义 op 对象语义变体(FT 独有) |
| `al.register_custom_op` | A2/A3 · 950 | ✅ | ✅ | 注册自定义 op |
| `al.SyncEventSlot` | A2/A3 · 950 | ✅ | ❌ | 同步事件槽(配合 custom op) |
| `al.atan2` | A2/A3 · 950 | ✅ | ✅ | atan2(扩展数学) |
| `al.isfinited` | A2/A3 · 950 | ✅ | ✅ | 是否有限(double) |
| `al.finitef` | A2/A3 · 950 | ✅ | ✅ | 是否有限(float) |
| `al.affine_map` 等 | A2/A3 · 950 | ✅ | ✅ | MLIR affine 绑定(expr/dim/symbol/map) |
| 枚举 `PIPE/MODE/CORE/IteratorType/Fixpipe*` | A2/A3 · 950 | ✅ | ✅ | 流水/模式/核/迭代器/fixpipe 枚举 |
| `al.AddressSpace` / `al.DataLayout` | A2/A3 · 950 | ✅ | ❌ | 地址空间 / 数据布局枚举 |
| `al.EVENT_ID` / `al.SYNC_HINT` | A2/A3 · 950 | ✅ | ❌ | 事件 id / 同步提示枚举 |
| `al.is_register_based_arch` | 950 | ✅ | ❌ | 判断是否 register-based(A5/950) |

### 1.2 `cann.libdevice`(数学库)

> TA **全量自实现**(~180);FT **精简**:39 个自实现 + ~17 个 rebind 到 upstream `math.*` + 4 个 rebind 到 `cann.extension`。
> FT「⚠️」= **rebind**:FlagTree 的 `libdevice.py` 无该 `def`,而在 [cann/__init__.py](../FlagTree/third_party/ascend/language/cann/__init__.py) 用 `libdevice.xxx = math.xxx`(或 `= extension.xxx`)把别处已有实现贴到该名字上。**调用方式与结果对用户无感知,但底层实现来源与 TA 的同名 extern 不同**,边界/精度/特殊 dtype 下可能有细微差异(`atan2` 还受 `triton_enable_libdevice_simt()` 开关控制)。完整说明见 [triton-ascend-vs-flagtree.md](triton-ascend-vs-flagtree.md) §2.3「rebind 到底是什么含义」。
>
> **硬件/SIMT 说明(逐个核查结论)**:libdevice 全部函数在 **A2/A3/950 硬件均可用**——没有任何一个是「某代芯片禁用」。按架构分三种内部行为:(a) **大量函数带 `if SIMT: static_assert(False)` 守卫**(带舍入的 `add_r*/mul_r*/sqrt_r*/rcp_r*/fma_r*`、类型转换 `float2*/`*`2float` 族、特殊函数 `erfc/j0/jn/normcdf` 等、位运算 `clz/popc/brev/mul24` 等)→ 仅 **SIMT 模式**(950+环境开关)下不可用,**普通模式 A2/A3/950 全支持**;(b) `mulhi`/`hadd`/`rhadd` 在 950 走不同实现(两路都可用);(c) `reciprocal`/`exp`/`sqrt` 等的 **fp32 在 950/A6 走优化路径**(非 fp32 与其他架构仍走通用路径)。因此下表硬件列统一记 **A2/A3 · 950**,SIMT 限制见本注。

| 接口名 | 适用硬件 | TA 支持 | FT 支持 | 功能 |
|---|---|---|---|---|
| `reciprocal` `log1p` `relu` `isinf` `isnan` `tan` `tanh` `atan` `ilogb` `ldexp` `pow` `fmod` `expm1` `hypot` `gamma` `lgamma` `erfinv` `round` `trunc` `nearbyint` `nextafter` `signbit` `copysign` `sinh` `cosh` `acos` `acosh` `asin` `asinh` `atanh` `atan2` `log10` `cyl_bessel_i0` `fast_expf` `fast_dividef` `float_as_int` `div_rz` | A2/A3 · 950 | ✅ | ✅ | 两仓均自实现的核心 extern 数学函数 |
| `abs` `ceil` `floor` `cos` `sin` `exp` `exp2` `log` `log2` `sqrt` `sqrt_rn` `rsqrt` `fma` `fdiv` `div_rn` `erf` `umulhi` `flip` `isfinited` `finitef` | A2/A3 · 950 | ✅ | ⚠️ | FT 经 rebind upstream `math.*` / extension 提供 |
| `add_rd/rn/ru/rz` `sub_r*` `mul_r*` `fma_r*` `rcp_r*` `sqrt_rd/ru/rz` | A2/A3 · 950 | ✅ | ❌ | 带舍入模式的算术(FT 未暴露) |
| `float2int_*` `int2float_*` `float2uint_*` `uint2float_*` `float2ll_*` `ll2float_*` `float2ull_*` `ull2float_*` `float2half_rn` `half2float` `float_as_uint` `int_as_float` `uint_as_float` | A2/A3 · 950 | ✅ | ❌ | 带舍入的类型转换族(FT 未暴露) |
| `erfc` `erfcx` `erfcinv` `tgamma` `normcdf` `normcdfinv` `j0` `j1` `jn` `y0` `y1` `yn` `cyl_bessel_i1` `norm3d` `norm4d` `rnorm3d` `rnorm4d` `rhypot` `cbrt` `rcbrt` | A2/A3 · 950 | ✅ | ❌ | 特殊函数(Bessel/误差/范数,FT 未暴露) |
| `clz` `popc` `ffs` `brev` `byte_perm` `mulhi` `mul24` `umul24` `sad` `usad` `hadd` `rhadd` `uhadd` `urhadd` | A2/A3 · 950 | ✅ | ❌ | 整数/位运算(FT 未暴露) |
| `fast_cosf` `fast_sinf` `fast_tanf` `fast_tanhf` `fast_logf` `fast_log2f` `fast_log10f` `fast_exp10f` `fast_powf` | A2/A3 · 950 | ✅ | ❌ | 快速近似(FT 仅 `fast_expf`/`fast_dividef`) |
| `exp10` `logb` `remainder` `rint` `llrint` `llround` `fdim` `scalbn` `saturatef` `nan` `max` `min` | A2/A3 · 950 | ✅ | ❌ | 其他数学(FT 未暴露) |
| `rsqrt_rn` | A2/A3 · 950 | ✅ | ❌ | 舍入 rsqrt |

### 1.3 `extension.buffer.language`(通用 buffer 抽象)

> 两仓源码逐函数一致;FT 放在 `python/triton/spec/ascend/...` 下。

| 接口名 | 适用硬件 | TA 支持 | FT 支持 | 功能 |
|---|---|---|---|---|
| `buffer` | A2/A3 · 950 | ✅ | ✅ | 片上 buffer 值类型 |
| `buffer_type` | A2/A3 · 950 | ✅ | ✅ | buffer 类型(element/shape/space/strides) |
| `address_space` | A2/A3 · 950 | ✅ | ✅ | 地址空间(UB/L1/L0A/L0B/L0C) |
| `alloc` | A2/A3 · 950 | ✅ | ✅ | 分配 buffer |
| `to_buffer` | A2/A3 · 950 | ✅ | ✅ | tensor→buffer |
| `to_tensor` | A2/A3 · 950 | ✅ | ✅ | buffer→tensor |
| `subview` | A2/A3 · 950 | ✅ | ✅ | buffer 子视图 |
| `builtin` / `is_builtin` | A2/A3 · 950 | ✅ | ✅ | builtin 装饰器/判定(辅助) |

### 1.4 `cann.kernels`(现成 kernel)

| 接口名 | 适用硬件 | TA 支持 | FT 支持 | 功能 |
|---|---|---|---|---|
| `gather_2d_simd` | A2/A3 · 950 | ✅ | ✅ | 2D SIMD gather 高层 kernel(两仓源码一致) |

### 1.5 `tla`(TLA 显存层级范式,TA 独有)

> FT **完全无此命名空间**,整表 FT 列均为 ❌。
> **架构说明**:tla 的 Python 层**无硬架构门**,原语在 A2/A3/950 均可发 IR;真正的架构约束落在 IR verifier/lowering 层——最典型是 **fixpipe 的 `dst=UB` 与 `NZ2DN` 变体被限定为 `isAscend950`**(见 [tla/compiler.py](../triton-ascend/third_party/ascend/language/tla/compiler.py) 注释)。即「用 L0C→UB fixpipe 做片上回写」是 950 专属能力,A2/A3 走 L0C→GM/L1 路径。

| 接口名 | 适用硬件 | TA 支持 | FT 支持 | 功能 |
|---|---|---|---|---|
| `tla.jit` | A2/A3 · 950 | ✅ | ❌ | TLA 前端 kernel 装饰器 |
| `tla.alloc` / `tla.alloc_buffer` | A2/A3 · 950 | ✅ | ❌ | 分配片上 buffer(UB/L1/L0,带版本) |
| `tla.buffer` / `tla.buffer_type` | A2/A3 · 950 | ✅ | ❌ | buffer 值/类型 |
| `tla.vector` / `tla.vector_type` | A2/A3 · 950 | ✅ | ❌ | SIMD 向量寄存器值/类型 |
| `tla.copy` | A2/A3 · 950 | ✅ | ❌ | GM↔buffer 拷贝 |
| `tla.view` / `tla.subview` / `tla.reshape` | A2/A3 · 950 | ✅ | ❌ | GM 视图 / 子视图 / 重塑 |
| `tla.load_gm_to_ub` / `store_ub_to_gm` | A2/A3 · 950 | ✅ | ❌ | GM↔UB DMA |
| `tla.load_gm_to_l1` / `copy_ub_to_l1` / `copy_l1_to_ub` | A2/A3 · 950 | ✅ | ❌ | L1 相关 DMA |
| `tla.copy_l1_to_l0` / `copy_l1_to_bt` / `copy_l1_to_fb` | A2/A3 · 950 | ✅ | ❌ | L1→L0/BT/FB DMA |
| `tla.store_l0c_to_gm` / `copy_l0c_to_ub` / `copy_l0c_to_l1` | A2/A3 · 950(部分路径仅 950) | ✅ | ❌ | L0C 输出 DMA(含 fixpipe)。**`dst=UB` / `NZ2DN` fixpipe 由 `FixpipeOp::verify` 限制为 `isAscend950`**(IR verifier 层) |
| `tla.set_ctrl` | A2/A3 · 950 | ✅ | ❌ | 设置控制寄存器 |
| `tla.simd.*`(~60) | A2/A3 · 950 | ✅ | ❌ | SIMD 向量指令(vadd/vmul/vfma/vexp/vreduce_sum/vcmp/vsel…) |
| `tla.cube.mmad` | A2/A3 · 950 | ✅ | ❌ | cube 核矩阵乘(L1/L0C 路由) |
| `tla.scope.simd` / `tla.scope.cube` | A2/A3 · 950 | ✅ | ❌ | 核模式作用域 |
| `tla.sync.*` | A2/A3 · 950 | ✅ | ❌ | 显式流水/跨核同步(barrier/set_flag/wait_flag) |
| `tla.where` | A2/A3 · 950 | ✅ | ❌ | eager 标量/SIMD select |
| `tla.compensated_rstd` / `tla.truncmod` | A2/A3 · 950 | ✅ | ❌ | 补偿 rstd / 截断取模 |
| `tla.device_print` / `tla.device_assert` | A2/A3 · 950 | ✅ | ❌ | 设备打印/断言 |
| `tla.annotate_buffer_versions` / `tla.assume` | A2/A3 · 950 | ✅ | ❌ | buffer 版本标注 / 假设 |

### 1.6 `triton.testing`(昇腾专属测试)

| 接口名 | 适用硬件 | TA 支持 | FT 支持 | 功能 |
|---|---|---|---|---|
| `do_bench_npu` | A2/A3 · 950 | ✅ | ✅ | NPU 算子性能基准 |

---

## 表 2 · 接口源码路径对照

> 路径相对本文件(`/workspace/FlagTree-vs-TritonAscend/`)。`—` = 该仓不提供。

| 算子名 | Triton-Ascend 源码路径 | FlagTree 源码路径 |
|---|---|---|
| `al.dot` | `../triton-ascend/third_party/ascend/language/cann/extension/core.py` | — |
| `al.conv1d` | `../triton-ascend/third_party/ascend/language/cann/extension/core.py` | `../FlagTree/third_party/ascend/language/cann/extension/core.py` |
| `al.conv2d` | `../triton-ascend/third_party/ascend/language/cann/extension/core.py` | — |
| `al.fixpipe` | `../triton-ascend/third_party/ascend/language/cann/extension/core.py` | `../FlagTree/third_party/ascend/language/cann/extension/core.py` |
| `al.copy` / `copy_from_ub_to_l1` | `../triton-ascend/third_party/ascend/language/cann/extension/core.py` | `../FlagTree/third_party/ascend/language/cann/extension/core.py` |
| `al.debug_barrier` / `sub_vec_id` / `sub_vec_num` | `../triton-ascend/third_party/ascend/language/cann/extension/core.py` | `../FlagTree/third_party/ascend/language/cann/extension/core.py` |
| `al.sync_block_all/set/wait` | `../triton-ascend/third_party/ascend/language/cann/extension/core.py` | `../FlagTree/third_party/ascend/language/cann/extension/aux_ops.py` |
| 枚举 `PIPE/MODE/CORE/IteratorType/Fixpipe*` | `../triton-ascend/third_party/ascend/language/cann/extension/core.py` | `../FlagTree/third_party/ascend/language/cann/extension/core.py` |
| `al.AddressSpace` / `al.DataLayout` | `../triton-ascend/third_party/ascend/language/cann/extension/core.py` | — |
| `al.scope` | `../triton-ascend/third_party/ascend/language/cann/extension/scope.py` | `../FlagTree/third_party/ascend/language/cann/extension/scope.py` |
| `al.parallel` / `compile_hint` / `multibuffer` | `../triton-ascend/third_party/ascend/language/cann/extension/aux_ops.py` | `../FlagTree/third_party/ascend/language/cann/extension/aux_ops.py` |
| `al.insert_slice` / `extract_slice` / `get_element` / `sort` / `flip` / `cast` | `../triton-ascend/third_party/ascend/language/cann/extension/vec_ops.py` | `../FlagTree/third_party/ascend/language/cann/extension/vec_ops.py` |
| `al.index_put` / `gather_out_to_ub` / `scatter_ub_to_out` / `index_select_simd` | `../triton-ascend/third_party/ascend/language/cann/extension/mem_ops.py` | `../FlagTree/third_party/ascend/language/cann/extension/mem_ops.py` |
| `al.custom` / `custom_semantic` / `register_custom_op` | `../triton-ascend/third_party/ascend/language/cann/extension/custom_op.py` | `../FlagTree/third_party/ascend/language/cann/extension/custom_op.py` |
| `al.custom_object_semantic` | — | `../FlagTree/third_party/ascend/language/cann/extension/custom_op.py` |
| `al.SyncEventSlot` | `../triton-ascend/third_party/ascend/language/cann/extension/custom_op.py` | — |
| `al.atan2` / `isfinited` / `finitef` | `../triton-ascend/third_party/ascend/language/cann/extension/math_ops.py` | `../FlagTree/third_party/ascend/language/cann/extension/math_ops.py` |
| `al.affine_*` | `../triton-ascend/third_party/ascend/language/cann/extension/__init__.py`(MLIR 透传) | `../FlagTree/third_party/ascend/language/cann/extension/__init__.py`(MLIR 透传) |
| `al.is_register_based_arch` | `../triton-ascend/third_party/ascend/backend/utils.py` | — |
| `cann.libdevice.*`(全量) | `../triton-ascend/third_party/ascend/language/cann/libdevice.py` | `../FlagTree/third_party/ascend/language/cann/libdevice.py`(精简 + rebind 见 `cann/__init__.py`) |
| `extension.buffer.language.*` | `../triton-ascend/python/triton/extension/buffer/language/core.py` | `../FlagTree/python/triton/spec/ascend/extension/buffer/language/core.py` |
| `kernels.gather_2d_simd` | `../triton-ascend/third_party/ascend/language/kernels/gather.py` | `../FlagTree/third_party/ascend/language/kernels/gather.py` |
| `tla.*`(jit/alloc/view/copy/dma/simd/cube/scope/sync/…) | `../triton-ascend/third_party/ascend/language/tla/` | — |
| `do_bench_npu` | `../triton-ascend/third_party/ascend/backend/testing.py` | `../FlagTree/third_party/ascend/backend/testing.py` |

---

## 表 3 · 通用 Triton API(`tl.*` / `triton` / `triton.testing`)

> **两仓完全一致**(Triton-Ascend 基于 3.6.0,FlagTree 基于 3.5.1,均暴露下列 API)。
> 分组依据 Triton-Ascend 在线文档;硬件列统一为 A2/A3 · 950。全部 TA ✅ / FT ✅。
> 已逐项核验 `map_elementwise` / `dot_scaled` / `gather` / `tensor_descriptor` / `topk` / `condition` / `clamp` / `sqrt_rn` 等较新符号在两仓 `language/__init__.py` 均存在。

### 3.1 `triton`(入口)

| 接口名 | 适用硬件 | TA 支持 | FT 支持 | 功能 |
|---|---|---|---|---|
| `triton.jit` | A2/A3 · 950 | ✅ | ✅ | kernel JIT 装饰器 |
| `triton.autotune` | A2/A3 · 950 | ✅ | ✅ | 自动调优装饰器 |
| `triton.heuristics` | A2/A3 · 950 | ✅ | ✅ | 启发式参数装饰器 |
| `triton.Config` | A2/A3 · 950 | ✅ | ✅ | autotune 配置(昇腾上 num_stages/num_ctas/maxnreg 无效) |

### 3.2 `triton.testing`

| 接口名 | 适用硬件 | TA 支持 | FT 支持 | 功能 |
|---|---|---|---|---|
| `triton.testing.Benchmark` | A2/A3 · 950 | ✅ | ✅ | 基准配置对象 |
| `triton.testing.do_bench` | A2/A3 · 950 | ✅ | ✅ | 通用性能基准 |
| `triton.testing.do_bench_cudagraph` | A2/A3 · 950 | ✅ | ✅ | CUDA graph 基准(昇腾不适用,接口存在) |
| `triton.testing.perf_report` | A2/A3 · 950 | ✅ | ✅ | 性能报告 |
| `triton.testing.assert_close` | A2/A3 · 950 | ✅ | ✅ | 数值近似断言 |

### 3.3 `triton.language`(`tl.*`,核心 DSL)

| 接口名 | 分组 | 适用硬件 | TA 支持 | FT 支持 | 功能 |
|---|---|---|---|---|---|
| `tl.tensor` | Programming Model | A2/A3 · 950 | ✅ | ✅ | tensor 类型 |
| `tl.tensor_descriptor` | Programming Model | A2/A3 · 950 | ✅ | ✅ | tensor 描述符类型 |
| `tl.program_id` | Programming Model | A2/A3 · 950 | ✅ | ✅ | 当前 program id |
| `tl.num_programs` | Programming Model | A2/A3 · 950 | ✅ | ✅ | program 总数 |
| `tl.map_elementwise` | Programming Model | A2/A3 · 950 | ✅ | ✅ | 逐元素映射 |
| `tl.arange` | Creation | A2/A3 · 950 | ✅ | ✅ | 连续区间 |
| `tl.cat` | Creation | A2/A3 · 950 | ✅ | ✅ | 拼接 |
| `tl.full` | Creation | A2/A3 · 950 | ✅ | ✅ | 常量填充 |
| `tl.zeros` | Creation | A2/A3 · 950 | ✅ | ✅ | 全零 |
| `tl.zeros_like` | Creation | A2/A3 · 950 | ✅ | ✅ | 同形全零 |
| `tl.cast` | Creation | A2/A3 · 950 | ✅ | ✅ | 类型转换 |
| `tl.broadcast` | Shape | A2/A3 · 950 | ✅ | ✅ | 广播 |
| `tl.broadcast_to` | Shape | A2/A3 · 950 | ✅ | ✅ | 广播到形状 |
| `tl.expand_dims` | Shape | A2/A3 · 950 | ✅ | ✅ | 扩维 |
| `tl.interleave` | Shape | A2/A3 · 950 | ✅ | ✅ | 交错 |
| `tl.join` | Shape | A2/A3 · 950 | ✅ | ✅ | 合并 |
| `tl.permute` | Shape | A2/A3 · 950 | ✅ | ✅ | 维度置换 |
| `tl.ravel` | Shape | A2/A3 · 950 | ✅ | ✅ | 展平 |
| `tl.reshape` | Shape | A2/A3 · 950 | ✅ | ✅ | 重塑 |
| `tl.split` | Shape | A2/A3 · 950 | ✅ | ✅ | 拆分 |
| `tl.trans` | Shape | A2/A3 · 950 | ✅ | ✅ | 转置 |
| `tl.view` | Shape | A2/A3 · 950 | ✅ | ✅ | 视图 |
| `tl.dot` | Linear Algebra | A2/A3 · 950 | ✅ | ✅ | 矩阵乘(昇腾经后端 lowering,tf32→hf32) |
| `tl.dot_scaled` | Linear Algebra | A2/A3 · 950 | ✅ | ✅ | 带 scale 的矩阵乘 |
| `tl.load` | Memory/Pointer | A2/A3 · 950 | ✅ | ✅ | 加载 |
| `tl.store` | Memory/Pointer | A2/A3 · 950 | ✅ | ✅ | 存储 |
| `tl.make_tensor_descriptor` | Memory/Pointer | A2/A3 · 950 | ✅ | ✅ | 创建 tensor 描述符 |
| `tl.load_tensor_descriptor` | Memory/Pointer | A2/A3 · 950 | ✅ | ✅ | 按描述符加载 |
| `tl.store_tensor_descriptor` | Memory/Pointer | A2/A3 · 950 | ✅ | ✅ | 按描述符存储 |
| `tl.make_block_ptr` | Memory/Pointer | A2/A3 · 950 | ✅ | ✅ | 块指针 |
| `tl.advance` | Memory/Pointer | A2/A3 · 950 | ✅ | ✅ | 块指针前进 |
| `tl.flip` | Indexing | A2/A3 · 950 | ✅ | ✅ | 翻转 |
| `tl.where` | Indexing | A2/A3 · 950 | ✅ | ✅ | 条件选择 |
| `tl.swizzle2d` | Indexing | A2/A3 · 950 | ✅ | ✅ | 2D swizzle |
| `tl.gather` | Indexing | A2/A3 · 950 | ✅ | ✅ | gather |
| `tl.abs` | Math | A2/A3 · 950 | ✅ | ✅ | 绝对值 |
| `tl.add` | Math | A2/A3 · 950 | ✅ | ✅ | 加 |
| `tl.cdiv` | Math | A2/A3 · 950 | ✅ | ✅ | 向上整除 |
| `tl.ceil` | Math | A2/A3 · 950 | ✅ | ✅ | 向上取整 |
| `tl.clamp` | Math | A2/A3 · 950 | ✅ | ✅ | 钳位 |
| `tl.cos` | Math | A2/A3 · 950 | ✅ | ✅ | cos |
| `tl.div` | Math | A2/A3 · 950 | ✅ | ✅ | 除 |
| `tl.div_rn` | Math | A2/A3 · 950 | ✅ | ✅ | 舍入除 |
| `tl.erf` | Math | A2/A3 · 950 | ✅ | ✅ | 误差函数 |
| `tl.exp` | Math | A2/A3 · 950 | ✅ | ✅ | 指数 |
| `tl.exp2` | Math | A2/A3 · 950 | ✅ | ✅ | 2^x |
| `tl.fdiv` | Math | A2/A3 · 950 | ✅ | ✅ | 浮点除 |
| `tl.floordiv` | Math | A2/A3 · 950 | ✅ | ✅ | 向下整除 |
| `tl.floor` | Math | A2/A3 · 950 | ✅ | ✅ | 向下取整 |
| `tl.fma` | Math | A2/A3 · 950 | ✅ | ✅ | 乘加 |
| `tl.log` | Math | A2/A3 · 950 | ✅ | ✅ | 自然对数 |
| `tl.log2` | Math | A2/A3 · 950 | ✅ | ✅ | log2 |
| `tl.maximum` | Math | A2/A3 · 950 | ✅ | ✅ | 取大 |
| `tl.minimum` | Math | A2/A3 · 950 | ✅ | ✅ | 取小 |
| `tl.mod` | Math | A2/A3 · 950 | ✅ | ✅ | 取模 |
| `tl.mul` | Math | A2/A3 · 950 | ✅ | ✅ | 乘 |
| `tl.neg` | Math | A2/A3 · 950 | ✅ | ✅ | 取负 |
| `tl.rsqrt` | Math | A2/A3 · 950 | ✅ | ✅ | 1/sqrt |
| `tl.sigmoid` | Math | A2/A3 · 950 | ✅ | ✅ | sigmoid |
| `tl.sin` | Math | A2/A3 · 950 | ✅ | ✅ | sin |
| `tl.softmax` | Math | A2/A3 · 950 | ✅ | ✅ | softmax |
| `tl.sqrt` | Math | A2/A3 · 950 | ✅ | ✅ | 开方 |
| `tl.sqrt_rn` | Math | A2/A3 · 950 | ✅ | ✅ | 舍入开方 |
| `tl.sub` | Math | A2/A3 · 950 | ✅ | ✅ | 减 |
| `tl.umulhi` | Math | A2/A3 · 950 | ✅ | ✅ | 无符号乘高位 |
| `tl.and` | Logical | A2/A3 · 950 | ✅ | ✅ | 按位与 |
| `tl.or` | Logical | A2/A3 · 950 | ✅ | ✅ | 按位或 |
| `tl.xor` | Logical | A2/A3 · 950 | ✅ | ✅ | 按位异或 |
| `tl.not` | Logical | A2/A3 · 950 | ✅ | ✅ | 按位非 |
| `tl.logical_and` | Logical | A2/A3 · 950 | ✅ | ✅ | 逻辑与 |
| `tl.logical_or` | Logical | A2/A3 · 950 | ✅ | ✅ | 逻辑或 |
| `tl.invert` | Logical | A2/A3 · 950 | ✅ | ✅ | 取反 |
| `tl.lshift` | Logical | A2/A3 · 950 | ✅ | ✅ | 左移 |
| `tl.rshift` | Logical | A2/A3 · 950 | ✅ | ✅ | 右移 |
| `tl.gt` `tl.ge` `tl.lt` `tl.le` `tl.eq` `tl.ne` | Comparison | A2/A3 · 950 | ✅ | ✅ | 比较(>/>=/</<=/==/!=) |
| `tl.argmax` | Reduction | A2/A3 · 950 | ✅ | ✅ | 最大值索引 |
| `tl.argmin` | Reduction | A2/A3 · 950 | ✅ | ✅ | 最小值索引 |
| `tl.max` | Reduction | A2/A3 · 950 | ✅ | ✅ | 最大值 |
| `tl.min` | Reduction | A2/A3 · 950 | ✅ | ✅ | 最小值 |
| `tl.reduce` | Reduction | A2/A3 · 950 | ✅ | ✅ | 通用归约 |
| `tl.reduce_or` | Reduction | A2/A3 · 950 | ✅ | ✅ | 归约或 |
| `tl.sum` | Reduction | A2/A3 · 950 | ✅ | ✅ | 求和 |
| `tl.xor_sum` | Reduction | A2/A3 · 950 | ✅ | ✅ | 异或归约 |
| `tl.associative_scan` | Scan/Sort | A2/A3 · 950 | ✅ | ✅ | 结合扫描 |
| `tl.bitonic_merge` | Scan/Sort | A2/A3 · 950 | ✅ | ✅ | 双调归并 |
| `tl.cumprod` | Scan/Sort | A2/A3 · 950 | ✅ | ✅ | 累积乘 |
| `tl.cumsum` | Scan/Sort | A2/A3 · 950 | ✅ | ✅ | 累积和 |
| `tl.histogram` | Scan/Sort | A2/A3 · 950 | ✅ | ✅ | 直方图 |
| `tl.sort` | Scan/Sort | A2/A3 · 950 | ✅ | ✅ | 排序 |
| `tl.topk` | Scan/Sort | A2/A3 · 950 | ✅ | ✅ | topk |
| `tl.atomic_add` | Atomic | A2/A3 · 950 | ✅ | ✅ | 原子加 |
| `tl.atomic_and` | Atomic | A2/A3 · 950 | ✅ | ✅ | 原子与 |
| `tl.atomic_cas` | Atomic | A2/A3 · 950 | ✅ | ✅ | 原子比较交换 |
| `tl.atomic_max` | Atomic | A2/A3 · 950 | ✅ | ✅ | 原子最大 |
| `tl.atomic_min` | Atomic | A2/A3 · 950 | ✅ | ✅ | 原子最小 |
| `tl.atomic_or` | Atomic | A2/A3 · 950 | ✅ | ✅ | 原子或 |
| `tl.atomic_xchg` | Atomic | A2/A3 · 950 | ✅ | ✅ | 原子交换 |
| `tl.atomic_xor` | Atomic | A2/A3 · 950 | ✅ | ✅ | 原子异或 |
| `tl.randint4x` | Random | A2/A3 · 950 | ✅ | ✅ | 4×随机整数 |
| `tl.randint` | Random | A2/A3 · 950 | ✅ | ✅ | 随机整数 |
| `tl.rand` | Random | A2/A3 · 950 | ✅ | ✅ | 均匀随机 |
| `tl.rand4x` | Random | A2/A3 · 950 | ✅ | ✅ | 4×均匀随机 |
| `tl.randn` | Random | A2/A3 · 950 | ✅ | ✅ | 正态随机 |
| `tl.randn4x` | Random | A2/A3 · 950 | ✅ | ✅ | 4×正态随机 |
| `tl.range` | Iterators | A2/A3 · 950 | ✅ | ✅ | 运行时循环(支持昇腾步长) |
| `tl.static_range` | Iterators | A2/A3 · 950 | ✅ | ✅ | 编译期循环 |
| `tl.condition` | Iterators | A2/A3 · 950 | ✅ | ✅ | 条件迭代 |
| `tl.assume` | Compiler Hint | A2/A3 · 950 | ✅ | ✅ | 假设提示 |
| `tl.debug_barrier` | Compiler Hint | A2/A3 · 950 | ✅ | ✅ | 调试屏障 |
| `tl.max_constancy` | Compiler Hint | A2/A3 · 950 | ✅ | ✅ | 常量性提示 |
| `tl.max_contiguous` | Compiler Hint | A2/A3 · 950 | ✅ | ✅ | 连续性提示 |
| `tl.multiple_of` | Compiler Hint | A2/A3 · 950 | ✅ | ✅ | 整除性提示 |
| `tl.static_print` | Debug | A2/A3 · 950 | ✅ | ✅ | 编译期打印 |
| `tl.static_assert` | Debug | A2/A3 · 950 | ✅ | ✅ | 编译期断言 |
| `tl.device_print` | Debug | A2/A3 · 950 | ✅ | ✅ | 设备打印 |
| `tl.device_assert` | Debug | A2/A3 · 950 | ✅ | ✅ | 设备断言 |
| `tl.inline_asm_elementwise` | Inline Assembly | A2/A3 · 950 | ✅ | ✅ | 逐元素内联汇编 |

> 来源:[triton-ascend 文档 rst](../triton-ascend/docs/zh/python-api/triton.language.rst)。FlagTree 对应 [python/triton/language/__init__.py](../FlagTree/python/triton/language/__init__.py),逐符号核验均存在。

---

## 表 4 · FlagTree TLE 扩展原语 → Triton-Ascend 对应情况

> **TLE(`triton.experimental.tle`)为 FlagTree 独有**(源码 [experimental/tle/](../FlagTree/python/triton/experimental/tle/)),是跨 GPU/NPU 的显式内存/流水语言,含完整昇腾实现(`dsa.ascend`)。
> 本表看每个 TLE 原语在 **Triton-Ascend** 里有没有「相同 API(仅接口层差异)」或「类似 API」。
> 「TA 对应」列:**相同**=语义一致仅命名/签名差异 · **类似**=概念对应但范式/粒度不同 · **无**=TA 无对应物。
> **架构说明**:TLE `dsa.ascend` 层的 Python 代码**无硬架构门**,原语在 A2/A3/950 均可发 IR,架构约束在 lowering/verifier 层(与 tla 同机制)。因此「适用硬件」列对 TLE 原语统一记 A2/A3 · 950;但凡涉及 **L0C→UB fixpipe、register-based 片上搬运**(如对应 `al.copy`/`al.fixpipe` 者)的实际可用性受 **950 专属**约束,与 TA 侧一致(见表 1.1 与 1.5 注)。

| TLE 接口名(FT) | 所属层 | 适用硬件 | TA 对应 | TA 对应 API | 功能 |
|---|---|---|---|---|---|
| `tle.dsa.ascend.sub_vec_id` | DSA-Ascend | A2/A3 · 950 | **相同** | `al.sub_vec_id` | 子向量核 id |
| `tle.dsa.ascend.sub_vec_num` | DSA-Ascend | A2/A3 · 950 | **相同** | `al.sub_vec_num` | 子向量核数量 |
| `tle.dsa.ascend.sync_block_set` | DSA-Ascend | A2/A3 · 950 | **相同** | `al.sync_block_set` | 跨核同步 set flag |
| `tle.dsa.ascend.sync_block_wait` | DSA-Ascend | A2/A3 · 950 | **相同** | `al.sync_block_wait` | 跨核同步 wait flag |
| `tle.dsa.ascend.sync_block_all` | DSA-Ascend | A2/A3 · 950 | **相同** | `al.sync_block_all` | 跨核块同步 |
| `tle.dsa.ascend.compile_hint` | DSA-Ascend | A2/A3 · 950 | **相同** | `al.compile_hint` | 编译提示 |
| `tle.dsa.ascend.multibuffer` | DSA-Ascend | A2/A3 · 950 | **相同** | `al.multibuffer` | 多缓冲标注 |
| `tle.dsa.ascend.UB/L1/L0A/L0B/L0C` | DSA-Ascend | A2/A3 · 950 | **相同** | `al.ascend_address_space` / `tla` space(`"ub"/"l1"/"l0"`) | 片上内存层级常量 |
| `tle.dsa.ascend.PIPE` | DSA-Ascend | A2/A3 · 950 | **相同** | `al.PIPE` / `tla.sync.PIPE` | 流水枚举 |
| `tle.dsa.ascend.SyncSpec` | DSA-Ascend | A2/A3 · 950 | **类似** | `al.SyncEventSlot` | 同步事件描述 |
| `tle.dsa.alloc` | DSA 通用 | A2/A3 · 950 | **类似** | `tla.alloc` / `tla.alloc_buffer` | 分配片上 buffer |
| `tle.dsa.copy` | DSA 通用 | A2/A3 · 950 | **类似** | `tla.copy` / `al.copy` | buffer 间拷贝 |
| `tle.dsa.subview` | DSA 通用 | A2/A3 · 950 | **类似** | `tla.subview` | buffer 子视图 |
| `tle.dsa.extract_slice` / `insert_slice` | DSA 通用 | A2/A3 · 950 | **相同** | `al.extract_slice` / `al.insert_slice` | 片上切片 |
| `tle.dsa.extract_element` | DSA 通用 | A2/A3 · 950 | **相同** | `al.get_element` | 取标量元素 |
| `tle.dsa.to_tensor` / `to_buffer` | DSA 通用 | A2/A3 · 950 | **类似** | `extension.buffer.language.to_tensor/to_buffer` | buffer↔tensor |
| `tle.dsa.add/sub/mul/div/max/min` | DSA 通用 | A2/A3 · 950 | **类似** | `tla.simd.v*`(vadd/vmul/…)/ `tl.*` | 片上算术 |
| `tle.dsa.tile_alloc` | DSA 通用 | A2/A3 · 950 | **类似** | `tla.alloc_buffer`(L1 多槽见 TA 经验) | tile 级分配 |
| `tle.dsa.tile_copy` | DSA 通用 | A2/A3 · 950 | **类似** | `tla.copy` + DMA 族 | tile 级搬运 |
| `tle.dsa.tile_subview` | DSA 通用 | A2/A3 · 950 | **类似** | `tla.subview` | tile 子视图 |
| `tle.dsa.tile_set_flag` / `tile_wait_flag` | DSA 通用 | A2/A3 · 950 | **相同** | `tla.sync.set_flag` / `wait_flag` | tile 流水 flag |
| `tle.dsa.tile_pipe_barrier` | DSA 通用 | A2/A3 · 950 | **相同** | `tla.sync.barrier` | 流水屏障 |
| `tle.dsa.tile_cube_launch` / `tile_cube_wait` | DSA 通用 | A2/A3 · 950 | **类似** | `tla.cube.mmad` + `tla.scope.cube` | cube 核启动/等待 |
| `tle.dsa.tile_gm_offset` | DSA 通用 | A2/A3 · 950 | **类似** | `tla.view`(GM 偏移) | GM 偏移计算 |
| `tle.dsa.pipeline` / `parallel` / `hint` | DSA 通用 | A2/A3 · 950 | **类似** | `al.parallel` / `al.compile_hint` | 流水/并行标注 |
| `tle.dsa.workspace` | DSA 通用 | A2/A3 · 950 | **类似** | (TA 经 kernel 入参/`tl` workspace) | 工作区分配 |
| `tle.scope(core_mode=)` | 通用 | A2/A3 · 950 | **相同** | `al.scope` / `tla.scope.simd/cube` | 核模式作用域 |
| `tle.dsa.ascend.pipe_scheduler` / `PipeScheduler` | DSA-Ascend | A2/A3 · 950 | **无** | — | 流水调度器(自动事件分配) |
| `tle.dsa.ascend.pipe_reader` / `pipe_writer` / `pipe_slot` / `run_pipe` | DSA-Ascend | A2/A3 · 950 | **无** | — | 生产者/消费者流水端点 |
| `tle.pipe`(顶层) | 通用 | A2/A3 · 950 | **无** | — | 管道原语(capacity/readers/one_shot) |
| `tle.dsa.ascend.device_mesh` / `MeshConfig` | DSA-Ascend | A2/A3 · 950 | **无** | — | 多卡设备网格 |
| `tle.dsa.ascend.remote` / `shard_id` / `distributed_barrier` | DSA-Ascend | A2/A3 · 950 | **无** | — | 分布式通信原语 |
| `tle.dsa.ascend.swizzle2d_Nz` / `gemm_swizzle2d_Nz` | DSA-Ascend | A2/A3 · 950 | **无** | — | NZ 布局 swizzle(分布式 GEMM) |
| `tle.dsa.ascend.custom_ops.gather_gm_to_ub/l1` | DSA-Ascend | A2/A3 · 950 | **类似** | `al.gather_out_to_ub` | GM→UB/L1 gather |
| `tle.dsa.ascend.custom_ops.sort32` / `mrgsort` / `unpack_sort` | DSA-Ascend | A2/A3 · 950 | **类似** | `al.sort` | 高性能排序算子 |
| `tle.dsa.ascend.custom_ops.cast_int4_to_fp16` | DSA-Ascend | A2/A3 · 950 | **类似** | `al.cast` | int4→fp16 转换 |
| `tle.dsa.ascend.custom_ops.pair_reduce_sum_continuous_mask` 等 | DSA-Ascend | A2/A3 · 950 | **无** | — | 特定 mask/reduce 融合算子 |
| `tle.load` | 通用 | GPU 为主 | **类似** | `tl.load`(昇腾无 async SMEM 语义) | 异步加载(Hopper) |
| `tle.tile_load` / `tile_store` | 通用 | GPU 为主 | **类似** | `tla` DMA 族 | tile 加载/存储 |
| `tle.tensor_view` / `make_partition_view` / `make_strided_view` / `make_gather_scatter_view` | 通用 | GPU 为主 | **类似** | `tla.view` / `tla.subview` | 张量视图构造 |
| `tle.local_ptr` | 通用(GPU) | GPU 专属 | **无** | — | SMEM 任意形状指针视图(Hopper) |
| `tle.smem` / `tle.tmem` | 通用(GPU) | GPU 专属 | **无** | — | 共享/tensor 内存作用域(Hopper) |

> 源码对照:TLE 昇腾层 [experimental/tle/language/dsa/ascend/](../FlagTree/python/triton/experimental/tle/language/dsa/ascend/);TA 对应 `al` = [cann/extension/](../triton-ascend/third_party/ascend/language/cann/extension/),`tla` = [language/tla/](../triton-ascend/third_party/ascend/language/tla/)。

**表 4 小结**:
- **相同(仅接口层差异)**:`dsa.ascend` 的跨核同步/子核/内存层级/编译提示族,与 TA 的 `al.*` 几乎一一对应;`tile_set_flag/wait_flag/pipe_barrier`、`scope` 与 TA `tla.sync`/`tla.scope` 对应。
- **类似(范式不同)**:`alloc`/`copy`/`subview`/`tile_*`/算术 → TA 的 `tla.*`(TLE 用 tile 抽象,TA 用 buffer+SIMD 指令);custom_ops 的 gather/sort/cast → TA 的 `al.*`。
- **TA 无对应**:TLE 的**流水调度器族**(`pipe_scheduler`/`pipe_reader/writer`)、**分布式通信族**(`device_mesh`/`remote`/`distributed_barrier`/`swizzle2d_Nz`)、以及 **Hopper SMEM/TMA 专属**(`local_ptr`/`smem`/`tmem`)—— 前两类是 TLE 相对 `tla` 的能力扩展,后一类是 GPU 专属不适用昇腾。

---

## 一句话总览

- **TA 独有**:整套 `tla` 范式、`al.dot` / `al.conv2d` builtin、全量 `libdevice`、`SyncEventSlot` / `AddressSpace` / `DataLayout` / `is_register_based_arch`。
- **FT 独有**:`al.custom_object_semantic`;**整套 TLE 扩展原语**(`triton.experimental.tle`,含 `dsa.ascend` 的流水调度器与分布式通信族,见表 4)。
- **共有且一致**:绝大多数 `al.*` builtin、`extension.buffer.language.*`、`kernels.gather_2d_simd`、`do_bench_npu`、通用 `tl.*`。
- **功能可用但实现不同**:FT 的部分 `libdevice` 函数 rebind 到 upstream `math.*`(表 1.2 ⚠️ 行)。
- **架构限制(两仓一致)**:`al.fixpipe`/`al.copy`/`al.copy_from_ub_to_l1`、`al.cast` 的 fp8/fp64、`tla` 的 L0C→UB/NZ2DN fixpipe 为 **950 专属**;libdevice 全量在 A2/A3/950 硬件可用,SIMT 守卫是「950+环境开关」模式限制。见各表「适用硬件」列。
- **TLE ↔ tla**:二者是 FlagTree / Triton-Ascend 各自的显式内存编程层,多数原语「相同或类似」,但 TLE 额外带流水调度器与分布式能力(表 4)。
