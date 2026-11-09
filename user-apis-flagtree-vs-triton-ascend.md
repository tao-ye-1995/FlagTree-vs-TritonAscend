# 用户层 API 对照:Triton-Ascend vs FlagTree(昇腾)

> 仓库:`/workspace/triton-ascend`(TA,Triton 3.6.0) · `/workspace/FlagTree`(FT,Triton 3.5.1)。
> 概要见 [triton-ascend-user-apis.md](triton-ascend-user-apis.md) / [flagtree-user-apis.md](flagtree-user-apis.md) / [flagtree-vs-triton-ascend.md](flagtree-vs-triton-ascend.md)。
>
> 硬件列说明:**A2/A3** = Atlas A2 / A3(910B / 910C);**950** = Ascend 950PR&950DT(910_95 / A5 register-based)。
> 支持列:✅ 支持 · ❌ 不暴露 · ⚠️ 以 rebind/间接方式提供。
> 范围:**聚焦昇腾专属扩展 API**(`cann.extension` / `cann.libdevice` / `tla` / `extension.buffer.language` / `kernels`);通用 `tl.*` 两仓一致,不逐一列出。

---

## 表 1 · 接口支持与功能对照

### 1.1 `cann.extension`(`al`,底层 builtin)

| 接口名 | 适用硬件 | TA 支持 | FT 支持 | 功能 |
|---|---|---|---|---|
| `al.dot` | A2/A3 · 950 | ✅ | ❌ | cube 矩阵乘 `D=A*B`,支持 ND / fractal(zN)layout |
| `al.conv1d` | A2/A3 · 950 | ✅ | ✅ | 1D 卷积 builtin |
| `al.conv2d` | A2/A3 · 950 | ✅ | ❌ | 2D 卷积 builtin |
| `al.fixpipe` | A2/A3 · 950 | ✅ | ✅ | L0C→UB 固定流水搬运(含量化/ReLU 后处理) |
| `al.copy` | A2/A3 · 950 | ✅ | ✅ | 片上内存间拷贝 |
| `al.copy_from_ub_to_l1` | A2/A3 · 950 | ✅ | ✅ | UB→L1 拷贝 |
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
| `al.cast` | A2/A3 · 950 | ✅ | ✅ | 显式类型转换 |
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

> TA **全量自实现**(~180);FT **精简**:39 个自实现 + ~20 个 rebind 到 upstream `math.*`。
> FT「⚠️」= 通过 rebind upstream `triton.language.math` 提供(功能可用,但非 cann 自实现)。

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
| `tla.store_l0c_to_gm` / `copy_l0c_to_ub` / `copy_l0c_to_l1` | A2/A3 · 950 | ✅ | ❌ | L0C 输出 DMA(含 fixpipe) |
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

## 一句话总览

- **TA 独有**:整套 `tla` 范式、`al.dot` / `al.conv2d` builtin、全量 `libdevice`、`SyncEventSlot` / `AddressSpace` / `DataLayout` / `is_register_based_arch`。
- **FT 独有**:`al.custom_object_semantic`。
- **共有且一致**:绝大多数 `al.*` builtin、`extension.buffer.language.*`、`kernels.gather_2d_simd`、`do_bench_npu`、通用 `tl.*`。
- **功能可用但实现不同**:FT 的部分 `libdevice` 函数 rebind 到 upstream `math.*`(表 1.2 ⚠️ 行)。
