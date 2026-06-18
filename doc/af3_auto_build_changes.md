# AF3 自动构建改动说明

本文档梳理当前针对 `onescience.flax_models.alphafold3` 的自动构建改造内容。

## 目标

改造前，AlphaFold3 需要用户手工执行：

```bash
python src/onescience/flax_models/alphafold3/build_extension.py
```

改造后，目标变为：

- 在 `pip install .` / `python setup.py build_py` 阶段自动触发 AF3 构建
- 不在顶层 `setup.py` 中显式暴露 AF3 编译脚本执行
- 保留原有 `build_extension.py` 作为兼容入口
- 支持离线依赖目录注入

---

## 本次代码改动概览

### 1. 新增内部构建模块

新增文件：

- [src/onescience/_build/__init__.py](../src/onescience/_build/__init__.py)
- [src/onescience/_build/af3.py](../src/onescience/_build/af3.py)
- [src/onescience/_build/setup_hooks.py](../src/onescience/_build/setup_hooks.py)

其中核心逻辑位于：

- [src/onescience/_build/af3.py](../src/onescience/_build/af3.py)

职责包括：

- 判断是否启用 AF3 自动构建
- 检查离线依赖目录
- 调用 CMake 执行 C++ 扩展构建
- 调用 `build_data()` 生成数据文件
- 清理临时构建目录
- 判断已有构建产物是否可复用

关键函数：

- `should_build()`
- `resolve_dep_dir()`
- `ensure_dependencies_ready()`
- `build_cpp_extension()`
- `build_data_files()`
- `build_all()`
- `build_if_needed()`

---

### 2. 顶层 setup.py 保持 hook 驱动

相关文件：

- [setup.py](../setup.py)

当前 `setup.py` 并不直接执行 AF3 构建脚本，而是：

1. 扫描 `src/onescience/**/package_config.py`
2. 发现子模块导出的 `build_hook`
3. 在 `build_py` / `develop` 生命周期调用这些 hook

本次调整后：

- 在 [setup.py:7-10](../setup.py#L7-L10) 提前注入 `src` 到 `sys.path`
- 确保 `package_config.py` 可以导入内部构建模块
- 顶层 `setup.py` 仍不出现 `build_extension.py` 的直接调用

即：AF3 构建仍然是“子模块自注册 hook，顶层统一调度”的模式。

---

### 3. package_config.py 改为调用内部构建模块

相关文件：

- [src/onescience/flax_models/alphafold3/package_config.py](../src/onescience/flax_models/alphafold3/package_config.py)

改造前：

- `build_hook()` 通过 `subprocess` 直接执行 `build_extension.py`

改造后：

- 直接导入内部模块：
  - `from onescience._build.af3 import AF3BuildError, build_if_needed, is_strict`
- `build_hook()` 直接调用 `build_if_needed()`
- 不再显式拼接 AF3 脚本路径并执行 subprocess

这样实现了：

- AF3 自动构建能力仍由 `package_config.py` 暴露给顶层
- 但具体实现已经隐藏到内部构建模块

---

### 4. build_extension.py 改为兼容入口

相关文件：

- [src/onescience/flax_models/alphafold3/build_extension.py](../src/onescience/flax_models/alphafold3/build_extension.py)

改造前：

- 文件内部直接包含完整的 CMake + 数据构建逻辑

改造后：

- 只保留兼容入口职责
- 通过导入内部模块转发到 `build_all()`

当前用途：

- 兼容已有手工调用方式
- 避免旧脚本路径失效
- 主构建逻辑统一收敛到 `_build/af3.py`

---

### 5. 修正 AF3 数据文件打包位置

相关文件：

- [src/onescience/flax_models/alphafold3/package_config.py](../src/onescience/flax_models/alphafold3/package_config.py)

注意到 `build_data.py` 生成的 pickle 文件实际落在：

- `onescience/flax_models/alphafold3/constants/converters/`

因此本次改动同步调整了 `package_data`：

```python
"onescience.flax_models.alphafold3.constants.converters": [
    "*.pickle",
]
```

并调整了 `MANIFEST` 规则：

```text
recursive-include src/onescience/flax_models/alphafold3/constants/converters *.pickle
```

这样源码包和安装包都能正确包含生成的数据文件。

---

## 自动构建触发链路

当前触发流程如下：

```text
pip install .
  -> setup.py
  -> build_py / develop
  -> run_build_hooks()
  -> discover package_config.py
  -> alphafold3.package_config.build_hook()
  -> onescience._build.af3.build_if_needed()
  -> cmake configure/build/install
  -> build_data()
```

这意味着：

- 用户不需要额外执行 `python build_extension.py`
- 只要进入标准打包流程，AF3 构建就会自动执行

---

## 当前支持的环境变量

内部构建模块支持以下控制项：

### 1. `ONESCIENCE_BUILD_AF3`

可选值：

- `auto`：默认值，自动判断是否构建
- `1` / `true` / `on` / `force`：强制构建
- `0` / `false` / `off`：跳过构建

### 2. `ONESCIENCE_SKIP_AF3_BUILD`

兼容旧控制方式：

- `1` / `true` / `on` / `yes`：跳过构建

### 3. `ONESCIENCE_AF3_FORCE_REBUILD`

- 为真时，即使已有产物也强制重新构建

### 4. `ONESCIENCE_AF3_STRICT`

- 为真时，AF3 构建失败会直接抛错中断流程
- 否则允许跳过

### 5. `ALPHAFOLD3_DEP_DIR`

- 指向 AF3 离线依赖目录
- 若未设置，则自动回退到：

```text
third_party/alphafold3_deps
```

---

## 内部构建逻辑说明

### 1. 构建前检查

`ensure_dependencies_ready()` 会检查离线依赖目录中是否存在：

- `abseil-cpp`
- `pybind11`
- `pybind11_abseil`
- `libcifpp`
- `dssp`

缺失任一目录则报错。

### 2. 构建环境注入

`_build_env()` 会自动注入：

- `ALPHAFOLD3_DEP_DIR`
- `PYTHONPATH=<project>/src`

确保：

- CMake 可以找到离线依赖根目录
- `build_data()` 运行时能导入源码树内的 `onescience`

### 3. 产物复用判断

`artifacts_exist()` 会检查：

- AF3 包目录下是否已有 `cpp*.so` / `cpp*.pyd` / `cpp*.dll` / `cpp*.dylib`
- converters 目录下是否已有：
  - `ccd.pickle`
  - `chemical_component_sets.pickle`

如果产物完整且未设置强制重建，则跳过重复编译。

---

## 当前收益

改造完成后，当前工程具备以下能力：

- AF3 自动构建能力已经接入标准打包流程
- 顶层 `setup.py` 不显式暴露 AF3 编译脚本执行
- AF3 构建逻辑已统一收敛到内部模块，便于后续维护
- 旧的 `build_extension.py` 仍可兼容手工调用
- 离线依赖路径和构建行为已有统一控制入口

---

## 当前仍需注意的点

### 1. 还未做实际安装验证

当前代码已完成改造，但还需要通过如下方式验证：

```bash
python setup.py build_py
```

或：

```bash
pip install -e .
```

### 2. Linux 二进制兼容问题未根本消除

本次改造解决的是“自动构建接入流程”问题，不是“跨系统二进制兼容”问题。

对于不同 Linux / gcc / glibc 环境，最佳策略仍然是：

- 主流环境发布 wheel
- 异构环境在目标机自动源码构建

### 3. 未来可继续演进为 pyproject.toml / PEP 517

当前方案已经把构建逻辑从脚本中抽离。后续若迁移到：

- `pyproject.toml`
- `scikit-build-core`
- PEP 517 backend

改造成本会明显更低。

---

## 建议的下一步

建议继续做两件事：

1. 实际跑一轮构建验证，修正安装链路中的剩余问题
2. 后续将 AF3 构建链路迁移到更标准的 `pyproject.toml` 体系
