# OneScience VASP DCU Demo

本目录包含两个 OneScience VASP 算例，用于演示如何在 DCU 环境下运行 VASP。两个算例分别展示了 **4 卡** 与 **8 卡** 并行计算的配置与提交方式，用户可直接参考对应的 `submit.sh` 脚本，并根据自身任务进行调整。

> **版权与许可证声明**
> VASP 为商业软件，使用本算例前请确保您已具备有效的 VASP 使用许可（license）。本 demo 仅提供运行方式参考，不包含 VASP 源代码或授权文件。

## 算例说明

| 算例目录 | 并行规模 | 说明 |
|---------|---------|------|
| `B.hR105` | 8 卡 | 8 DCU 并行 VASP 标准计算 |
| `silicalFPEN` | 4 卡 | 4 DCU 并行 VASP 标准计算 |

每个算例目录下均已提供：
- `POSCAR` / `INCAR` / `KPOINTS` / `POTCAR` —— VASP 输入文件
- `submit.sh` —— Slurm 作业提交脚本（含环境加载、CPU/GPU 亲和性绑定及运行前清理逻辑）

## 使用方式

1. 进入对应算例目录：
   ```bash
   cd B.hR105        # 或 silicalFPEN
   ```

2. 根据实际情况修改 `submit.sh` 中的队列、资源、VASP 路径等参数。

3. 提交任务：
   ```bash
   sbatch submit.sh
   ```

4. 计算完成后，可在当前目录查看 `OUTCAR`、`vasprun.xml` 等输出文件。

## 适用场景

这些算例及其脚本主要用于 **生成第一性原理计算数据**，以支持后续 AI 模型（如势函数、性质预测模型等）的训练与验证。用户可在此基础上扩展至更复杂的体系或更大规模的计算任务。
