# PINO

## 配置文件

`config` 目录包含许多命名格式为 `config_{1/2/3}D_{PDE name}.yaml` 的 `yaml` 配置文件。以下是一些 PINO 特定参数的说明：

* 训练参数:
    * `training_type`：字符串，`single` 指定 PINO 的非自回归处理。
    * `initial_step`：整数，输入时间步的数量。（默认：1）

* 模型参数:
    * `modes`：列表[int]，列表长度指定傅里叶块数，列表中值指定 FNO 架构中的傅里叶模式数。注意，存在多个模式需要设置，例如 3D 架构中有 `modes1`、`modes2`、`modes3`，如 3D 空间模型或 2D 空间 + 1D 时间模型，其中最后一维是时间维度。
    * `width`：整数，傅里叶层的通道数。
    * `in_channels`：整数，输入通道数，等于待求解变量的数量。例如，1D 可压缩 NS 方程有3个变量需要求解：密度、压力和速度。
    * `out_channels`：整数，输出通道数。

* 数据集参数:
    * `reduced_resolution`：整数，PINO 中数据驱动部分空间分辨率的下采样率。例如，数据空间分辨率为 1024，设 `reduced_resolution`=16，则数据驱动部分使用的分辨率为 1024/16=64。
    * `reduced_resolution_t`：整数，数据驱动部分时间维度的下采样率。
    * `reduced_resolution_pde`：整数，PINO 中物理驱动部分空间分辨率的下采样率。
    * `reduced_resolution_pde_t`：整数，物理驱动部分时间维度的下采样率。

    注意，我们针对 PINO 训练提供了空间维度上 4 倍的物理/数据分辨率比，时间维度上为 1 倍，因此默认 `reduced_resolution` 是 `reduced_resolution_pde` 的 4 倍。config目录下4x的空间分辨率会比

* 损失权重:
    * ic_loss：浮点数，默认 2.0，物理驱动部分初始条件损失的权重。
    * f_loss：浮点数，默认 1.0，物理驱动部分方程损失的权重。
    * xy_loss：浮点数，默认 10.0，数据驱动损失的权重。

我们使用的训练超参数如下：

| PDE 名称                    | 空间分辨率 / 下采样率（pde）      | 时间分辨率 / 下采样率           | 学习率 | 训练轮数 | 批大小 | 权重衰减 | 宽度 | 模式数 |
| :-------------------------- | :----------------------------- | :----------------------------- | :----- | :------- | :----- | :------- | :--- | :----- |
| 1D Advection                | 1024/4                        | 201/5                          | 1.e-3  | 500      | 50     | 1.e-4    | 32   | 12     |
| 1D Diffusion-Reaction       | 1024/4                        | 101/1                          | 1.e-3  | 500      | 50     | 1.e-4    | 32   | 12     |
| 1D Burgers                  | 1024/4                        | 201/5                          | 1.e-3  | 500      | 50     | 1.e-4    | 32   | 12     |
| 1D Diffusion-Sorption       | 1024/4                        | 101/1                          | 1.e-3  | 500      | 20     | 1.e-4    | 32   | 12     |
| 1D Allen Cahn               | 1024/4                        | 101/1                          | 1.e-3  | 500      | 50     | 1.e-4    | 32   | 12     |
| 1D Cahn Hilliard            | 1024/4                        | 101/1                          | 1.e-3  | 500      | 50     | 1.e-4    | 32   | 12     |
| 1D Compressible NS          | 1024/4                        | 101/1                          | 1.e-3  | 500      | 50     | 1.e-4    | 64   | 12     |
| 2D Burgers                  | 128/1                         | 101/1                          | 1.e-3  | 200      | 2      | 1.e-4    | 64   | 12     |
| 2D Compressible NS          | 128/1                         | 21/1                           | 1.e-3  | 200      | 8      | 1.e-4    | 64   | 12     |
| 2D DarcyFlow                | 128/1                         | -                             | 1.e-3  | 500      | 50     | 1.e-4    | 32   | 12     |
| 2D Shallow Water            | 128/1                         | 101/1                          | 1.e-3  | 200      | 2      | 1.e-4    | 64   | 12     |
| 2D Allen Cahn               | 128/1                         | 101/1                          | 1.e-3  | 200      | 2      | 1.e-4    | 64   | 12     |
| 2D Black-Scholes-Barenblatt | 128/1                         | 101/1                          | 1.e-3  | 200      | 2      | 1.e-4    | 64   | 12     |

## 损失函数

PINO 以非自回归方式求解 PDE，模型 $f_{\theta}$ 基于初始时间步（`initial_step`=1）解 $\hat{u}^0$，预测所有时间步的解集合 $\{\hat{u}^0 \dots \hat{u}^T\}$。

损失函数形式为：

$$
\mathcal{L} = \frac{1}{N_b} \sum_{i=1}^{N_b} \left( W_1 * l(u_{\text{pred}}, u) + W_2 * l(\mathcal{F}(u_{\text{pred}}), \mathcal{F}(u)) + W_3 * l(u^{0}_{\text{pred}}, u^{0}) \right)
$$

其中 $N_b$ 是批大小，$u$ 和 $u_{\text{pred}}$ 分别是真实解和预测解，$\mathcal{F}$ 表示 PDE 算子，$l$ 是我们实现中的均方误差（MSE）。$W_1$，$W_2$，$W_3$ 分别是数据损失、物理损失和初始条件损失的权重。

## 训练

1. 检查配置文件中的以下参数：
    1. `file_name` 和 `saved_folder` 路径是否正确；
    2. `if_training` 是否为 `True`；
2. 设置训练超参数，如学习率、批大小等。可以使用我们提供的默认值；
3. 运行命令：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/4x/${config_filename}
# 示例：CUDA_VISIBLE_DEVICES=0 python train.py ./config/train/4x/config_2D_Darcy_Flow.yaml
```

## 继续训练

1. 修改配置文件：
    1. 确保 `if_training` 为 `True`；
    2. 设置 `continue_training` 为 `True`；
    3. 将 `model_path` 设置为重新训练所用的检查点路径；
2. 运行命令：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/4x/${config_filename}
```

## 测试

1. 修改配置文件：
    1. 将 `if_training` 设置为 `False`；
    2. 将 `model_path` 设置为待评估模型的检查点路径；
2. 运行命令：
```bash
CUDA_VISIBLE_DEVICES=0 python train.py ./config/4x/${config_filename}