# R-PMNN (Physics-mechanism-guided deep learning for predicting path-dependent multiphysics evolution and slope stability under reservoir water-level fluctuations)

### 面向库水位波动的多物理场与边坡稳定性联合预测

**Reservoir-level Fluctuation · Multi-physical-field Prediction · Slope Stability · Ablation Study**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)
![Data](https://img.shields.io/badge/Dataset-10%2C908%20images-2E8B57)
![Experiments](https://img.shields.io/badge/Ablation-6%20cases-7B61FF)

一套用于预测库水位升降过程中边坡饱和度、孔隙压力、有效应力和稳定性系数的
物理—机理融合神经网络及完整实验流程。

</div>

---

## 目录

- [项目简介](#项目简介)
- [主要特点](#主要特点)
- [任务定义](#任务定义)
- [模型结构](#模型结构)
- [项目结构](#项目结构)
- [数据集说明](#数据集说明)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [运行独立 R_PMNN](#运行独立-r_pmnn)
- [运行消融实验](#运行消融实验)
- [配置参数](#配置参数)
- [输出结果](#输出结果)
- [评价指标](#评价指标)
- [数据审核](#数据审核)
- [实验复现](#实验复现)
- [常见问题](#常见问题)

## 项目简介

库水位升降会改变边坡内部的渗流场和应力场，并进一步影响边坡稳定性。R_PMNN
同时接收初始物理场图像与库水位运行条件，通过共享编码、机理特征调制、孔压/应力
控制分支和弱物理损失，联合完成以下四项预测：

1. 目标时刻的边坡饱和度场；
2. 目标时刻的孔隙压力场；
3. 目标时刻的有效应力场；
4. 目标时刻的边坡稳定性系数。

仓库提供两条完整运行路径：

- **独立完整模型**：直接运行 `code/R_PMNN.py`，完成 Full 模型训练与预测；
- **消融实验流程**：通过统一配置生成并运行六组 M/S/L 实验，自动保存日志、预测结果并汇总指标。

## 主要特点

| 特点 | 说明 |
| --- | --- |
| 多任务联合预测 | 在同一网络中预测饱和度、孔隙压力、有效应力与稳定性系数 |
| 机理感知特征 | 从库水位、方向、速率和历时推导有符号速率、水位变化量与最终水位 |
| 物理结构嵌入 | 为孔压和有效应力设置独立控制分支、残差预测及跳跃连接 |
| 弱物理约束 | 使用物理场增量一致性与弱孔压扩散残差约束训练过程 |
| 完整消融实验 | 同时支持完整模型移除实验与从基础模型逐步添加组件的实验 |
| 严格数据对齐 | 根据 Excel 中的样本名逐项匹配六类图像，启动前检查缺失与数量不一致 |
| 可复现训练 | 固定随机种子，并启用 PyTorch/CUDA 确定性设置 |
| 数据质量审核 | 内置只读审核工具，可检查空值、重复、缺失、额外及损坏图像 |

## 任务定义

### 模型输入

| 输入 | 代码符号 | 形态 | 含义 |
| --- | --- | --- | --- |
| 初始饱和度图像 | `x1` | RGB，3 × 320 × 560 | 库水位变化前的边坡饱和状态 |
| 库水位运行特征 | `x2` | 4 或 7 维向量 | 初始水位、升降方向、速率、历时及派生特征 |
| 初始孔隙压力图像 | `x3` | RGB，3 × 320 × 560 | 库水位变化前的孔隙压力场 |
| 初始有效应力图像 | `x4` | RGB，3 × 320 × 560 | 库水位变化前的有效应力场 |

### 库水位运行特征

| 字段 | 含义 | 计算方式 |
| --- | --- | --- |
| `INITIAL_WATER_LEVEL` | 初始库水位 | Excel 原始字段 |
| `DIRECTION` | 水位变化方向 | `-1` 表示下降，`1` 表示上升 |
| `RATE` | 水位变化速率 | Excel 原始字段 |
| `DURATION` | 变化历时 | 由 Excel 中的 `TIME` 映射得到 |
| `SIGNED_RATE` | 有符号变化速率 | `DIRECTION × RATE` |
| `DELTA_WATER_LEVEL` | 水位变化量 | `SIGNED_RATE × DURATION` |
| `FINAL_WATER_LEVEL` | 目标时刻水位 | `INITIAL_WATER_LEVEL + DELTA_WATER_LEVEL` |

### 模型输出

| 输出 | 代码符号 | 含义 |
| --- | --- | --- |
| 目标饱和度图像 | `outputs1` | 目标时刻的边坡饱和度场 |
| 目标孔隙压力图像 | `outputs2` | 目标时刻的孔隙压力场 |
| 目标有效应力图像 | `outputs3` | 目标时刻的有效应力场 |
| 稳定性索引 | `outputs5` | 用于恢复稳定性系数的离散索引 |

稳定性索引按照以下序列转换为实际稳定性系数：

```text
Fr = 0.93 + index × 0.0001
```

代码会对预测索引进行四舍五入和范围裁剪，再从 `0.93` 至 `1.30` 的序列中读取
对应稳定性系数。

## 模型结构

### 主要组成

- **共享编码器**：使用卷积、残差块和多尺度注意力模块提取初始饱和度特征；
- **机理特征模块**：把库水位运行向量映射到空间特征图，并对图像特征进行调制；
- **孔压与应力控制分支**：分别编码初始孔隙压力和初始有效应力；
- **多尺度解码器**：逐级恢复空间分辨率，并输出三个独立 RGB 物理场；
- **残差场预测**：完整结构预测孔压/应力相对初始场的有限增量，降低无约束漂移；
- **稳定性分支**：融合孔压、应力和运行条件特征，输出稳定性索引；
- **不确定性加权**：学习图像任务与稳定性任务之间的共享权重。

### 训练损失

完整模型的总损失由以下部分构成：

```text
总损失
├── 饱和度场损失：MSE + 梯度一致性
├── 孔隙压力场损失：MSE + 梯度一致性 + TV 正则
├── 有效应力场损失：MSE + 梯度一致性 + TV 正则
├── 稳定性索引 MSE
├── 物理场增量一致性损失
└── 弱孔压扩散残差损失
```

弱物理损失在预热阶段后逐步增加权重，避免训练初期物理项压制数据拟合。

## 项目结构

```text
R_PMNN-reservoir-fluctuation/
├── README.md                         # GitHub 项目主页
├── requirements.txt                 # Python 依赖
├── .gitignore                       # 忽略检查点、日志、结果和缓存
├── .gitattributes                   # Git LFS 数据规则
├── code/
│   ├── R_PMNN.py                    # 完整模型及独立运行入口
│   ├── slope_experiment.py          # 配置驱动的训练、预测与评价流程
│   ├── ablation_experiment.py       # 消融配置生成、运行与汇总
│   ├── audit_dataset.py             # 只读数据完整性审核
│   ├── README.md                    # 代码目录简要说明
│   └── configs/
│       └── full_template.json       # 消融实验配置模板
└── dataset/
    ├── H15-JS5-T2-50.xlsx           # 训练元数据，读取 Sheet2
    ├── mask.png                     # 弱物理损失使用的土体区域掩膜
    ├── SaturabilityTraincopy/       # 训练：初始饱和度
    ├── PorePressureTraincopy/       # 训练：初始孔隙压力
    ├── EffectiveStressTraincopy/    # 训练：初始有效应力
    ├── Saturabilitycopy/            # 训练：目标饱和度
    ├── PorePressurecopy/            # 训练：目标孔隙压力
    ├── EffectiveStresscopy/         # 训练：目标有效应力
    ├── testfull/
    │   ├── H15-JS5-T4-40.xlsx       # 测试元数据，读取 Sheet2
    │   └── [同样的六类图像目录]
    └── README.md                    # 数据清理与审核说明
```

训练或预测后还会自动生成 `code/checkpoints/`、`code/results/` 和 `code/logs/`。
这些目录已写入 `.gitignore`，不会默认上传到 GitHub。

## 数据集说明

### 数据规模

| 划分 | 元数据 | 样本数 | 每样本图像数 | 物理场图像数 |
| --- | --- | ---: | ---: | ---: |
| 训练集 | `dataset/H15-JS5-T2-50.xlsx` | 1,010 | 6 | 6,060 |
| 测试集 | `dataset/testfull/H15-JS5-T4-40.xlsx` | 808 | 6 | 4,848 |
| **合计** | 2 个工作簿 | **1,818** | — | **10,908** |

- 图像格式：PNG；
- 图像尺寸：560 × 320；
- 图像模式：RGB；
- 土体掩膜：1 张 560 × 320 灰度 PNG；
- 当前 `dataset/` 总体积约 199.7 MB。

### 单个样本的六类图像

| 目录 | 角色 |
| --- | --- |
| `SaturabilityTraincopy` | 初始饱和度输入 |
| `PorePressureTraincopy` | 初始孔隙压力输入 |
| `EffectiveStressTraincopy` | 初始有效应力输入 |
| `Saturabilitycopy` | 目标饱和度标签 |
| `PorePressurecopy` | 目标孔隙压力标签 |
| `EffectiveStresscopy` | 目标有效应力标签 |

Excel 的 `name` 字段保存饱和度图像名称。代码通过名称替换规则定位其他物理场：

```text
S_ → P_    # 孔隙压力 / Pore pressure
S_ → E_    # 有效应力 / Effective stress
```

### 元数据字段

训练与测试均读取工作簿的 `Sheet2`，使用以下字段：

| 字段 | 用途 |
| --- | --- |
| `INITIAL_WATER_LEVEL` | 初始水位输入 |
| `DIRECTION` | 升降方向输入 |
| `RATE` | 水位变化速率输入 |
| `TIME` | 历时输入，运行时映射为 `DURATION` |
| `StabilizationFactor` | 实际稳定性系数，供结果核对 |
| `StabilizationFactorIndex` | 网络训练使用的稳定性标签 |
| `name` | 六类图像的对齐键 |
| `C`、`PHI` | 元数据保留字段，当前默认特征集不使用 |

更详细的数据清理说明见 [`dataset/README.md`](dataset/README.md)。

## 环境配置

### 建议环境

- Python 3.10 或更高版本；
- 支持当前 PyTorch 版本的 NVIDIA GPU 与 CUDA 环境；
- CPU 也可运行，但完整训练和全量预测耗时会明显增加；
- 建议为训练结果和临时图像预留充足磁盘空间。

### 1. 克隆项目

本项目的数据文件由 Git LFS 管理，克隆前请先安装
[Git LFS](https://git-lfs.com/)。

```bash
git lfs install
git clone <repository-url>
cd <repository-directory>
git lfs pull
```

### 2. 创建虚拟环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux / macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果需要特定 CUDA 版本，请先根据
[PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)安装对应的
`torch` 和 `torchvision`，再安装其余依赖。

## 快速开始

建议先审核数据，再启动完整模型：

```bash
# 1. 审核元数据与全部 PNG
python code/audit_dataset.py

# 2. 训练完整模型并在测试集上预测
python code/R_PMNN.py --action train,predict
```

默认参数为：

```text
seed = 60
epochs = 100
train batch size = 4
predict batch size = 1
learning rate = 0.001
```

## 运行独立 R_PMNN

`code/R_PMNN.py` 包含完整网络、训练损失、数据读取、预测和结果保存逻辑，默认使用
Full 设置，即 `M=1、S=1、L=1`。

### 训练并预测

```bash
python code/R_PMNN.py --action train,predict
```

### 仅训练

```bash
python code/R_PMNN.py --action train
```

### 使用已有检查点预测

```bash
python code/R_PMNN.py \
  --action predict \
  --checkpoint code/checkpoints/full/R_PMNN_full_epoch_100.pth
```

PowerShell 可写成单行：

```powershell
python code\R_PMNN.py --action predict --checkpoint code\checkpoints\full\R_PMNN_full_epoch_100.pth
```

### 自定义训练参数

```bash
python code/R_PMNN.py \
  --action train,predict \
  --epochs 100 \
  --batch-size 4 \
  --predict-batch-size 1 \
  --lr 0.001 \
  --output-dir code/results/full_standalone
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--action` | `train,predict` | 执行训练、预测或两者 |
| `--epochs` | `100` | 训练轮数；最终轮一定保存检查点 |
| `--batch-size` | `4` | 训练批量大小 |
| `--predict-batch-size` | `1` | 预测批量大小 |
| `--lr` | `0.001` | Adam 学习率 |
| `--output-dir` | `code/results/full_standalone` | 独立模型结果目录 |
| `--checkpoint` | 按轮数自动生成 | 自定义读取或保存的检查点路径 |

## 运行消融实验

### 组件定义

| 符号 | 组件 | 实现 |
| --- | --- | --- |
| M | 机理感知输入特征 | 增加 `SIGNED_RATE`、`DELTA_WATER_LEVEL`、`FINAL_WATER_LEVEL` |
| S | 物理结构嵌入 | 启用孔压/应力残差预测和对应分支跳连 |
| L | 弱物理损失 | 启用物理场增量损失和弱孔压扩散残差损失 |

### 六组实验

| `case_id` | M | S | L | 实验含义 |
| --- | ---: | ---: | ---: | --- |
| `base` | 0 | 0 | 0 | 基础数据驱动模型 |
| `base_m` | 1 | 0 | 0 | 基础模型加入机理特征 |
| `wo_l` | 1 | 1 | 0 | 完整结构去除弱物理损失 |
| `full` | 1 | 1 | 1 | 完整 R_PMNN |
| `wo_m` | 0 | 1 | 1 | 完整模型去除机理特征 |
| `wo_s` | 1 | 0 | 1 | 完整模型去除物理结构 |

结果可按两种顺序解读：

- **完整模型移除消融**：`full → wo_m / wo_s / wo_l`；
- **基础模型逐步加入**：`base → base_m → wo_l → full`。

### 查看实验组

```bash
python code/ablation_experiment.py list
```

### 生成六组配置

```bash
python code/ablation_experiment.py make-configs
```

生成的配置位于 `code/configs/`，并以 `full_template.json` 为唯一模板。

### 运行全部实验

```bash
python code/ablation_experiment.py run --cases all --action train,predict
```

### 运行指定实验

```bash
python code/ablation_experiment.py run --cases full,wo_m,wo_s,wo_l --action train,predict
```

### 仅训练或仅预测

```bash
python code/ablation_experiment.py run --cases full --action train
python code/ablation_experiment.py run --cases full --action predict
```

### 跳过已有检查点的训练

```bash
python code/ablation_experiment.py run \
  --cases all \
  --action train,predict \
  --skip-trained
```

### 汇总全部实验

```bash
python code/ablation_experiment.py summarize
```

长历时子集默认以 `DURATION >= 30` 划分，可自定义阈值：

```bash
python code/ablation_experiment.py summarize --long-duration-threshold 30
```

## 配置参数

完整配置模板位于 [`code/configs/full_template.json`](code/configs/full_template.json)。

### 当前完整模型的关键参数

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `seed` | `60` | 随机种子 |
| `image_loss_weights.saturation` | `1.0` | 饱和度场损失权重 |
| `image_loss_weights.pore_pressure` | `1.5` | 孔隙压力场损失权重 |
| `image_loss_weights.effective_stress` | `2.8` | 有效应力场损失权重 |
| `gradient_loss_weight` | `0.2` | 梯度一致性损失权重 |
| `tv_loss_weight` | `0.05` | 孔压/应力场 TV 正则权重 |
| `physics_warmup_epochs` | `20` | 弱物理损失预热轮数 |
| `delta_physics_loss_weight` | `0.05` | 物理场增量损失权重 |
| `pore_diffusion_loss_weight` | `0.0001` | 弱孔压扩散残差权重 |
| `pore_diffusivity` | `0.01` | 扩散残差中的扩散系数 |
| `pore_residual_scale` | `0.25` | 孔压残差最大缩放 |
| `stress_residual_scale` | `0.25` | 应力残差最大缩放 |
| `train.epochs` | `100` | 默认训练轮数 |
| `train.batch_size` | `4` | 默认训练批量 |
| `train.lr` | `0.001` | 默认学习率 |

除非要建立新的实验设计，建议通过模板和消融生成器统一修改参数，避免六组配置发生
非预期差异。

## 输出结果

### 独立模型输出

```text
code/
├── checkpoints/full/
│   ├── R_PMNN_full_epoch_100.pth
│   └── reservoir_feature_stats.json
└── results/full_standalone/
    ├── R_PMNN_full_saturation/
    ├── R_PMNN_full_pore_pressure/
    ├── R_PMNN_full_effective_stress/
    ├── R_PMNN_full_explainability/
    ├── R_PMNN_full_prediction_results.xlsx
    ├── R_PMNN_full_sample_metrics_results.xlsx
    ├── R_PMNN_full_explainability_results.xlsx
    ├── R_PMNN_full_metrics_results_summary.xlsx
    └── R_PMNN_fullResults.txt
```

### 消融实验输出

```text
code/
├── checkpoints/<case_id>/R_PMNN_<case_id>_epoch_100.pth
├── logs/<case_id>.log
└── results/
    ├── <case_id>/
    ├── ablation_all_cases_summary.csv
    ├── ablation_removal_table.csv
    ├── ablation_incremental_table.csv
    └── ablation_summary.xlsx
```

### 可解释性输出

每个样本会保存以下孔压与应力增量图：

- 预测增量图；
- 真实增量图；
- 增量绝对误差图。

相应工作簿还记录增量均值、平均绝对误差、均方误差、最大绝对误差以及 RGB 增量误差。

## 评价指标

| 任务 | 指标 | 说明 |
| --- | --- | --- |
| 饱和度场 | PSNR、SSIM | 图像重建质量与结构相似性 |
| 孔隙压力场 | PSNR、SSIM | 孔压场预测质量 |
| 有效应力场 | PSNR、SSIM | 应力场预测质量 |
| 稳定性系数 | MSE、MAE、R² | 数值预测误差与拟合程度 |
| 场增量解释 | Delta MAE/MSE、Mean Error | 相对初始场的变化预测误差 |

汇总程序同时计算各指标的均值、标准差、最小值、最大值，以及长历时样本上的单独指标。

## 数据审核

完整审核会读取两个工作簿并解码全部 10,908 张物理场 PNG：

```bash
python code/audit_dataset.py
```

只检查元数据和文件对应关系：

```bash
python code/audit_dataset.py --skip-image-verify
```

保存 JSON 审核报告：

```bash
python code/audit_dataset.py --output dataset_audit.json
```

审核内容包括：

- 元数据空值；
- 重复元数据行和重复样本名；
- 每类图像的预期数量与实际数量；
- 元数据引用但不存在的图像；
- 目录中存在但元数据未引用的额外图像；
- 无法由 Pillow 解码的 PNG；
- 图像尺寸与颜色模式分布。

当前有效数据已通过全量审核：没有缺失、额外、损坏、空值或重复记录，所有物理场
图像均为 560 × 320 RGB。

## 实验复现

为提高实验可复现性，代码默认执行以下设置：

- 固定 Python、NumPy、PyTorch 和 CUDA 随机种子为 `60`；
- 关闭 cuDNN benchmark，并启用确定性算法；
- 训练集统计量用于特征标准化，并随检查点保存；
- 测试集沿用训练集统计量，不单独拟合；
- 六组消融实验共用同一模板、数据划分和主要超参数；
- 图像按照元数据 `name` 字段对齐，不依赖文件系统遍历顺序。

> 不同操作系统、GPU、CUDA、cuDNN 和 PyTorch 版本仍可能产生轻微数值差异。发布
> 实验结果时，建议同时记录硬件、CUDA/PyTorch 版本和实际配置文件。

## 常见问题

<details>
<summary><strong>1. 提示 Checkpoint not found</strong></summary>

先运行训练，或通过 `--checkpoint` 指向实际存在的 `.pth` 文件。检查点文件名默认包含
训练轮数，例如 `R_PMNN_full_epoch_100.pth`。

</details>

<details>
<summary><strong>2. CUDA out of memory</strong></summary>

降低 `--batch-size`；消融实验则修改 `code/configs/full_template.json` 中的
`train.batch_size`，然后重新生成配置。预测显存不足时降低 `--predict-batch-size` 或
配置中的 `predict.batch_size`。

</details>

<details>
<summary><strong>3. 提示 Missing image files</strong></summary>

先运行 `python code/audit_dataset.py`。检查 Excel 的 `name` 字段、六个图像目录名称，
以及 `S_ → P_ / E_` 的命名映射是否被破坏。

</details>

<details>
<summary><strong>4. 安装 PyTorch 后仍无法使用 CUDA</strong></summary>

确认显卡驱动、CUDA 运行环境和 PyTorch 构建版本一致。建议使用 PyTorch 官方安装命令，
并通过 `python -c "import torch; print(torch.cuda.is_available())"` 检查。

</details>

## 相关文档

- [`code/README.md`](code/README.md)：代码入口与命令速查；
- [`dataset/README.md`](dataset/README.md)：数据布局、审核结论和已清理冗余说明；
- [`code/configs/full_template.json`](code/configs/full_template.json)：完整实验参数模板；
- [`requirements.txt`](requirements.txt)：运行依赖。

---

<div align="center">

**R_PMNN · Reservoir-level fluctuation prediction**
