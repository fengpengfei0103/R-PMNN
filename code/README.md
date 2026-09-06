# R_PMNN reservoir-fluctuation experiments / R_PMNN 库水波动实验

This directory contains one shared R_PMNN implementation, the complete M/S/L
ablation workflow, and a read-only dataset auditor. Generated checkpoints,
logs, figures, and result tables are intentionally excluded from source control.

本目录只保留一份共享的 R_PMNN 实现、完整的 M/S/L 消融流程，以及只读数据审核工具。
检查点、日志、训练图片和结果表均属于可再生成产物，不纳入源码版本控制。

## Files / 文件

| File / 文件 | Purpose / 用途 |
| --- | --- |
| `R_PMNN.py` | Full R_PMNN model plus independent training/prediction entrypoint / 完整 R_PMNN 模型及独立训练预测入口 |
| `slope_experiment.py` | Config-driven data alignment, training, prediction, and metrics / 基于配置的数据对齐、训练、预测与指标计算 |
| `ablation_experiment.py` | Generate, run, and summarize all six ablation cases / 生成、运行并汇总六组消融实验 |
| `audit_dataset.py` | Check metadata/image alignment and PNG validity without changing data / 只读检查元数据、图像对齐和 PNG 有效性 |
| `configs/full_template.json` | Single source configuration used to generate case configs / 用于生成各实验配置的唯一模板 |

## R_PMNN inputs and outputs / R_PMNN 输入与输出

Inputs are the initial saturation image, reservoir-operation features, initial
pore-pressure image, and initial effective-stress image. Outputs are the three
future physical-field images and the stability factor.

输入为初始饱和度图、库水位运行特征、初始孔隙压力图和初始有效应力图；输出为三个
未来物理场图像及稳定性系数。

## Independent full model / 独立完整模型

Train and predict with the full M=1, S=1, L=1 model:

使用完整的 M=1、S=1、L=1 模型训练并预测：

```powershell
python code\R_PMNN.py --action train,predict
```

Predict with an existing 100-epoch checkpoint:

使用已有的 100 轮检查点进行预测：

```powershell
python code\R_PMNN.py --action predict --checkpoint code\checkpoints\full\R_PMNN_full_epoch_100.pth
```

The independent runner trains on `dataset/H15-JS5-T2-50.xlsx` with the six
training image directories and predicts on `dataset/testfull`.

独立入口使用 `dataset/H15-JS5-T2-50.xlsx` 及六个训练图像目录训练，并在
`dataset/testfull` 上预测。

## Ablation cases / 消融实验组

| Case | M: mechanism features / 机理特征 | S: physical structure / 物理结构 | L: weak-physics loss / 弱物理损失 |
| --- | ---: | ---: | ---: |
| `base` | 0 | 0 | 0 |
| `base_m` | 1 | 0 | 0 |
| `wo_l` | 1 | 1 | 0 |
| `full` | 1 | 1 | 1 |
| `wo_m` | 0 | 1 | 1 |
| `wo_s` | 1 | 0 | 1 |

M adds `SIGNED_RATE`, `DELTA_WATER_LEVEL`, and `FINAL_WATER_LEVEL`. S enables
pore-pressure/effective-stress residual prediction and branch skip connections.
L enables the incremental-field loss and weak pore-diffusion residual loss.

M 增加 `SIGNED_RATE`、`DELTA_WATER_LEVEL` 和 `FINAL_WATER_LEVEL`；S 启用孔压/有效
应力残差预测及分支跳连；L 启用增量场损失和弱孔压扩散残差损失。

```powershell
# List cases / 查看实验组
python code\ablation_experiment.py list

# Generate configs / 生成配置
python code\ablation_experiment.py make-configs

# Run all cases / 运行全部实验
python code\ablation_experiment.py run --cases all --action train,predict

# Reuse existing checkpoints / 复用已有检查点
python code\ablation_experiment.py run --cases all --action train,predict --skip-trained

# Aggregate metrics / 汇总指标
python code\ablation_experiment.py summarize
```

## Dataset audit / 数据审核

```powershell
python code\audit_dataset.py
```

A valid audit has no missing, extra, unreadable, duplicated, or null active
records. Add `--skip-image-verify` for a faster metadata-only check.

有效数据应不存在缺失、额外、不可读取、重复或空值记录。仅检查元数据时可增加
`--skip-image-verify`。

## Environment / 环境

Python 3.10 or later is recommended. Install dependencies with:

建议使用 Python 3.10 或更高版本，并安装依赖：

```powershell
pip install -r requirements.txt
```

Training requires substantial GPU memory. Checkpoints and generated outputs are
written below `code/checkpoints`, `code/results`, and `code/logs`.

训练需要较大的 GPU 显存。检查点和生成结果分别写入 `code/checkpoints`、
`code/results` 和 `code/logs`。
