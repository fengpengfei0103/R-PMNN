# Dataset layout / 数据集结构

Only the aggregate training split and aggregate test split are required by the
retained R_PMNN and ablation code.

保留的 R_PMNN 与消融代码只需要聚合训练集和聚合测试集。

## Active data / 有效数据

| Split / 划分 | Metadata / 元数据 | Samples / 样本数 | Image directories / 图像目录 |
| --- | --- | ---: | ---: |
| Train / 训练 | `H15-JS5-T2-50.xlsx`, `Sheet2` | 1010 | 6 × 1010 PNG |
| Test / 测试 | `testfull/H15-JS5-T4-40.xlsx`, `Sheet2` | 808 | 6 × 808 PNG |

Every active PNG is 560 × 320 RGB. 

所有有效 PNG 均为 560 × 320 RGB。

The six image directories in each split are:

每个划分包含以下六个图像目录：

- `SaturabilityTraincopy`: initial saturation / 初始饱和度
- `PorePressureTraincopy`: initial pore pressure / 初始孔隙压力
- `EffectiveStressTraincopy`: initial effective stress / 初始有效应力
- `Saturabilitycopy`: target saturation / 目标饱和度
- `PorePressurecopy`: target pore pressure / 目标孔隙压力
- `EffectiveStresscopy`: target effective stress / 目标有效应力

`mask.png` is the soil-domain mask used by the weak-physics losses.
`mask.png` 是弱物理损失使用的土体区域掩膜。
