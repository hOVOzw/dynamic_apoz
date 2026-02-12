# ResNet18-PAPM: Parameter-based Adaptive Pruning Method (APoZ)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-blue.svg)](https://pytorch.org/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于 **Parameter-based Adaptive APoZ Pruning (PAPM)** 的卷积神经网络模型压缩框架。

## 核心创新

### 🎯 Parameter-based Adaptive APoZ Pruning (PAPM)

**PAPM** 是一种自适应的通道剪枝方法，通过分析模型**参数分布**与**激活特性**的联合信息，动态确定最优剪枝阈值，既能满足边缘设备对模型的体积要求，又能够通过APOZ较好的保持精度。相较于传统APOZ对不同神经网络的阈值需要手动探索，本项目的方法**PAPM**只需要计算好目标边缘模型体积，例如Resnet18大小为42MB左右，想要剪枝后控制在10MB以内，只需要把params-reduction参数设为0.77（-77%的体积），脚本能够自动找到对应的APOZ阈值匹配模型大小，更方便。

| 特性 | 传统 APoZ | PAPM (本文方法) |
|------|-----------|------------------|
| 评估维度 | 仅激活值 | 参数 + 激活 |
| 阈值设定 | 固定阈值 | 自适应阈值 |
| 剪枝粒度 | 全局 | 层自适应 |
| 创新点 | - | **自动寻找目标体积下的APOZ阈值不用多次尝试**|

### 技术原理

**核心思想**：用「参数重要性 + 激活稀疏性」联合评估通道重要性，自适应寻找目标模型体积对应的剪枝阈值。

**通俗理解**：

1. **Stage 1 - 参数分析**：每个卷积通道的权重越大、分布越集中，该通道越重要

2. **Stage 2 - APoZ 分析**：推理时，如果某个通道经常输出 0，说明这个通道「偷懒」了，可以剪掉

3. **Stage 3 - 自动找阈值**：给定目标压缩比例（如剪掉77%参数），自动调整 APoZ 阈值，精确匹配模型体积

**核心公式**：

**通道重要性评分：**
\[
\text{PAPM}_c = \alpha \cdot \|W_c\|_2 + (1-\alpha) \cdot (1 - \text{APoZ}_c)
\]

- \(W_c\)：第 \(c\) 个通道的权重
- \(\|W_c\|_2\)：权重 L2 范数（越大越重要）
- \(\text{APoZ}_c\)：该通道的零激活比例（越大越不重要）
- \(\alpha\)：权重系数（默认 0.5）

**自适应阈值搜索：**
\[
T = \text{APoZ}^{(k)} \quad \text{where} \quad
k = \text{round}(N \times \text{params\_reduction})
\]

- 找到排序后第 \(k\) 个 APoZ 值作为阈值
- \(N\)：通道总数
- `params-reduction`：目标压缩比例（如 0.77 表示剪掉 77%）

**通俗解释**：给每个通道打分，把低分的通道剪掉，直到模型体积达到目标。

## 项目结构

```
Net-P-Resnet18/
├── train.py              # 训练基础 ResNet18 模型
├── prune.py              # APoZ 剪枝和微调
├── fuse.py               # 联邦学习融合训练
├── infer.py              # 推理测试
├── deploy.py             # 设备分组分析（协方差聚类）
├── requirements.txt      # 依赖
├── data/                 # CIFAR-10 数据集
├── src/
│   ├── models/resnet.py  # ResNet18 模型定义
│   ├── training/         # 基础训练模块
│   ├── pruning/          # APoZ 剪枝模块 ⭐ 核心
│   ├── fusion/           # 联邦融合模块
│   ├── infer/            # 推理模块
│   └── deploy/           # 分组分析模块
├── checkpoints/          # 剪枝模型和 Fusion 模型保存目录
├── result/               # 实验结果
└── result-all/           # 用于存放完整实验结果
```

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

### 1. 训练基础模型

```bash
python train.py --epochs 200 --batch-size 128 --lr 0.1
```

### 2. 设备分组分析（协方差聚类）

```bash
python deploy.py
```

分析结果保存在 `result/deploy_inf.csv`，查看最佳分组方案。

**输出示例：**
```
Device 10: [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]  # 每个设备1类
Device 2: [[0, 1, 8, 9], [2, 3, 4, 5, 6, 7]]  # 2个设备
```

### 3. 剪枝训练

```bash
python prune.py --classes 0 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
python prune.py --classes 1 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
python prune.py --classes 2 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
python prune.py --classes 3 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
python prune.py --classes 4 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
python prune.py --classes 5 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
python prune.py --classes 6 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
python prune.py --classes 7 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
python prune.py --classes 8 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
python prune.py --classes 9 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search

python prune.py --classes 3 5 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
python prune.py --classes 0 8 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
python prune.py --classes 1 9 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
python prune.py --classes 4 7 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
python prune.py --classes 2 6 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
python prune.py --classes 0 8 1 9 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
python prune.py --classes 2 3 5 6 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
python prune.py --classes 2 3 4 5 6 7 --batch-size 128 --params-reduction 0.77 --epochs 20 --auto-search
```

**输出文件：** `checkpoints/pruned_classX_th77.pth`
**注意：** `为了方便才直接运行所有，最好在fusion前把与实验无关的模型放到tmp目录中，建议从device10开始，device2结束，节省实验时间`


### 4. 融合模型训练

```bash
python fuse.py --threshold 77 --epochs 200 --batch-size 128 --lr 0.0005 --patience 20
```

**功能说明：**
- 自动加载 `checkpoints/` 下所有 `pruned_class*_th77.pth` 文件
- 冻结特征提取器，只训练 Fusion 分类头
- 将多个剪枝模型的特征融合

**输出文件：** `checkpoints/fusion_model.pth`

### 5. 推理测试

```bash
# 剪枝模型 + Fusion 融合模型推理（仅需关注prune模型中最大的推理时间）
python infer.py --threshold 77 --split

# Base模型（Resnet18）推理
cd result-all
python evaluate_baseline.py
```

**参数说明：**
- `--threshold`: 剪枝阈值，用于定位模型文件
- `--split`: 启用 Split 模式（多模型 + Fusion）
- `--fusion-path`: Fusion 模型路径 (默认: `checkpoints/fusion_model.pth`)

## 检查点命名规则

| 类型 | 文件名格式 | 说明 |
|------|-----------|------|
| 基础模型 | `best_model.pth` | 完整 ResNet18 |
| 剪枝模型 | `pruned_classX-Y-Z_th77.pth` | 多类剪枝模型 |
| Fusion模型 | `fusion_model.pth` | 融合模型 |

## 实验结果

### Baseline (ResNet18)

| 指标 | 数值 |
|------|------|
| Mean Acc | **95.59%** |
| 推理时间 | **3.20 ms/样本** |
| 参数量 | ~11M |
| 模型大小 | 42.7 MB |

**各类别准确率**：

| Class | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-------|---|---|---|---|---|---|---|---|---|---|
| Acc% | 96.9 | 98.5 | 94.3 | 90.0 | 96.6 | 91.5 | 96.9 | 97.0 | 97.0 | 97.2 |

---

### Fusion (融合)

| 设备分组 | Best Acc | Mean Acc | 推理时间(ms/样本) |
|----------|----------|----------|-------------------|
| device2 | **90.17%** | **89.88%** | 3.0 |
| device3 | 89.78% | 89.56% | 3.3 |
| device4 | 89.30% | 89.00% | 3.5 |
| device5 | 89.36% | 89.14% | 3.4 |
| device6 | 88.77% | 88.49% | 3.3 |
| device7 | 88.27% | 88.10% | 3.4 |
| device8 | 87.60% | 87.38% | 3.4 |
| device9 | 87.24% | 86.83% | 3.4 |
| device10 | 85.64% | 85.63% | 3.3 |

**训练配置：** th=77, epochs=200, lr=0.0005

**推理时间说明：** 按各分组中 pmodel 的最大推理时间计算（单样本推理）。

---

### 剪枝效果 (th=77)

**单类剪枝（每个模型负责1个类别）：**

| 剪枝模型 | 目标类别 | 目标类准确率 | Mean Acc |
|----------|----------|-------------|----------|
| class0 | 0 | **96.70%** | 82.55% |
| class1 | 1 | **98.60%** | 84.23% |
| class2 | 2 | **93.40%** | 78.31% |
| class3 | 3 | **92.10%** | 76.92% |
| class4 | 4 | **95.60%** | 81.32% |
| class5 | 5 | **93.90%** | 81.32% |
| class6 | 6 | **98.20%** | 78.73% |
| class7 | 7 | **96.60%** | 86.08% |
| class8 | 8 | **97.60%** | 83.38% |
| class9 | 9 | **96.10%** | 80.33% |

**分组剪枝（每个模型负责多个类别）：**

| 分组 | Acc% |
|------|------|
| 4-7 | 81.98% |
| 2-3-4-5-6-7 | 80.70% |
| 3-5 | 79.91% |
| 0-8 | 79.89% |
| 2-6 | 79.72% |
| 1-9 | 79.09% |
| 2-3-5-6 | 74.84% |
| 0-8-1-9 | 73.48% |

---

### 模型大小对比

| 模型类型 | 文件大小 | 压缩比 |
|----------|----------|--------|
| Baseline | 42.7 MB | 1x |
| Pruned | 9.88 MB | ~0.23x |
| Fusion | 506 KB - 3.6 MB | ~0.01x - 0.08x |

**结论：** 剪枝后模型大小约为原始的 23%，Fusion 模型进一步压缩至约 1%-8%。

## 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习框架 | PyTorch + torchvision |
| 数据集 | CIFAR-10 |
| **剪枝方法** | **Parameter-based Adaptive APoZ Pruning (PAPM)** |
| 特征融合 | Feature Fusion |
| 设备分组 | 协方差聚类 |

## 参考资料

- **APoZ 原始论文**：Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures
- **PAPM 创新点**：参数-激活联合驱动的自适应剪枝策略
- **ResNet 原始论文**：Deep Residual Learning for Image Recognition

## 许可证

MIT License
