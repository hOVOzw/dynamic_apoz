#!/usr/bin/env python3
"""
ResNet18 参数约束剪枝工具
===========================

使用 ApoZ + 参数量约束的自动剪枝方法，自动搜索最优剪枝方案。
专为边缘设备部署设计，直接减少模型参数量。

使用方法:
    # 参数约束模式 (推荐) - 减少 50% 参数量
    python prune.py --classes 0 1 --batch-size 128 --flops-reduction 0.5
    
    # 减少 70% 参数量
    python prune.py --classes 0 1 --batch-size 128 --flops-reduction 0.7
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import csv

# ============================================================================
# 导入 ResNet 模型和剪枝模块
# ============================================================================
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models.resnet import resnet18, BasicBlock
from src.pruning.apoz import apply_pruning_masks as apply_structured_pruning, print_model_size


# ============================================================================
# 参数量剪枝核心代码
# ============================================================================

def compute_layer_params(module: nn.Module) -> Dict[str, int]:
    """计算模型各层的参数量"""
    params_dict = {}
    
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Conv2d):
            # Conv2d 参数量: out_channels * in_channels * kernel_h * kernel_w + bias
            num_params = layer.out_channels * layer.in_channels * \
                        layer.kernel_size[0] * layer.kernel_size[1]
            if layer.bias is not None:
                num_params += layer.out_channels
            params_dict[name] = num_params
        elif isinstance(layer, nn.BatchNorm2d):
            # BatchNorm2d 参数量: 2 * num_features (weight + bias)
            num_params = 2 * layer.num_features
            params_dict[name] = num_params
        elif isinstance(layer, nn.Linear):
            # Linear 参数量: in_features * out_features + bias
            num_params = layer.in_features * layer.out_features
            if layer.bias is not None:
                num_params += layer.out_features
            params_dict[name] = num_params
    
    return params_dict


def compute_model_params(module: nn.Module) -> int:
    """计算模型总参数量"""
    return sum(p.numel() for p in module.parameters())


def compute_layer_flops(module: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 32, 32)) -> Dict[str, float]:
    """计算模型各层的 FLOPs（保留用于参考）"""
    flops_dict = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(module, nn.Conv2d):
                kernel_h, kernel_w = module.kernel_size
                out_channels = module.out_channels
                in_channels = module.in_channels
                out_h, out_w = output.shape[2], output.shape[3]
                flops = 2 * out_channels * kernel_h * kernel_w * in_channels * out_h * out_w
                flops_dict[name] = flops
        return hook
    
    hooks = []
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    device = next(module.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)
    with torch.no_grad():
        module(dummy_input)
    
    for hook in hooks:
        hook.remove()
    
    return flops_dict


def compute_model_flops(module: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 32, 32)) -> float:
    """计算模型总 FLOPs（保留用于参考）"""
    layer_flops = compute_layer_flops(module, input_shape)
    return sum(layer_flops.values())


def safe_prune_channels(original_channels: int, target_prune_ratio: float, 
                        min_channels: int = 8, divisor: int = 8) -> Tuple[int, float]:
    """
    安全剪枝通道数计算
    """
    if original_channels < min_channels:
        return original_channels, 0.0
    
    raw_keep = int(original_channels * (1 - target_prune_ratio))
    aligned_keep = int(round(raw_keep / divisor)) * divisor
    aligned_keep = max(aligned_keep, divisor)
    final_keep = max(min_channels, aligned_keep)
    final_keep = min(final_keep, original_channels)
    
    actual_prune_ratio = 1.0 - (final_keep / original_channels)
    return final_keep, actual_prune_ratio


def get_layer_channel_info(model: nn.Module) -> Dict[str, Tuple[int, int]]:
    """获取各层的输入输出通道数"""
    channel_info = {}
    channel_info['conv1'] = (3, 64)
    
    for i in range(2):
        channel_info[f'layer1.{i}.conv1'] = (64, 64)
        channel_info[f'layer1.{i}.conv2'] = (64, 64)
    
    for i in range(2):
        channel_info[f'layer2.{i}.conv1'] = (64 if i == 0 else 128, 128)
        channel_info[f'layer2.{i}.conv2'] = (128, 128)
    
    for i in range(2):
        channel_info[f'layer3.{i}.conv1'] = (128 if i == 0 else 256, 256)
        channel_info[f'layer3.{i}.conv2'] = (256, 256)
    
    for i in range(2):
        channel_info[f'layer4.{i}.conv1'] = (256 if i == 0 else 512, 512)
        channel_info[f'layer4.{i}.conv2'] = (512, 512)
    
    return channel_info


def compute_apoz_representative(model: nn.Module, test_loader: DataLoader, 
                                  device: torch.device) -> Dict[str, np.ndarray]:
    """
    计算模型各层 Conv 层的代表性 ApoZ 值
    返回每层一个代表性值，用于全局参数量约束排序
    """
    model.eval()
    layer_outputs = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            mean_activation = output.mean(dim=[0, 2, 3])
            layer_outputs[name] = mean_activation.detach()
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            model(inputs)
            break
    
    for hook in hooks:
        hook.remove()
    
    apoz_per_layer = {}
    for name, activations in layer_outputs.items():
        min_val = activations.min()
        max_val = activations.max()
        if max_val > min_val:
            normalized = (activations - min_val) / (max_val - min_val)
        else:
            normalized = torch.ones_like(activations)
        apoz_per_layer[name] = normalized.cpu().numpy()
    
    return apoz_per_layer


def find_apoz_threshold_for_params(
    model: nn.Module,
    apoz_values: Dict[str, np.ndarray],
    target_params_reduction: float,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32),
    min_channels: int = 8,
    max_iterations: int = 30
) -> Tuple[float, float, Dict[str, np.ndarray]]:
    """
    自动搜索能达成目标参数量减少的 ApoZ 阈值

    Args:
        model: 模型
        apoz_values: 各层 Filter 的归一化 ApoZ 值
        target_params_reduction: 目标参数量减少比例 (0~1)
        input_shape: 输入形状
        min_channels: 最小通道数
        max_iterations: 最大迭代次数

    Returns:
        (最佳阈值, 实际参数量减少比例, 剪枝掩码)
    """
    from src.pruning.apoz import APoZ

    # 计算原始参数量
    original_params = compute_model_params(model)
    target_params = original_params * (1 - target_params_reduction)
    print(f"原始参数量: {original_params / 1e6:.2f} M")
    print(f"目标参数量: {target_params / 1e6:.2f} M (减少 {target_params_reduction*100:.1f}%)")

    # 创建 APoZ 实例（用于获取层索引）
    apoz_instance = APoZ(model)

    # 预处理 apoz_values，确保是字典格式
    if hasattr(apoz_instance, 'layer_apoz'):
        apoz_instance.layer_apoz = apoz_values

    def get_masks_for_threshold(threshold: float) -> Dict[str, np.ndarray]:
        """根据阈值生成剪枝掩码"""
        masks = {}
        for layer_idx, (layer_name, apoz) in enumerate(apoz_values.items()):
            # 计算该层的阈值（与 get_pruning_indices 相同的逻辑）
            layer_threshold = (threshold - layer_idx) / 100

            # 选择 ApoZ 值小于阈值的通道（低 ApoZ = 较少零激活 = 重要 = 保留）
            # 注意：True = 保留，False = 剪枝
            mask = apoz < layer_threshold

            # 确保至少保留 min_channels 个通道
            if mask.sum() < min_channels:
                indices = np.argsort(apoz)[:min_channels]
                new_mask = np.zeros_like(mask)
                new_mask[indices] = True
                mask = new_mask

            # 如果层通道数少于 min_channels，保留所有
            if len(apoz) < min_channels:
                mask = np.ones(len(apoz), dtype=bool)

            masks[layer_name] = mask

        return masks

    def calculate_params_reduction(masks: Dict[str, np.ndarray]) -> float:
        """计算剪枝后的参数量减少比例"""
        try:
            pruned_model = apply_structured_pruning(model, masks, num_classes=10)
            pruned_model = pruned_model.cpu()  # 确保在 CPU 上计算参数量
            pruned_params = compute_model_params(pruned_model)
            reduction = 1 - (pruned_params / original_params)

            # 调试：打印剪枝详情
            if reduction > 0:
                kept = sum(m.sum() for m in masks.values())
                total = sum(len(m) for m in masks.values())
                print(f"    DEBUG: 参数量减少={reduction*100:.1f}%, 保留通道={kept}/{total}")

            return reduction
        except Exception as e:
            print(f"    ERROR: {str(e)[:80]}...")
            return 0.0

    # 二分搜索最佳阈值
    low, high = 0.0, 100.0  # 阈值范围扩大
    best_threshold = 50.0
    best_masks = None
    best_reduction = 0.0

    print(f"\n开始自动搜索最佳 ApoZ 阈值...")

    for iteration in range(max_iterations):
        mid = (low + high) / 2

        # 生成掩码
        masks = get_masks_for_threshold(mid)

        # 计算参数量减少比例
        reduction = calculate_params_reduction(masks)

        # 打印进度（减少打印频率）
        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(f"  迭代 {iteration+1:2d}: 阈值={mid:.2f}, 参数量减少={reduction*100:.1f}%")

        # 判断是否达到目标
        # 注意：高阈值 = 少剪枝 = 高参数量；低阈值 = 多剪枝 = 低参数量
        if reduction >= target_params_reduction:
            # 达到目标，可以保守一些 = 更高的阈值
            low = mid
        else:
            # 未达到目标，需要更多剪枝 = 更低的阈值
            high = mid

    # 最终结果
    final_threshold = (low + high) / 2
    final_masks = get_masks_for_threshold(final_threshold)

    # 计算最终参数量
    try:
        pruned_model = apply_structured_pruning(model, final_masks, num_classes=10)
        pruned_model = pruned_model.cpu()  # 确保在 CPU 上计算参数量
        final_params = compute_model_params(pruned_model)
        final_reduction = 1 - (final_params / original_params)
    except Exception as e:
        final_params = original_params
        final_reduction = reduction  # 使用最后一次计算的 reduction

    print(f"\n自动搜索完成!")
    print(f"  最佳阈值: {final_threshold:.2f}")
    print(f"  实际参数量: {final_params / 1e6:.2f} M (减少 {final_reduction*100:.1f}%)")

    return final_threshold, final_reduction, final_masks


def params_based_pruning_masks(
    apoz_values: Dict[str, np.ndarray],
    model: nn.Module,
    target_params_reduction: float,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32),
    min_channels: int = 8,
    divisor: int = 8
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    基于参数量的剪枝掩码生成
    
    Args:
        apoz_values: 各层 Filter 的 ApoZ 值
        model: 模型
        target_params_reduction: 目标参数量减少比例 (0~1)
        input_shape: 输入形状
        min_channels: 最小通道数
        divisor: 对齐因子
    
    Returns:
        (剪枝掩码字典, 统计信息)
    """
    # Step 1: 获取各层信息
    layer_params = compute_layer_params(model)
    layer_channel_map = get_layer_channel_info(model)
    original_total_params = compute_model_params(model)
    target_params = original_total_params * (1 - target_params_reduction)
    
    print(f"\n原始模型参数量: {original_total_params / 1e6:.2f} M")
    print(f"目标参数量: {target_params / 1e6:.2f} M")
    
    # Step 2: 收集所有层的得分信息
    layer_scores = []
    for layer_name, apoz in apoz_values.items():
        params = layer_params.get(layer_name, 0)
        if isinstance(layer_channel_map.get(layer_name), tuple):
            out_channels = layer_channel_map[layer_name][1]
        elif layer_name in [n for n, _ in model.named_modules()]:
            module = dict(model.named_modules()).get(layer_name)
            if hasattr(module, 'out_channels'):
                out_channels = module.out_channels
            else:
                out_channels = len(apoz)
        else:
            out_channels = len(apoz)
        
        for idx in range(len(apoz)):
            # 使用参数量作为权重 (ApoZ × Params)
            score = apoz[idx] * params
            layer_scores.append({
                'layer': layer_name,
                'idx': idx,
                'apoz': apoz[idx],
                'params': params,
                'score': score,
                'out_channels': out_channels
            })
    
    # Step 3: 按得分降序排序
    layer_scores.sort(key=lambda x: x['score'], reverse=True)
    
    # Step 4: 初始化所有层的保留掩码
    pruning_masks: Dict[str, np.ndarray] = {}
    for layer_name in layer_channel_map.keys():
        out_ch = layer_channel_map[layer_name][1]
        pruning_masks[layer_name] = np.ones(out_ch, dtype=bool)
    
    # 记录每层已剪掉的数量
    pruned_count: Dict[str, int] = {layer: 0 for layer in layer_channel_map.keys()}
    
    # Step 5: 迭代剪枝直到达到目标参数量
    current_params = original_total_params
    
    for item in layer_scores:
        layer_name = item['layer']
        idx = item['idx']
        
        if pruning_masks[layer_name][idx] == 0:
            continue  # 已经被标记为剪掉
        
        # 检查是否还能剪
        current_pruned = pruned_count[layer_name]
        original_ch = layer_channel_map[layer_name][1]
        channels_after = original_ch - current_pruned - 1
        
        if channels_after < min_channels:
            continue  # 达到最小通道数保护
        
        # 计算对齐后的保留数
        prune_ratio = (current_pruned + 1) / original_ch
        aligned_keep, _ = safe_prune_channels(original_ch, prune_ratio, min_channels, divisor)
        
        # 估算剪枝后的参数量
        estimated_prune_ratio = 1 - (aligned_keep / original_ch)
        estimated_params = original_total_params * (1 - estimated_prune_ratio)
        
        if estimated_params >= target_params:
            # 标记这个 filter 为剪掉
            pruning_masks[layer_name][idx] = 0
            pruned_count[layer_name] += 1
        else:
            # 达到目标，停止
            break
    
    # Step 6: 统计
    stats = {
        'original_params': original_total_params,
        'target_reduction': target_params_reduction,
        'actual_reduction': None,  # 稍后计算
        'layer_channel_map': layer_channel_map,
        'pruned_count': pruned_count,
        'pruning_masks': pruning_masks
    }
    
    return pruning_masks, stats


def compute_pruning_stats(model: nn.Module, pruning_masks: Dict[str, np.ndarray]) -> Tuple[int, Dict[str, int]]:
    """计算剪枝后的参数量和各层保留通道数"""
    layer_channel_map = get_layer_channel_info(model)
    
    kept_channels = {}
    total_original = 0
    total_kept = 0
    
    for layer_name, original_ch in layer_channel_map.items():
        if isinstance(original_ch, tuple):
            original_ch = original_ch[1]
        mask = pruning_masks.get(layer_name, np.ones(original_ch, dtype=bool))
        kept = int(mask.sum())
        kept_channels[layer_name] = kept
        total_original += original_ch
        total_kept += kept
    
    # 计算参数量
    original_params = compute_model_params(model)
    
    # 估算剪枝后的参数量 (基于保留通道数比例)
    keep_ratio = total_kept / total_original if total_original > 0 else 1.0
    # 参数量与通道数的平方成正比 (对于 Conv 层)
    pruned_params = original_params * (keep_ratio ** 1.5)
    
    actual_reduction = 1 - (pruned_params / original_params)
    
    stats = {
        'original_params': original_params,
        'pruned_params': pruned_params,
        'actual_reduction': actual_reduction,
        'kept_channels': kept_channels,
        'total_original': total_original,
        'total_kept': total_kept
    }
    
    return int(pruned_params), stats


def print_pruning_stats(stats: Dict, min_channels: int, divisor: int) -> None:
    """打印剪枝统计信息"""
    print(f"\n{'='*60}")
    print("参数量约束剪枝统计")
    print(f"{'='*60}")
    print(f"目标参数量减少: {stats['target_reduction']*100:.1f}%")
    print(f"实际参数量减少: {stats['actual_reduction']*100:.1f}%")
    print(f"最小通道数保护: {min_channels}")
    print(f"对齐因子: {divisor}")
    print(f"\n各层详情:")
    print("-" * 60)
    
    channel_map = stats['layer_channel_map']
    kept = stats['kept_channels']
    
    for layer_name in sorted(channel_map.keys()):
        if isinstance(channel_map[layer_name], tuple):
            original = channel_map[layer_name][1]
        else:
            original = channel_map[layer_name]
        
        kept_ch = kept.get(layer_name, original)
        pruned = original - kept_ch
        
        if original > 0:
            ratio = pruned / original * 100
            print(f"{layer_name:20s}: {original:4d} -> {kept_ch:4d} (剪枝 {pruned:3d}, {ratio:5.1f}%)")


# ============================================================================
# 训练代码
# ============================================================================

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True, mode='acc'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.counter = 0
        self.best_metric = None
        self.best_weights = None
        self.early_stop = False

    def __call__(self, metric_value, model):
        if self.best_metric is None:
            self.best_metric = metric_value
            self.save_checkpoint(model)
        elif self.mode == 'loss' and metric_value < self.best_metric - self.min_delta:
            self.best_metric = metric_value
            self.counter = 0
            self.save_checkpoint(model)
        elif self.mode == 'acc' and metric_value > self.best_metric + self.min_delta:
            self.best_metric = metric_value
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print(f'早停触发，恢复最佳模型权重 ({self.mode}: {self.best_metric:.4f})')

    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


def get_class_weighted_dataloader(dataset, target_classes, batch_size, shuffle=True):
    """
    创建一个对目标类别和非目标类别使用不同权重的数据加载器
    目标类别权重 : 非目标类别权重 = 9 : 1
    """
    # 获取所有样本的索引
    all_indices = list(range(len(dataset)))
    targets = dataset.targets if hasattr(dataset, 'targets') else [t for _, t in dataset]

    # 将样本分为目标类和非目标类
    target_indices = []
    non_target_indices = []

    for i, target in enumerate(targets):
        if target in target_classes:
            target_indices.append(i)
        else:
            non_target_indices.append(i)

    total_samples = len(dataset)
    target_ratio = len(target_indices) / total_samples
    non_target_ratio = len(non_target_indices) / total_samples

    # 计算权重，使得目标类:非目标类 = 9:1
    target_weight = 9 / (9 * target_ratio + non_target_ratio)
    non_target_weight = 1 / (9 * target_ratio + non_target_ratio)

    print(f"目标类样本: {len(target_indices)}, 非目标类样本: {len(non_target_indices)}")
    print(f"目标类权重: {target_weight:.4f}, 非目标类权重: {non_target_weight:.4f}")

    # 设置权重
    weights = torch.ones(len(dataset))
    for idx in target_indices:
        weights[idx] = target_weight
    for idx in non_target_indices:
        weights[idx] = non_target_weight

    # 创建加权采样器
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )


def train(model, train_loader, criterion, optimizer, device, scaler=None):
    """训练函数，返回训练损失和准确率"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'    Batch[{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc


def validate(model, test_loader, criterion, device):
    """验证模型，返回验证损失和准确率"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_loss = total_loss / len(test_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def validate_class(model, test_loader, class_id, device):
    """验证特定类别的准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            mask = targets == class_id
            if mask.sum() == 0:
                continue
            
            inputs_masked = inputs[mask]
            targets_masked = targets[mask]
            
            outputs = model(inputs_masked)
            _, predicted = outputs.max(1)
            
            correct += predicted.eq(targets_masked).sum().item()
            total += targets_masked.size(0)
    
    if total == 0:
        return 0.0
    return 100. * correct / total


def validate_all_classes(model, test_dataset, device, num_classes=10, batch_size=128):
    """评估所有类别的准确率"""
    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for i in range(num_classes):
                mask = (targets == i)
                if mask.sum() > 0:
                    class_correct[i] += predicted[mask].eq(targets[mask]).sum().item()
                    class_total[i] += mask.sum().item()
    
    class_acc = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc[i] = 100. * class_correct[i] / class_total[i]
        else:
            class_acc[i] = 0.0
    
    return class_acc


def save_acc_to_csv(class_acc, args, result_path='result/prune_acc.csv'):
    """
    将剪枝后的模型各类别准确率保存到CSV文件（追加模式）
    """
    # 确保result目录存在
    result_dir = os.path.dirname(result_path)
    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    
    # 检查文件是否存在，决定是否写入表头
    file_exists = os.path.isfile(result_path)
    
    with open(result_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # 写入表头（如果文件不存在）
        if not file_exists:
            header = ['timestamp', 'classes', 'threshold', 'epochs', 'batch_size', 'lr'] + \
                     [f'class_{i}' for i in range(10)] + ['mean_acc']
            writer.writerow(header)
        
        # 获取时间戳
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 构建数据行
        classes_str = ' ' + '-'.join(map(str, args.classes))  # 前面加空格防止 Excel 解析为日期
        
        # 获取 reduction 或 threshold
        reduction_val = getattr(args, 'params_reduction', None)
        if reduction_val is None:
            reduction_val = args.threshold
        
        class_acc_list = [class_acc.get(i, 0.0) for i in range(10)]
        mean_acc = sum(class_acc_list) / len(class_acc_list)
        
        row = [timestamp, classes_str, reduction_val, args.epochs, args.batch_size, args.lr] + \
              [f'{acc:.2f}' for acc in class_acc_list] + [f'{mean_acc:.2f}']
        writer.writerow(row)
    
    print(f"准确率结果已保存到: {result_path}")


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='ResNet18 参数量约束剪枝工具 (自动模式)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 自动阈值搜索模式 (推荐) - 自动寻找合适阈值
    python prune.py --classes 0 1 --batch-size 128 --flops-reduction 0.8 --auto-search
"""
    )
    
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, required=True, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    
    # 剪枝模式
    parser.add_argument('--params-reduction', type=float, default=None,
                        help='目标参数量减少比例 (0~1)，例如 0.5 表示减少 50%%')
    parser.add_argument('--auto-search', action='store_true',
                        help='自动搜索能达成参数量目标的 ApoZ 阈值 (推荐)')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='剪枝阈值 (传统模式，当 --flops-reduction 未指定时使用)')
    
    # 通用参数
    parser.add_argument('--min-channels', type=int, default=8, 
                        help='每层最小保留通道数 (默认: 8)')
    parser.add_argument('--divisor', type=int, default=8,
                        help='通道对齐因子 (默认: 8)')
    parser.add_argument('--data-dir', type=str, default='data', help='数据目录')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='保存目录')
    parser.add_argument('--classes', type=int, nargs='+', required=True,
                        help='目标类别，例如: --classes 0 1')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # =========================================================================
    # 1. 加载 CIFAR-10 数据集
    # =========================================================================
    print("\n加载 CIFAR-10 数据集...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=False, transform=transform_test)

    # ========================================================================
    # 使用加权采样器进行训练（包含所有类别，目标类别权重更高）
    # ========================================================================
    train_loader = get_class_weighted_dataloader(train_dataset, args.classes, args.batch_size)
    print(f"训练集使用加权采样器，包含所有类别")

    # 验证使用完整测试集（不是筛选后的子集）
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"验证使用完整测试集 (10000 样本)")

    # 打印数据集信息
    all_targets = np.array(test_dataset.targets)
    for cls in args.classes:
        count = np.sum(all_targets == cls)
        print(f"  类别 {cls} 测试样本数: {count}")
    
    # =========================================================================
    # 2. 加载 ResNet18 模型
    # =========================================================================
    print("\n加载 ResNet18 模型...")
    model = resnet18()
    model = model.to(device)
    
    # 计算原始 FLOPs 和参数
    original_flops = compute_model_flops(model)
    original_params = compute_model_params(model)
    print(f"原始模型 FLOPs: {original_flops / 1e9:.2f} G")
    print(f"原始模型参数量: {original_params / 1e6:.2f} M")
    
    # =========================================================================
    # 3. 计算 ApoZ 并进行剪枝
    # =========================================================================
    print("\n计算模型各层代表性 ApoZ 值...")
    apoz_values = compute_apoz_representative(model, train_loader, device)

    if args.params_reduction is not None and args.auto_search:
        # 模式 1: 自动阈值搜索模式 (推荐)
        print(f"\n使用自动阈值搜索模式 (目标减少: {args.params_reduction*100:.1f}%)")
        from src.pruning.apoz import APoZ

        # 自动搜索能达成参数量目标的 ApoZ 阈值
        auto_threshold, actual_reduction, pruning_masks = find_apoz_threshold_for_params(
            model=model,
            apoz_values=apoz_values,
            target_params_reduction=args.params_reduction,
            input_shape=(1, 3, 32, 32),
            min_channels=args.min_channels
        )

        # 计算统计
        pruned_params, params_stats = compute_pruning_stats(model, pruning_masks)
        stats = {
            'target_reduction': args.params_reduction,
            'actual_reduction': actual_reduction,
            'auto_threshold': auto_threshold,
            'layer_channel_map': params_stats['kept_channels'],
            'kept_channels': params_stats['kept_channels']
        }
        print(f"\n剪枝后参数量: {pruned_params / 1e6:.2f} M (减少 {actual_reduction*100:.1f}%)")
        print(f"自动搜索到的阈值: {auto_threshold:.4f}")

    elif args.params_reduction is not None:
        # 模式 2: 原始参数量约束模式
        print(f"\n使用参数量约束模式 (目标减少: {args.params_reduction*100:.1f}%)")
        pruning_masks, stats = params_based_pruning_masks(
            apoz_values,
            model,
            target_params_reduction=args.params_reduction,
            input_shape=(1, 3, 32, 32),
            min_channels=args.min_channels,
            divisor=args.divisor
        )

        # 计算实际剪枝统计
        pruned_params, params_stats = compute_pruning_stats(model, pruning_masks)
        stats['actual_reduction'] = params_stats['actual_reduction']
        stats['kept_channels'] = params_stats['kept_channels']
        print_pruning_stats(stats, args.min_channels, args.divisor)
    else:
        # 模式 3: 传统阈值剪枝 (使用 apoz.py 中的方法)
        print(f"\n使用传统阈值模式 (阈值: {args.threshold})")
        from src.pruning.apoz import APoZ

        # 计算每个类别的 APoZ
        apoz = APoZ(model, train_dataset, device)
        target_apoz = apoz.compute_class_specific_apoz(args.classes)

        # 获取剪枝掩码
        pruning_masks = apoz.get_pruning_indices(
            target_apoz,
            threshold=args.threshold,
            min_channels=args.min_channels
        )

        # 计算统计
        pruned_params, params_stats = compute_pruning_stats(model, pruning_masks)
        stats = {
            'target_reduction': None,
            'actual_reduction': params_stats['actual_reduction'],
            'layer_channel_map': params_stats['kept_channels'],
            'kept_channels': params_stats['kept_channels']
        }
        print(f"\n剪枝后参数量: {pruned_params / 1e6:.2f} M (减少 {params_stats['actual_reduction']*100:.1f}%)")
    
    # =========================================================================
    # 4. 应用结构化剪枝 - 创建新模型架构
    # =========================================================================
    print("\n应用结构化剪枝 (创建新模型架构)...")
    pruned_model = apply_structured_pruning(model, pruning_masks, num_classes=10)
    pruned_model = pruned_model.to(device)
    
    # 打印剪枝后模型信息
    print_model_size(pruned_model, "剪枝后模型")
    
    pruned_params = compute_model_params(pruned_model)
    pruned_flops = compute_model_flops(pruned_model)
    print(f"\n剪枝后模型 FLOPs: {pruned_flops / 1e9:.2f} G (减少 {(1-pruned_flops/original_flops)*100:.1f}%)")
    print(f"剪枝后模型参数量: {pruned_params / 1e6:.2f} M (减少 {(1-pruned_params/original_params)*100:.1f}%)")
    
    # =========================================================================
    # 5. 微调剪枝后的模型
    # =========================================================================
    print(f"\n开始微调模型 ({args.epochs} 轮)...")
    
    # 设置损失函数和优化器
    class_weights = torch.ones(10).to(device)
    for cls in args.classes:
        class_weights[cls] = 5.0
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"使用加权损失函数，目标类别权重: 5.0")
    
    optimizer = optim.SGD(pruned_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 早停
    early_stopper = EarlyStopping(patience=args.patience, mode='acc')
    
    max_test_acc = 0.0
    
    for epoch in range(args.epochs):
        # 训练
        train_start = time.time()
        train_loss, train_acc = train(pruned_model, train_loader, criterion, optimizer, device)
        train_time = time.time() - train_start
        
        # 验证
        val_start = time.time()
        val_loss, val_acc = validate(pruned_model, test_loader, criterion, device)
        val_time = time.time() - val_start
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # 计算目标类别准确率
        print("\n特定类别评估结果：")
        target_class_accs = []
        for cls in args.classes:
            cls_acc = validate_class(pruned_model, test_loader, cls, device)
            print(f"  类别 {cls} 准确率: {cls_acc:.2f}%")
            target_class_accs.append(cls_acc)
        target_class_acc = np.mean(target_class_accs)
        print(f"  目标类别平均准确率: {target_class_acc:.2f}%")
        
        print(f'Epoch[{epoch + 1}/{args.epochs}] train_loss: {train_loss:.4f}, train_acc={train_acc:.4f}, '
              f'val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, max_test_acc={max_test_acc:.4f}')
        print(f'train time: {train_time:.3f}s, val time: {val_time:.3f}s')
        
        if target_class_acc > max_test_acc:
            max_test_acc = target_class_acc

            class_info = '_class' + '-'.join(map(str, args.classes))
            if args.params_reduction and args.auto_search:
                # 自动搜索模式：使用目标值作为 th 标识
                th_info = int(args.params_reduction * 100)
                save_name = f'pruned{class_info}_th{th_info}.pth'
            elif args.params_reduction:
                params_info = f'_params{int(args.params_reduction*100)}'
                save_name = f'pruned{class_info}{params_info}.pth'
            else:
                th_info = int(args.threshold * 100)
                save_name = f'pruned{class_info}_th{th_info}.pth'
            
            checkpoint = {
                'state_dict': pruned_model.state_dict(),
                'max_test_acc': max_test_acc
            }
            save_path = os.path.join(args.save_dir, save_name)
            torch.save(checkpoint, save_path)
            print(f'Model saved to {save_path}')
        
        # 早停检查
        early_stopper(target_class_acc, pruned_model)
        if early_stopper.early_stop:
            print(f'早停触发于第 {epoch+1} 轮')
            break
    
    print(f'Finish retrain the pruned model')
    
    # =========================================================================
    # 6. 最终评估并保存 CSV
    # =========================================================================
    print("\n" + "="*60)
    print("最终评估结果:")
    print("="*60)
    
    class_acc = validate_all_classes(pruned_model, test_dataset, device, num_classes=10, batch_size=args.batch_size)
    
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        print(f"  类别 {i} ({cifar10_classes[i]}): {class_acc[i]:.2f}%")
    
    mean_acc = sum(class_acc.values()) / 10
    print(f"  平均准确率: {mean_acc:.2f}%")
    
    # 保存到 CSV
    save_acc_to_csv(class_acc, args)
    
    print("\n" + "="*60)
    print("剪枝完成!")
    print("="*60)


if __name__ == "__main__":
    main()
