import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from ..models.resnet import BasicBlock, ResNet


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class FilterInfo:
    """单个 Filter 的信息"""
    layer_name: str           # 层名称，如 'layer1.0.conv1'
    filter_idx: int           # Filter 索引
    apoz: float               # ApoZ 值
    flops: float              # 该 Filter 的 FLOPs
    original_channels: int    # 该层的原始通道数
    is_pruned: bool = False   # 是否已被剪枝
    
    @property
    def score(self) -> float:
        """加权得分 = ApoZ × ΔFLOPs"""
        return self.apoz * self.flops


@dataclass
class LayerPruningPlan:
    """层的剪枝计划"""
    layer_name: str
    original_channels: int
    keep_channels: int        # 保护后的保留通道数
    prune_ratio: float        # 实际剪枝率
    filters_to_prune: List[int]  # 要剪掉的 filter 索引列表


# ============================================================================
# 工具函数
# ============================================================================

def compute_layer_flops(module: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 32, 32)) -> Dict[str, float]:
    """
    计算模型各层的 FLOPs
    
    Args:
        module: 神经网络模块
        input_shape: 输入形状 (batch, channels, height, width)
    
    Returns:
        Dict[层名称, FLOPs]
    """
    flops_dict = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            # 计算 FLOPs: 2 * output_channels * kernel_h * kernel_w * input_channels * output_h * output_w
            if isinstance(module, nn.Conv2d):
                kernel_h, kernel_w = module.kernel_size
                out_channels = module.out_channels
                in_channels = module.in_channels
                out_h, out_w = output.shape[2], output.shape[3]
                
                # FLOPs for one forward pass
                flops = 2 * out_channels * kernel_h * kernel_w * in_channels * out_h * out_w
                flops_dict[name] = flops
        return hook
    
    hooks = []
    for name, layer in module.named_modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    # 虚拟前向传播
    device = next(module.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)
    with torch.no_grad():
        module(dummy_input)
    
    # 移除 hooks
    for hook in hooks:
        hook.remove()
    
    return flops_dict


def compute_model_flops(module: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 32, 32)) -> float:
    """
    计算模型总 FLOPs
    
    Args:
        module: 神经网络模块
        input_shape: 输入形状
    
    Returns:
        总 FLOPs
    """
    layer_flops = compute_layer_flops(module, input_shape)
    return sum(layer_flops.values())


def get_layer_input_output_channels(model: nn.Module) -> Dict[str, Tuple[int, int]]:
    """
    获取各层的输入输出通道数
    
    Returns:
        Dict[层名称, (in_channels, out_channels)]
    """
    channel_info = {}
    
    # First layer
    channel_info['conv1'] = (3, 64)
    
    for i in range(2):  # layer1
        channel_info[f'layer1.{i}.conv1'] = (64, 64)
        channel_info[f'layer1.{i}.conv2'] = (64, 64)
    
    for i in range(2):  # layer2
        channel_info[f'layer2.{i}.conv1'] = (64 if i == 0 else 128, 128)
        channel_info[f'layer2.{i}.conv2'] = (128, 128)
    
    for i in range(2):  # layer3
        channel_info[f'layer3.{i}.conv1'] = (128 if i == 0 else 256, 256)
        channel_info[f'layer3.{i}.conv2'] = (256, 256)
    
    for i in range(2):  # layer4
        channel_info[f'layer4.{i}.conv1'] = (256 if i == 0 else 512, 512)
        channel_info[f'layer4.{i}.conv2'] = (512, 512)
    
    return channel_info


# ============================================================================
# 核心算法: 最小通道数保护函数
# ============================================================================

def safe_prune_channels(
    original_channels: int,
    target_prune_ratio: float,
    min_channels: int,
    divisor: int = 8
) -> Tuple[int, float]:
    """
    【核心安全函数】安全剪枝计算，包含最小通道数保护
    
    流程:
        1. 通道对齐 (四舍五入到 divisor 的倍数)
        2. 应用最小通道数约束: keep = max(min_channels, aligned_keep)
        3. 计算实际剪枝率
    
    Args:
        original_channels: 原始通道数
        target_prune_ratio: 目标剪枝比例 (0~1, 如 0.5 表示剪掉 50%)
        min_channels: 全局最小通道数保护阈值
        divisor: 对齐因子 (默认 8)
    
    Returns:
        Tuple[最终保留通道数, 实际剪枝率]
    
    Example:
        >>> safe_prune_channels(64, 0.5, min_channels=8, divisor=8)
        >>> (32, 0.5)  # 保留 32 通道，实际剪枝率 50%
        
        >>> safe_prune_channels(64, 0.9, min_channels=8, divisor=8)
        >>> (8, 0.875)  # 受保护，只能剪到 8 通道，实际剪枝率 87.5%
    """
    # Step 1: 原始计算要保留的通道数
    raw_keep = int(original_channels * (1 - target_prune_ratio))
    
    # Step 1.5: 【边界保护】如果原始通道数小于最小通道数，保留全部
    if original_channels < min_channels:
        return original_channels, 0.0
    
    # Step 2: 通道对齐 (四舍五入到 divisor 的倍数)
    aligned_keep = int(round(raw_keep / divisor)) * divisor
    
    # Step 3: 确保对齐后不为零
    aligned_keep = max(aligned_keep, divisor)
    
    # Step 4: 【最小通道数保护机制】
    # 这是最关键的一步：使用 max() 确保不会剪爆
    final_keep = max(min_channels, aligned_keep)
    
    # Step 5: 边界检查 (不超过原始通道数)
    final_keep = min(final_keep, original_channels)
    
    # Step 6: 计算实际剪枝率
    actual_prune_ratio = 1.0 - (final_keep / original_channels)
    
    return final_keep, actual_prune_ratio


# ============================================================================
# 核心算法: Filter 收集与排序
# ============================================================================

def collect_all_filters(
    apoz_values: Dict[str, np.ndarray],
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32)
) -> List[FilterInfo]:
    """
    收集所有层的所有 Filter 信息
    
    Args:
        apoz_values: 各层的 ApoZ 值字典
        model: 神经网络模型
        input_shape: 输入形状用于计算 FLOPs
    
    Returns:
        List[FilterInfo]: 所有 Filter 的信息列表
    """
    # 计算各层 FLOPs
    layer_flops = compute_layer_flops(model, input_shape)
    
    # 获取层通道信息
    channel_info = get_layer_input_output_channels(model)
    
    all_filters = []
    
    for layer_name, apoz in apoz_values.items():
        num_filters = len(apoz)
        
        # 获取该层的 FLOPs
        layer_flop = layer_flops.get(layer_name, 1e6)
        flops_per_filter = layer_flop / num_filters
        
        # 获取原始通道数
        _, out_channels = channel_info.get(layer_name, (64, 64))
        
        for idx in range(num_filters):
            filter_info = FilterInfo(
                layer_name=layer_name,
                filter_idx=idx,
                apoz=float(apoz[idx]),
                flops=flops_per_filter,
                original_channels=out_channels
            )
            all_filters.append(filter_info)
    
    return all_filters


def global_filter_ranking(filters: List[FilterInfo]) -> List[FilterInfo]:
    """
    全局 Filter 排序 (按加权得分降序)
    
    得分公式: Score(f_i) = ApoZ(f_i) × ΔFLOPs(f_i)
    
    排序逻辑:
        - ApoZ 越高 → 激活越少 → 越不重要 → 应该先剪掉
        - FLOPs 越大 → 剪掉后节省越多 → 应该优先剪掉
        - 两者相乘得到的得分越高 → 越应该被剪掉
    
    Args:
        filters: 所有 Filter 信息列表
    
    Returns:
        按得分降序排列的 Filter 列表
    """
    # 按得分降序排序 (得分越高越应该被剪掉)
    sorted_filters = sorted(filters, key=lambda x: x.score, reverse=True)
    
    return sorted_filters


# ============================================================================
# 核心算法: FLOPS 预算自动搜索
# ============================================================================

def flops_constraint_search(
    sorted_filters: List[FilterInfo],
    target_flops_reduction: float,
    original_total_flops: float,
    layer_channel_map: Dict[str, int],
    min_channels: int,
    divisor: int = 8
) -> Tuple[Dict[str, List[int]], float]:
    """
    【核心搜索函数】自动搜索满足 FLOPS 约束的剪枝方案
    
    算法流程:
        1. 从得分最高的 Filter 开始遍历
        2. 累加 FLOPs 节省量
        3. 检查是否达到目标预算
        4. 考虑最小通道数约束
    
    Args:
        sorted_filters: 已排序的 Filter 列表 (得分从高到低)
        target_flops_reduction: 目标 FLOPs 减少百分比 (如 0.5 表示减少 50%)
        original_total_flops: 原始模型总 FLOPs
        layer_channel_map: 各层原始通道数映射
        min_channels: 全局最小通道数
        divisor: 对齐因子
    
    Returns:
        Tuple[各层要剪掉的 Filter 索引字典, 实际 FLOPs 减少率]
    """
    target_flops = original_total_flops * (1 - target_flops_reduction)
    current_flops = original_total_flops
    
    # 记录各层已经被标记为剪掉的 filter 索引
    layer_pruned_indices: Dict[str, List[int]] = {layer: [] for layer in layer_channel_map.keys()}
    
    # 记录每个层当前已剪掉多少个 filter
    layer_pruned_count: Dict[str, int] = {layer: 0 for layer in layer_channel_map.keys()}
    
    # 遍历排序后的 filters
    for filter_info in sorted_filters:
        layer_name = filter_info.layer_name
        channel_info = layer_channel_map[layer_name]
        
        # 兼容: channel_info 可能是 int (只包含 out_channels) 或 tuple (in, out)
        if isinstance(channel_info, tuple):
            original_channels = channel_info[1]  # out_channels
        else:
            original_channels = channel_info
        
        pruned_count = layer_pruned_count[layer_name]
        
        # 计算如果剪掉这个 filter 后，该层还剩多少通道
        # 剪掉这个 filter 后，该层将保留: original_channels - pruned_count - 1
        channels_after_prune = original_channels - pruned_count - 1
        
        # 【最小通道数保护】检查剪掉后是否低于最小通道数
        # 只有当剪掉后仍然 >= min_channels 时才允许剪掉
        if channels_after_prune < min_channels:
            # 达到最小通道数限制，跳过这个 filter
            continue
        
        # 计算剪掉后的通道数（考虑对齐）
        # 这是一个简化的计算，实际应该使用 safe_prune_channels
        potential_keep = max(min_channels, channels_after_prune)
        
        # 检查是否满足对齐要求
        potential_keep = int(round(potential_keep / divisor)) * divisor
        
        if potential_keep < min_channels:
            continue
        
        # 标记这个 filter 需要被剪掉
        filter_info.is_pruned = True
        layer_pruned_indices[layer_name].append(filter_info.filter_idx)
        layer_pruned_count[layer_name] += 1
        
        # 更新当前 FLOPs
        current_flops -= filter_info.flops
        
        # 检查是否达到目标
        if current_flops <= target_flops:
            break
    
    actual_reduction = 1.0 - (current_flops / original_total_flops)
    
    return layer_pruned_indices, actual_reduction


def flops_based_pruning_indices(
    apoz_values: Dict[str, np.ndarray],
    model: nn.Module,
    target_flops_reduction: float,
    min_channels: int,
    divisor: int = 8,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32)
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    【主入口函数】基于 FLOPS 约束的剪枝索引生成
    
    Args:
        apoz_values: 各层的 ApoZ 值字典
        model: 神经网络模型
        target_flops_reduction: 目标 FLOPs 减少百分比 (0~1)
        min_channels: 全局最小通道数保护阈值
        divisor: 对齐因子 (默认 8)
        input_shape: 输入形状
    
    Returns:
        Tuple[剪枝掩码字典, 剪枝统计信息]
    
    Example:
        >>> apoz = compute_apoz(model, dataloader)
        >>> masks, stats = flops_based_pruning_indices(
        ...     apoz, model, 0.5, min_channels=8, divisor=8
        ... )
    """
    # Step 1: 收集所有 Filter 信息
    all_filters = collect_all_filters(apoz_values, model, input_shape)
    
    # Step 2: 获取模型总 FLOPs 和层通道信息
    layer_flops = compute_layer_flops(model, input_shape)
    layer_channel_map = get_layer_input_output_channels(model)
    original_total_flops = sum(layer_flops.values())
    
    # Step 3: 全局排序
    sorted_filters = global_filter_ranking(all_filters)
    
    # Step 4: FLOPS 约束搜索
    pruning_indices, actual_reduction = flops_constraint_search(
        sorted_filters,
        target_flops_reduction,
        original_total_flops,
        layer_channel_map,
        min_channels,
        divisor
    )
    
    # Step 5: 生成剪枝掩码
    pruning_masks = {}
    pruning_stats = {
        'original_flops': original_total_flops,
        'target_reduction': target_flops_reduction,
        'actual_reduction': actual_reduction,
        'layer_details': {}
    }
    
    for layer_name, apoz in apoz_values.items():
        num_channels = len(apoz)
        original_channels = layer_channel_map.get(layer_name, (64, 64))[1]
        
        # 获取要剪掉的索引
        prune_indices = set(pruning_indices.get(layer_name, []))
        
        # 生成掩码 (True = 保留, False = 剪掉)
        mask = np.array([i not in prune_indices for i in range(num_channels)])
        
        # Step 6: 【最小通道数保护】重新检查并修正
        kept_channels = mask.sum()
        
        if kept_channels < min_channels:
            # 需要补充保留一些重要通道
            # 选择 ApoZ 最低（最重要）的通道保留
            if kept_channels > 0:
                # 找出当前保留的通道中 ApoZ 最高的（即最不重要的）
                kept_indices = np.where(mask)[0]
                kept_apoz = [(i, apoz[i]) for i in kept_indices]
                kept_apoz.sort(key=lambda x: x[1], reverse=True)  # 按 ApoZ 降序
                
                # 移除最不重要的通道，直到满足最小通道数
                channels_to_remove = kept_channels - min_channels
                for i in range(min(channels_to_remove, len(kept_apoz))):
                    mask[kept_apoz[i][0]] = False
            else:
                # 如果全部被剪掉，保留 ApoZ 最低的 min_channels 个
                indices = np.argsort(apoz)[:min_channels]
                mask = np.zeros_like(mask, dtype=bool)
                mask[indices] = True
        
        # Step 7: 通道对齐检查
        if kept_channels % divisor != 0:
            # 对齐到 divisor 的倍数
            target_keep = int(kept_channels / divisor) * divisor
            target_keep = max(target_keep, min_channels)
            
            if target_keep < kept_channels:
                # 移除一些 ApoZ 最高的通道（最不重要的）
                current_kept = np.where(mask)[0]
                current_apoz = [(i, apoz[i]) for i in current_kept]
                current_apoz.sort(key=lambda x: x[1], reverse=True)
                
                channels_to_remove = kept_channels - target_keep
                for i in range(min(channels_to_remove, len(current_apoz))):
                    mask[current_apoz[i][0]] = False
        
        pruning_masks[layer_name] = mask
        
        # 记录统计信息
        pruning_stats['layer_details'][layer_name] = {
            'original': original_channels,
            'kept': mask.sum(),
            'pruned': num_channels - mask.sum(),
            'prune_ratio': 1.0 - (mask.sum() / num_channels)
        }
    
    return pruning_masks, pruning_stats


# ============================================================================
# 简化的 FLOPS 模式接口 (更易于使用)
# ============================================================================

def flops_pruning_get_indices(
    apoz_values: Dict[str, np.ndarray],
    model: nn.Module,
    target_flops_reduction: float,
    min_channels: int = 8,
    divisor: int = 8,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32)
) -> Dict[str, np.ndarray]:
    """
    【简化接口】基于 FLOPS 约束的剪枝掩码生成
    
    这是一个更简洁的入口函数，封装了完整的 FLOPS 约束剪枝逻辑。
    
    Args:
        apoz_values: 各层的 ApoZ 值字典 (来自 APoZ.compute_class_specific_apoz)
        model: 神经网络模型
        target_flops_reduction: 目标 FLOPs 减少百分比 (0.0 ~ 1.0)
            - 例如 0.5 表示减少 50% FLOPs
            - 例如 0.9 表示减少 90% FLOPs
        min_channels: 全局最小通道数保护阈值 (默认 8)
        divisor: 对齐因子，用于通道对齐 (默认 8)
        input_shape: 模型输入形状 (默认 CIFAR-10: 1x3x32x32)
    
    Returns:
        剪枝掩码字典 (Dict[层名称, np.ndarray])，True 表示保留该通道
    
    Example:
        >>> # 完整使用流程
        >>> from src.pruning.apoz import APoZ
        >>> from src.pruning.flops_pruning import flops_pruning_get_indices
        >>> 
        >>> # 1. 计算类特定 ApoZ
        >>> apoz_tool = APoZ(model)
        >>> apoz_values = apoz_tool.compute_class_specific_apoz(
        ...     dataloader, target_classes=[0, 1]
        ... )
        >>> 
        >>> # 2. FLOPS 约束剪枝 (减少 50% FLOPs，最少保留 8 通道)
        >>> masks = flops_pruning_get_indices(
        ...     apoz_values, model,
        ...     target_flops_reduction=0.5,
        ...     min_channels=8,
        ...     divisor=8
        ... )
    
    Algorithm:
        1. 收集所有 Filter 信息 (ApoZ, FLOPs, 通道数)
        2. 计算加权得分: Score = ApoZ × ΔFLOPs
        3. 全局排序 Filter (按得分降序)
        4. 遍历 Filter，累加 FLOPs 节省直到达到目标
        5. 对每层应用最小通道数保护
        6. 通道对齐到 divisor 的倍数
    """
    masks, stats = flops_based_pruning_indices(
        apoz_values=apoz_values,
        model=model,
        target_flops_reduction=target_flops_reduction,
        min_channels=min_channels,
        divisor=divisor,
        input_shape=input_shape
    )
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"FLOPS 约束剪枝统计")
    print(f"{'='*60}")
    print(f"目标 FLOPs 减少: {target_flops_reduction*100:.1f}%")
    print(f"实际 FLOPs 减少: {stats['actual_reduction']*100:.1f}%")
    print(f"最小通道数保护: {min_channels}")
    print(f"对齐因子: {divisor}")
    print(f"\n各层详情:")
    print("-"*60)
    for layer, detail in stats['layer_details'].items():
        print(f"{layer:20s}: {detail['original']:3d} -> {detail['kept']:3d} (剪枝 {detail['pruned']:3d}, {detail['prune_ratio']*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return masks


# ============================================================================
# 演示代码 (可用于测试)
# ============================================================================

if __name__ == "__main__":
    # 演示 safe_prune_channels 函数的使用
    print("演示最小通道数保护机制:")
    print("-" * 50)
    
    test_cases = [
        (64, 0.5, 8, 8),   # 正常情况: 64 -> 32
        (64, 0.9, 8, 8),   # 受保护: 只能到 8 通道
        (32, 0.7, 8, 8),   # 受保护: 32*0.3=9.6 -> 对齐到 8
        (16, 0.5, 8, 8),   # 边界: 16 -> 8 (正好最小值)
        (10, 0.2, 8, 8),   # 边界: 原始 < 最小值，保留全部
    ]
    
    for original, ratio, min_ch, div in test_cases:
        keep, actual = safe_prune_channels(original, ratio, min_ch, div)
        print(f"原始={original}, 目标剪枝={ratio*100:.0f}%, 最小={min_ch}, 对齐={div}")
        print(f"  → 保留 {keep} 通道, 实际剪枝 {actual*100:.1f}%\n")
