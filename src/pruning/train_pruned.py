import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
# TensorBoard support removed
from datetime import datetime
from torch.amp import autocast  # 引入混合精度训练所需的库
import csv

from ..models.resnet import resnet18
from .apoz import APoZ, apply_pruning_masks, print_model_size
from .flops_pruning import flops_pruning_get_indices, compute_model_flops, safe_prune_channels

class EarlyStopping:
    """早停机制，支持基于损失或准确率监控"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True, mode='loss'):
        """
        Args:
            patience (int): 容忍多少个epoch没有改善
            min_delta (float): 最小的改善阈值
            restore_best_weights (bool): 是否在早停时恢复最佳权重
            mode (str): 'loss' (损失下降) 或 'acc' (准确率上升)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode  # 'loss' 或 'acc'
        self.counter = 0
        self.best_metric = None
        self.best_weights = None
        self.early_stop = False

    def __call__(self, metric_value, model):
        if self.best_metric is None:
            self.best_metric = metric_value
            self.save_checkpoint(model)
        elif self.mode == 'loss' and metric_value < self.best_metric - self.min_delta:
            # 损失模式：损失越小越好
            self.best_metric = metric_value
            self.counter = 0
            self.save_checkpoint(model)
        elif self.mode == 'acc' and metric_value > self.best_metric + self.min_delta:
            # 准确率模式：准确率越大越好
            self.best_metric = metric_value
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    metric_name = "验证损失" if self.mode == 'loss' else "目标类别准确率"
                    print(f'早停触发，恢复最佳模型权重 ({metric_name}: {self.best_metric:.4f})')

    def save_checkpoint(self, model):
        """保存最佳模型权重"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

def parse_args():
    parser = argparse.ArgumentParser(description='Prune and Finetune ResNet18 (class-specific)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, required=True, help='batch size for training')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    
    # ====== 剪枝模式选择 ======
    # 模式1: 传统 ApoZ 阈值模式
    parser.add_argument('--threshold', type=float, default=0.8, 
                        help='pruning threshold for APoZ (traditional mode)')
    
    # 模式2: FLOPS 约束模式 (新增)
    parser.add_argument('--flops-reduction', type=float, default=None,
                        help='target FLOPs reduction ratio (0~1), e.g., 0.5 means reduce 50%% FLOPs')
    
    # ====== 通用剪枝参数 ======
    parser.add_argument('--min-channels', type=int, default=8, 
                        help='minimum number of channels to keep per layer (default: 8)')
    parser.add_argument('--divisor', type=int, default=8,
                        help='channel alignment divisor (default: 8)')
    
    parser.add_argument('--data-dir', type=str, default='data', help='data directory')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='path to latest checkpoint')
    parser.add_argument('--classes', type=int, nargs='+', default=None, required=True,
                        help='specific classes to focus on for pruning (e.g., --classes 0 1 3)')
    parser.add_argument('--log-dir', type=str, default='logs', help='directory to save tensorboard logs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs)')
    
    return parser.parse_args()

def train(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        # 使用混合精度训练
        with autocast(device.type):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        # 使用scaler来放大损失并进行反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            with autocast(device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / len(val_loader), 100. * correct / total

def validate_target_classes(model, val_loader, device, target_classes):
    """
    计算目标类别的准确率
    评估重映射标签（0=负样本，1,2,3...=目标类）上的准确率
    模型输出 10 类（CIFAR-10 原始类别），预测值范围 [0, 9]

    Args:
        model: 剪枝后的模型（输出 10 类）
        val_loader: 验证数据加载器（标签 0=负样本, 1,2,3...=目标类）
        device: 计算设备
        target_classes: 目标类别列表，如 [2, 3] 表示原始 CIFAR-10 的第 2、3 类
    """
    model.eval()
    num_target_classes = len(target_classes)

    # 创建映射：重映射标签 -> 原始 CIFAR-10 类别
    # 重映射标签 1, 2, 3... 对应 target_classes[0], target_classes[1], ...
    remap_to_original = {i + 1: target_classes[i] for i in range(num_target_classes)}

    class_correct = [0] * num_target_classes
    class_total = [0] * num_target_classes

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # 输出维度 = 10
            _, predicted = outputs.max(1)  # 预测值范围 [0, 9]

            for i in range(len(labels)):
                label = labels[i].item()

                # 只评估正样本（目标类别）
                if label == 0:
                    continue

                # 将重映射标签转换为原始 CIFAR-10 类别
                original_class = remap_to_original.get(label)
                if original_class is None:
                    continue

                # 检查预测是否正确
                if predicted[i].item() == original_class:
                    class_correct[label - 1] += 1
                class_total[label - 1] += 1

    # 打印各类别准确率
    target_accs = []
    for i in range(num_target_classes):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            original_cls = target_classes[i]
            print(f"  Class {original_cls}: {acc:.2f}%")
            target_accs.append(acc)
        else:
            print(f"  Class {target_classes[i]}: 0.00% (no samples)")
            target_accs.append(0.0)

    # 返回目标类别的平均准确率
    return sum(target_accs) / len(target_accs) if target_accs else 0

def validate_class(model, val_loader, target_class, device):
    """
    评估模型在特定类别上的性能
    val_loader 中的标签是重映射后的 (0=负样本, 1,2,3...=正类别)
    模型输出是 10 类，预测值是原始类别索引 (0-9)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # 找出重映射标签等于 target_class 的样本
            mask = (targets == target_class)
            if not mask.any():
                continue

            # 只评估目标类别
            class_inputs = inputs[mask]
            class_targets = targets[mask]

            with autocast(device.type):
                outputs = model(class_inputs)

            _, predicted = outputs.max(1)
            # 预测值是原始类别索引 (0-9)，需要比较是否等于 target_class 对应的原始类别
            total += class_targets.size(0)
            correct += (predicted == target_class).sum().item()

    return 100. * correct / total if total > 0 else 0


def get_class_weighted_dataloader(dataset, target_classes, batch_size, shuffle=True):
    """创建一个对目标类别和非目标类别使用不同权重的数据加载器"""
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

    print(f"目标类样本数量: {len(target_indices)}")
    print(f"非目标类样本数量: {len(non_target_indices)}")

    # 计算目标类和非目标类的比例
    total_samples = len(dataset)
    target_ratio = len(target_indices) / total_samples
    non_target_ratio = len(non_target_indices) / total_samples

    # 计算权重，使得:
    # 1. 目标类:非目标类 = 9:1
    # 2. target_ratio * target_weight + non_target_ratio * non_target_weight = 1
    target_weight = 9 / (9 * target_ratio + non_target_ratio)
    non_target_weight = 1 / (9 * target_ratio + non_target_ratio)

    print(f"目标类权重: {target_weight:.4f}")
    print(f"非目标类权重: {non_target_weight:.4f}")

    # 设置权重
    weights = torch.ones(len(dataset))

    # 目标类权重
    for idx in target_indices:
        weights[idx] = target_weight

    # 非目标类权重
    for idx in non_target_indices:
        weights[idx] = non_target_weight

    # 创建加权采样器
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True
    )

    # 返回使用加权采样器的数据加载器
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )


def validate_all_classes(model, test_dataset, device, num_classes=10, batch_size=128):
    """
    评估模型在数据集所有类别上的性能，返回各类别准确率

    Args:
        model: 要评估的模型
        test_dataset: 完整测试数据集
        device: 计算设备
        num_classes: 类别数
        batch_size: 批大小

    Returns:
        dict: 各类别的准确率 {class_idx: accuracy}
    """
    model.eval()

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 统计每个类别的正确预测数和总数
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast(device.type):
                outputs = model(inputs)

            _, predicted = outputs.max(1)

            # 统计每个类别的正确数
            for c in range(num_classes):
                mask = (targets == c)
                class_total[c] += mask.sum().item()
                class_correct[c] += ((predicted == c) & mask).sum().item()

    # 计算各类别准确率
    class_acc = {}
    for c in range(num_classes):
        if class_total[c] > 0:
            class_acc[c] = 100. * class_correct[c] / class_total[c]
        else:
            class_acc[c] = 0.0

    return class_acc


def expand_fc_layer(model, original_classes, target_classes=10, device=None, class_mapping=None):
    """
    将模型的 FC 层从 original_classes 扩展到 target_classes

    Args:
        model: 要修改的模型
        original_classes: 原始输出类别数
        target_classes: 目标输出类别数（默认 10）
        device: 计算设备
        class_mapping: 类别映射列表，如 [0, 3, 5] 表示输出1->类别0, 输出2->类别3, 输出3->类别5

    Returns:
        修改后的模型
    """
    if original_classes == target_classes:
        return model

    # 获取设备
    if device is None:
        device = next(model.parameters()).device

    # 新的 FC 层
    in_features = model.fc.in_features
    new_fc = nn.Linear(in_features, target_classes).to(device)

    # 复制原始权重
    with torch.no_grad():
        # 初始化所有权重为小随机值
        nn.init.normal_(new_fc.weight, mean=0, std=0.01)
        nn.init.zeros_(new_fc.bias)

        # 复制原始类别对应的权重
        # 输出索引 0 是负样本（不复制到任何 CIFAR-10 类别）
        # 输出索引 1,2,3,... 分别对应 class_mapping 中的类别
        if class_mapping is not None:
            for new_idx, original_class_idx in enumerate(class_mapping, start=1):
                # new_fc 的第 original_class_idx 行对应旧 fc 的第 new_idx 行
                new_fc.weight[original_class_idx] = model.fc.weight[new_idx].clone().to(device)
                new_fc.bias[original_class_idx] = model.fc.bias[new_idx].clone().to(device)

    model.fc = new_fc
    print(f"FC 层已从 {original_classes} 类扩展到 {target_classes} 类 (类别映射: {class_mapping})")

    return model


def save_acc_to_csv(class_acc, args, result_path='result/prune_acc.csv'):
    """
    将剪枝后的模型各类别准确率保存到CSV文件（追加模式）

    Args:
        class_acc: 各类别准确率字典 {class_idx: accuracy}
        args: 命令行参数
        result_path: 结果保存路径
    """
    # 确保result目录存在
    result_dir = os.path.dirname(result_path)
    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir)

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
        class_acc_list = [class_acc.get(i, 0.0) for i in range(10)]
        mean_acc = sum(class_acc_list) / len(class_acc_list)

        row = [timestamp, classes_str, args.threshold, args.epochs, args.batch_size, args.lr] + \
              [f'{acc:.2f}' for acc in class_acc_list] + [f'{mean_acc:.2f}']
        writer.writerow(row)

    print(f"准确率结果已保存到: {result_path}")

def main():
    args = parse_args()
    
    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # TensorBoard disabled due to compatibility issues
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用！所有训练必须使用CUDA GPU。")
    device = torch.device("cuda")
    print(f"使用设备: {device}")
    
    # 配置混合精度训练
    scaler = torch.amp.GradScaler('cuda')
    print("使用混合精度训练 (AMP)")
    
    # 准备数据
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
        root=args.data_dir, train=False, download=True, transform=transform_test)
    
    # 获取基础模型
    model = resnet18(num_classes=10)
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    model = model.to(device)
    
    # 记录原始模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print_model_size(model, "Original model")
    
    # 计算原始模型 FLOPs
    original_flops = compute_model_flops(model)
    print(f"原始模型 FLOPs: {original_flops / 1e6:.2f} M")
    
    # 打印目标类别
    classes = args.classes
    print(f"Pruning model specifically for classes: {classes}")
    
    # 计算APoZ值并获取剪枝掩码
    apoz = APoZ(model)

    # 使用CUDA事件进行计时
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

    # 使用参考实现的 compute_class_specific_apoz API (使用 DataLoader)
    print(f"Using class-specific APoZ pruning method for classes {args.classes}")

    # 使用加权采样器进行 APoZ 计算
    prune_loader = get_class_weighted_dataloader(
        train_dataset, args.classes, args.batch_size)

    apoz_values = apoz.compute_class_specific_apoz(
        prune_loader,
        target_classes=args.classes,
        num_batches=100  # 控制计算 APoZ 的批次数
    )

    # 记录GPU计算时间
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        print(f"APoZ计算时间: {start_event.elapsed_time(end_event) / 1000:.2f} 秒")

    apoz.remove_hooks()

    # ====== 选择剪枝模式 ======
    if args.flops_reduction is not None:
        # ====== FLOPS 约束模式 (新增) ======
        print(f"\n{'='*60}")
        print(f"使用 FLOPS 约束剪枝模式")
        print(f"{'='*60}")
        print(f"目标 FLOPs 减少: {args.flops_reduction * 100:.1f}%")
        print(f"最小通道数: {args.min_channels}")
        print(f"对齐因子: {args.divisor}")
        
        # 使用新的 FLOPS 约束剪枝方法
        pruning_masks = flops_pruning_get_indices(
            apoz_values=apoz_values,
            model=model,
            target_flops_reduction=args.flops_reduction,
            min_channels=args.min_channels,
            divisor=args.divisor
        )
        
        # 计算剪枝后的 FLOPs
        pruned_flops = original_flops * (1 - args.flops_reduction)
        print(f"目标 FLOPs: {pruned_flops / 1e6:.2f} M")
        
    else:
        # ====== 传统阈值模式 (向后兼容) ======
        print(f"\n使用传统 ApoZ 阈值模式 (threshold={args.threshold})")
    pruning_masks = apoz.get_pruning_indices(apoz_values, threshold=args.threshold, min_channels=args.min_channels)

    # 应用剪枝掩码（保持 10 类输出，参考实现）
    pruned_model = apply_pruning_masks(model, pruning_masks, num_classes=10)
    pruned_model = pruned_model.to(device)

    # 记录剪枝后的模型参数数量
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    print_model_size(pruned_model, "Pruned model")

    # 微调阶段使用类别加权的数据加载器（参考实现）
    train_loader = get_class_weighted_dataloader(
        train_dataset, args.classes, args.batch_size)
    print(f"微调阶段使用类别 {args.classes} 的加权数据加载器")

    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 设置优化器和损失函数（参考实现）
    # 为目标类别设置更高权重的损失函数
    class_weights = torch.ones(10).to(device)
    for cls in args.classes:
        class_weights[cls] = 5.0  # 设置更高的权重
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"使用加权损失函数，目标类别权重: 5.0")

    # 使用 SGD + Momentum（参考实现）
    optimizer = optim.SGD(pruned_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 初始化早停机制 (使用准确率模式)
    early_stopper = EarlyStopping(patience=args.patience, mode='acc')

    # 微调模型
    max_test_acc = 0.0
    train_time_record = []

    for epoch in range(args.epochs):
        start_time = time.time()

        # 训练
        train_loss, train_acc = train(pruned_model, train_loader, criterion, optimizer, device, scaler)
        train_time = time.time() - start_time
        train_time_record.append(train_time)

        # 验证
        test_start = time.time()
        val_loss, val_acc = validate(pruned_model, val_loader, criterion, device)
        test_time = time.time() - test_start

        # 计算目标类别准确率（计算所有目标类别的平均准确率）
        print("\n特定类别评估结果：")
        target_class_accs = []
        for cls in args.classes:
            cls_acc = validate_class(pruned_model, val_loader, cls, device)
            print(f"  类别 {cls} 准确率: {cls_acc:.2f}%")
            target_class_accs.append(cls_acc)
        # 使用所有目标类别的平均准确率作为保存依据
        target_class_acc = np.mean(target_class_accs)
        print(f"  目标类别平均准确率: {target_class_acc:.2f}%")

        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f'Epoch[{epoch + 1}/{args.epochs}] train_loss: {train_loss:.4f}, train_acc={train_acc:.4f}, '
              f'test_loss={val_loss:.4f}, test_acc={val_acc:.4f}, max_test_acc={max_test_acc:.4f}')
        print(f'train time: {train_time:.3f}s, test time: {test_time:.3f}s')

        if target_class_acc > max_test_acc:
            max_test_acc = target_class_acc

            # 构建保存文件名
            class_info = '_class' + '-'.join(map(str, args.classes))
            checkpoint = {
                'state_dict': pruned_model.state_dict(),
                'max_test_acc': max_test_acc
            }
            save_path = os.path.join(args.save_dir, f'pruned{class_info}_th{args.threshold}.pth')
            torch.save(checkpoint, save_path)
            print(f'Model saved to {save_path}')

        # 早停检查（使用目标类别准确率）
        early_stopper(target_class_acc, pruned_model)
        if early_stopper.early_stop:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    print(f'Finish retrain the pruned model')

    # 使用完整测试集评估所有类别的准确率并保存到CSV
    class_acc = validate_all_classes(pruned_model, test_dataset, device, num_classes=10, batch_size=args.batch_size)

    print("各类别准确率:")
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        print(f"  类别 {i} ({cifar10_classes[i]}): {class_acc[i]:.2f}%")
    mean_acc = sum(class_acc.values()) / 10
    print(f"  平均准确率: {mean_acc:.2f}%")

    # 保存到CSV文件
    save_acc_to_csv(class_acc, args)


if __name__ == '__main__':
    main()
