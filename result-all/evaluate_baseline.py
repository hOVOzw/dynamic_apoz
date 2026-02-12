#!/usr/bin/env python3
"""
评估 ResNet18 基准模型精度

评估 src/training/checkpoints/best_model.pth 并生成基准精度 CSV
支持在 CIFAR-10 测试集的多个子集上评估
"""

import os
import sys
import argparse
import time
import torch
import pandas as pd
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.resnet import resnet18


def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate_baseline(model_path=None, batch_size=128, device=None):
    """
    评估 ResNet18 基准模型
    
    Args:
        model_path: 模型文件路径 (默认: checkpoints/best_model.pth)
        batch_size: 批次大小
        device: 计算设备
    
    Returns:
        dict: 包含各类别精度和平均精度的字典
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    project_root = get_project_root()
    if model_path is None:
        model_path = os.path.join(project_root, 'src', 'training', 'checkpoints', 'best_model.pth')
    
    print(f"使用设备: {device}")
    print(f"模型路径: {model_path}")
    
    # 数据转换
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载数据集
    data_dir = os.path.join(project_root, 'data')
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # 加载模型
    model = resnet18(num_classes=10).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['net'])
        print(f"加载模型准确率: {checkpoint.get('acc', 'N/A')}")
        print(f"训练轮次: {checkpoint.get('epoch', 'N/A')}")
    else:
        print(f"警告: 模型文件不存在: {model_path}")
        return None
    
    # 评估
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            c = (predicted == targets).squeeze()
            for i in range(len(targets)):
                label = targets[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # 计算各类别精度
    class_acc = []
    for i in range(10):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            class_acc.append(acc)
        else:
            class_acc.append(0.0)
    
    mean_acc = sum(class_acc) / len(class_acc)

    # ==================== 推理时间测试 ====================
    # 获取单个样本进行推理时间测试
    sample_dataset = test_dataset
    num_samples = len(sample_dataset)
    infer_times = []

    print(f"\n开始推理时间测试 ({num_samples} 个样本)...")

    with torch.no_grad():
        for sample_idx in range(num_samples):
            # 获取单个样本
            inputs, labels = sample_dataset[sample_idx]
            inputs = inputs.unsqueeze(0).to(device)  # (1, C, H, W)
            label = torch.tensor([labels]).to(device)

            # 测量推理时间
            start_time = time.time()
            outputs = model(inputs)
            infer_time = time.time() - start_time
            infer_times.append(infer_time)

            if (sample_idx + 1) % 1000 == 0:
                print(f"  Sample {sample_idx + 1}/{num_samples}")

    avg_infer_time = np.mean(infer_times)  # 秒
    avg_infer_time_ms = avg_infer_time * 1000  # 毫秒
    min_infer_time_ms = np.min(infer_times) * 1000
    max_infer_time_ms = np.max(infer_times) * 1000

    print(f"推理时间测试完成:")
    print(f"  平均推理时间: {avg_infer_time_ms:.2f} ms/样本")
    print(f"  最小推理时间: {min_infer_time_ms:.2f} ms/样本")
    print(f"  最大推理时间: {max_infer_time_ms:.2f} ms/样本")
    # =====================================================

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model': 'resnet18_baseline',
        'subset': 'full_test',
        'mean_acc': mean_acc,
        'infer_time_ms': avg_infer_time_ms,
        'class_0': class_acc[0],
        'class_1': class_acc[1],
        'class_2': class_acc[2],
        'class_3': class_acc[3],
        'class_4': class_acc[4],
        'class_5': class_acc[5],
        'class_6': class_acc[6],
        'class_7': class_acc[7],
        'class_8': class_acc[8],
        'class_9': class_acc[9],
    }
    
    return results


def evaluate_on_subsets(model_path=None, batch_size=128, num_subsets=42, device=None):
    """
    在 CIFAR-10 测试集的多个子集上评估模型
    
    子集划分方式：
    - 按类别数量划分：1个样本、2个样本...逐步增加
    - 每个子集包含所有类别的样本
    
    Args:
        model_path: 模型文件路径
        batch_size: 批次大小
        num_subsets: 子集数量 (默认42)
        device: 计算设备
    
    Returns:
        list: 包含每个子集评估结果的列表
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    project_root = get_project_root()
    if model_path is None:
        model_path = os.path.join(project_root, 'src', 'training', 'checkpoints', 'best_model.pth')
    
    print(f"使用设备: {device}")
    print(f"模型路径: {model_path}")
    print(f"子集数量: {num_subsets}")
    
    # 数据转换
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载完整测试集
    data_dir = os.path.join(project_root, 'data')
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=transform_test
    )
    
    # 加载模型
    model = resnet18(num_classes=10).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    model.eval()
    
    # 按样本数量划分42个子集
    # 每个子集包含固定数量的样本
    results_list = []
    total_samples = len(test_dataset)
    
    # 计算每个子集的样本数量
    subset_sizes = [max(100, int(total_samples * i / num_subsets)) for i in range(1, num_subsets + 1)]
    
    CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("\n开始在子集上评估...")
    
    for subset_id, subset_size in enumerate(subset_sizes, 1):
        # 随机选择 subset_size 个样本
        indices = torch.randperm(total_samples)[:subset_size].tolist()
        subset_dataset = torch.utils.data.Subset(test_dataset, indices)
        subset_loader = torch.utils.data.DataLoader(
            subset_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # 评估
        class_correct = [0] * 10
        class_total = [0] * 10
        
        with torch.no_grad():
            for inputs, targets in subset_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                c = (predicted == targets).squeeze()
                for i in range(len(targets)):
                    label = targets[i].item()
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        # 计算各类别精度
        class_acc = []
        for i in range(10):
            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
                class_acc.append(acc)
            else:
                class_acc.append(0.0)
        
        mean_acc = sum(class_acc) / len(class_acc)

        # ==================== 推理时间测试 ====================
        # 获取子集中的样本进行推理时间测试
        subset_infer_times = []
        subset_loader_for_time = torch.utils.data.DataLoader(
            subset_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        print(f"\n  子集 {subset_id}: 推理时间测试 ({subset_size} 个样本)...")

        with torch.no_grad():
            for sample_idx, (inputs, labels) in enumerate(subset_loader_for_time):
                inputs = inputs.to(device)
                label = torch.tensor([labels[0]]).to(device) if isinstance(labels, torch.Tensor) else torch.tensor([labels]).to(device)

                # 测量推理时间
                start_time = time.time()
                outputs = model(inputs)
                infer_time = time.time() - start_time
                subset_infer_times.append(infer_time)

        avg_subset_infer_time_ms = np.mean(subset_infer_times) * 1000
        # =====================================================

        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': 'resnet18_baseline',
            'subset_id': subset_id,
            'subset_size': subset_size,
            'mean_acc': mean_acc,
            'infer_time_ms': avg_subset_infer_time_ms,
            'class_0': class_acc[0],
            'class_1': class_acc[1],
            'class_2': class_acc[2],
            'class_3': class_acc[3],
            'class_4': class_acc[4],
            'class_5': class_acc[5],
            'class_6': class_acc[6],
            'class_7': class_acc[7],
            'class_8': class_acc[8],
            'class_9': class_acc[9],
        }
        results_list.append(results)
        
        if subset_id % 10 == 0 or subset_id == 1:
            print(f"  子集 {subset_id:2d}/{num_subsets}: size={subset_size:5d}, mean_acc={mean_acc:.2f}%")
    
    return results_list


def save_baseline_csv(results, output_path):
    """保存基准精度到 CSV 文件"""
    # 处理只有文件名没有目录的情况
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(results, list):
        df = pd.DataFrame(results)
    else:
        df = pd.DataFrame([results])
    df.to_csv(output_path, index=False)
    
    print(f"\n基准精度已保存到: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='评估 ResNet18 基准模型')
    parser.add_argument('--model', type=str, default=None,
                        help='模型文件路径')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='批次大小')
    parser.add_argument('--output', type=str, 
                        default='baseline_resnet18_acc.csv',
                        help='输出CSV文件路径')
    parser.add_argument('--subsets', type=int, default=0,
                        help='在多少个子集上评估 (0=不评估子集, 默认42)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("评估 ResNet18 基准模型")
    print("=" * 60)
    
    if args.subsets > 0:
        # 在子集上评估
        results_list = evaluate_on_subsets(
            model_path=args.model, 
            batch_size=args.batch_size,
            num_subsets=args.subsets
        )
        
        if results_list:
            print("\n子集评估结果汇总:")
            print("-" * 40)
            mean_accs = [r['mean_acc'] for r in results_list]
            print(f"  平均精度范围: {min(mean_accs):.2f}% - {max(mean_accs):.2f}%")
            print(f"  总体平均: {sum(mean_accs)/len(mean_accs):.2f}%")
            save_baseline_csv(results_list, args.output)
    else:
        # 完整测试集评估
        results = evaluate_baseline(model_path=args.model, batch_size=args.batch_size)
        
        if results:
            print("\n评估结果:")
            print("-" * 40)
            cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                             'dog', 'frog', 'horse', 'ship', 'truck']
            for i, cls in enumerate(cifar_classes):
                print(f"  {cls:>10}: {results[f'class_{i}']:.2f}%")
            print("-" * 40)
            print(f"  {'Mean':>10}: {results['mean_acc']:.2f}%")
            print(f"  {'Infer Time':>10}: {results['infer_time_ms']:.2f} ms/样本")

            # 保存结果
            save_baseline_csv(results, args.output)
    
    print("=" * 60)


if __name__ == "__main__":
    main()
