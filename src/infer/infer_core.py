#!/usr/bin/env python3
"""
推理测试核心模块
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
import argparse
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import torchvision
import torchvision.transforms as transforms
import numpy as np


def get_cifar10_loaders(batch_size=128, data_dir='data'):
    """获取 CIFAR-10 测试数据加载器"""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, transform=transform_test, download=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    return test_loader


def infer_single_mode(models, model_classes, test_loader, device):
    """Single 模式：单个剪枝模型推理"""
    print("\n" + "="*60)
    print("Single 模式：测试单个剪枝模型")
    print("="*60)

    results = []

    for i, model in enumerate(models):
        model = model.to(device)
        model.eval()

        target_class = model_classes[i]
        last_layer = model.layer4[-1].conv2
        out_channels = last_layer.out_channels

        print(f"\n模型 {i}: 目标类别 {target_class}")
        print(f"  输出通道数: {out_channels}")

        correct = 0
        total = 0
        infer_times = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                start_time = time.time()
                outputs = model(inputs)
                infer_times.append(time.time() - start_time)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        avg_infer_time = np.mean(infer_times)

        print(f"  准确率: {acc:.2f}%")
        print(f"  infer time: {avg_infer_time:.4f}s")

        results.append({
            'model_idx': i,
            'target_class': target_class,
            'accuracy': acc,
            'infer_time': avg_infer_time,
            'channels': out_channels
        })

    return results


def infer_split_mode(models, model_classes, test_loader, device, fusion_path):
    """Split 模式：多个剪枝模型 + Fusion 推理"""
    print("\n" + "="*60)
    print("Split 模式：测试多个剪枝模型 + Fusion")
    print("="*60)

    # 先统计当前加载的模型数量和特征维度
    current_dims = []
    for model in models:
        out_channels = model.layer4[-1].conv2.out_channels
        current_dims.append(out_channels)
    current_total = sum(current_dims)
    current_model_num = len(models)

    print(f"\n当前加载的剪枝模型数量: {current_model_num}")
    print(f"当前特征维度: {current_dims}")
    print(f"当前总输入维度: {current_total}")

    # 延迟导入 FusionModel，避免循环依赖
    from ..fusion.train_fusion import FusionModel

    # 如果 fusion_path 不存在或需要自动选择
    if not os.path.exists(fusion_path):
        print(f"\n指定的 Fusion 模型不存在: {fusion_path}")
        print("尝试在 checkpoints 目录中查找匹配的检查点...")

        # 查找 checkpoints 目录下所有 fusion_model.pth 文件
        checkpoints_dir = 'checkpoints'
        if os.path.exists(checkpoints_dir):
            import glob
            checkpoint_files = glob.glob(os.path.join(checkpoints_dir, '**/*.pth'), recursive=True)
            checkpoint_files = [f for f in checkpoint_files if 'fusion' in f.lower()]

            matched_path = None
            for cp in checkpoint_files:
                try:
                    cp_checkpoint = torch.load(cp, map_location='cpu', weights_only=False)
                    cp_feature_dims = cp_checkpoint.get('feature_dims', None)
                    if cp_feature_dims and len(cp_feature_dims) == current_model_num:
                        matched_path = cp
                        print(f"找到匹配的检查点: {cp}")
                        print(f"  特征维度: {cp_feature_dims}")
                        print(f"  模型数量: {len(cp_feature_dims)}")
                        break
                except Exception as e:
                    continue

            if matched_path:
                fusion_path = matched_path
            else:
                print(f"\n错误: 在 checkpoints 目录中找不到与 {current_model_num} 个模型匹配的 Fusion 检查点")
                print("请先运行 fuse.py 训练 Fusion 模型")
                return None

    print(f"\n加载 Fusion 模型: {fusion_path}")
    checkpoint = torch.load(fusion_path, map_location=device, weights_only=False)

    # 向后兼容：支持新旧两种检查点格式
    if 'model_state_dict' in checkpoint:
        fusion_model_state = checkpoint['model_state_dict']
        saved_feature_dims = checkpoint.get('feature_dims', None)
    else:
        if 'net' in checkpoint:
            fusion_model_state = checkpoint['net'].state_dict() if hasattr(checkpoint['net'], 'state_dict') else checkpoint['net']
        else:
            fusion_model_state = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        saved_feature_dims = None

    saved_model_num = len(saved_feature_dims) if saved_feature_dims else 0
    saved_total = sum(saved_feature_dims) if saved_feature_dims else 1368

    # 检查模型数量是否匹配
    if saved_feature_dims and len(saved_feature_dims) != current_model_num:
        print(f"\n错误: Fusion 检查点模型数量不匹配!")
        print(f"  检查点中的模型数量: {len(saved_feature_dims)}")
        print(f"  当前加载的模型数量: {current_model_num}")
        print(f"  检查点特征维度: {saved_feature_dims}")
        print(f"\n请使用 --fusion-path 指定正确的检查点文件")
        return None

    # 从检查点中推断 hidden_dim（从 classifier.0.weight 的形状）
    if 'classifier.0.weight' in fusion_model_state:
        saved_hidden_dim = fusion_model_state['classifier.0.weight'].shape[0]
    else:
        saved_hidden_dim = 384  # 默认值

    print(f"Checkpoint hidden_dim: {saved_hidden_dim}")

    print(f"创建 Fusion 模型...")
    fusion_model = FusionModel(saved_total, pmodel_num=current_model_num, num_classes=10, hidden_dim=saved_hidden_dim)
    print(f"Checkpoint 特征维度: {saved_feature_dims}")

    print("加载 Fusion 模型权重...")
    fusion_model.load_state_dict(fusion_model_state)

    # 由于已经检查过模型数量，这里只做次要警告
    if current_total != saved_total:
        print(f"注意: 特征维度微调 - 当前: {current_total}, Checkpoint: {saved_total}")

    feature_extractors = []
    for model in models:
        fe = nn.Sequential(*list(model.children())[:-1])
        fe = fe.to(device)
        fe.eval()
        feature_extractors.append(fe)

    fusion_model = fusion_model.to(device)
    fusion_model.eval()

    print("\n开始推理测试...")

    # 单样本推理 + 模拟并行（取max）
    correct = 0
    total = 0
    infer_times = []  # 总推理时间
    pmodel_infer_times = []  # 每个 pmodel 的推理时间

    # 获取单个样本进行推理测试
    sample_dataset = test_loader.dataset
    num_samples = len(sample_dataset)

    with torch.no_grad():
        for sample_idx in range(num_samples):
            # 获取单个样本
            inputs, labels = sample_dataset[sample_idx]
            inputs = inputs.unsqueeze(0).to(device)  # (1, C, H, W)
            label = torch.tensor([labels]).to(device)

            pmodel_times = []
            features = []

            # 测量每个 pmodel 的推理时间
            for fe in feature_extractors:
                fe_start = time.time()
                feat = fe(inputs)
                feat = torch.flatten(feat, 1)
                pmodel_time = time.time() - fe_start
                pmodel_times.append(pmodel_time)
                features.append(feat)

            # 模拟并行推理：取最慢模型的推理时间
            parallel_time = max(pmodel_times)

            # Fusion 推理
            fusion_start = time.time()
            fused_features = torch.cat(features, dim=1)
            outputs = fusion_model(fused_features)
            fusion_time = time.time() - fusion_start

            # 总时间 = 并行时间 + Fusion 时间
            total_time = parallel_time + fusion_time
            infer_times.append(total_time)
            pmodel_infer_times.append(pmodel_times)

            # 计算准确率
            _, predicted = outputs.max(1)
            correct += predicted.eq(label).sum().item()
            total += 1

            if (sample_idx + 1) % 1000 == 0:
                print(f"  Sample {sample_idx + 1}/{num_samples}")

    acc = 100. * correct / total
    avg_infer_time = np.mean(infer_times)
    pmodel_infer_times_array = np.array(pmodel_infer_times)
    avg_pmodel_time = pmodel_infer_times_array.mean(axis=0)

    print(f"\nFusion 模型准确率: {acc:.2f}%")
    print(f"平均推理时间: {avg_infer_time:.4f}s")
    print(f"每样本平均推理时间: {avg_infer_time:.6f}s")
    print(f"average of infer one frame of a sample per pruned model: {np.mean(avg_pmodel_time):.4f}s, "
          f"max: {np.max(avg_pmodel_time):.4f}s, min: {np.min(avg_pmodel_time):.4f}s, "
          f"all: {[f'{v:.4f}s' for v in avg_pmodel_time]}")

    return {
        'accuracy': acc,
        'total_infer_time': avg_infer_time,
        'time_per_sample': avg_infer_time,  # avg_infer_time 已经是单样本平均时间
        'pmodel_times': avg_pmodel_time.tolist(),
        'feature_dims': current_dims,
        'pmodel_num': current_model_num
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Inference Test')
    parser.add_argument('--threshold', type=float, required=True,
                        help='pruning threshold used to locate model files')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    # 推理必须在 CPU 上进行，因为要部署到嵌入式设备
    parser.add_argument('--device', type=str, default='cpu', help='device (cpu only for inference)')
    parser.add_argument('--split', action='store_true',
                        help='use split mode (multiple models + Fusion)')
    parser.add_argument('--fusion-path', type=str, default='checkpoints/fusion_model.pth',
                        help='path to fusion model checkpoint')
    return parser.parse_args()


def main():
    from ..fusion.train_fusion import set_seed, load_pruned_models

    args = parse_args()
    set_seed(42)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    print("加载剪枝模型...")
    models, model_classes = load_pruned_models(args.threshold, device)
    print(f"模型数量: {len(models)}")

    print("\n加载 CIFAR-10 测试数据...")
    test_loader = get_cifar10_loaders(args.batch_size)
    print(f"测试样本数: {len(test_loader.dataset)}")

    if args.split:
        results = infer_split_mode(models, model_classes, test_loader, device, args.fusion_path)
    else:
        results = infer_single_mode(models, model_classes, test_loader, device)

    # 保存结果到 CSV
    if results:
        save_results_to_csv(results, args)

    print("\n" + "="*60)
    print("推理测试完成")
    print("="*60)


def save_results_to_csv(results, args, result_path='result/infer_inf.csv'):
    """保存推理结果到 CSV 文件"""
    import os
    result_dir = os.path.dirname(result_path)
    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir)

    file_exists = os.path.isfile(result_path)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(result_path, 'a', newline='') as f:
        writer = csv.writer(f)

        if args.split:
            # Split 模式：Fusion 模型推理时间
            if not file_exists:
                header = ['timestamp', 'threshold', 'batch_size', 'device', 'split',
                          'total_infer_time', 'time_per_sample']
                if 'pmodel_times' in results:
                    header += [f'pmodel_{i}_time' for i in range(len(results['pmodel_times']))]
                writer.writerow(header)

            # 获取 pmodel 时间
            pmodel_times = results.get('pmodel_times', [])
            pmodel_time_cols = [f"{t:.6f}" for t in pmodel_times]

            # time_per_sample 已经是单样本平均时间（total_infer_time = np.mean(infer_times)）
            time_per_sample = results.get('time_per_sample', results['total_infer_time'])
            row = [timestamp, args.threshold, args.batch_size, args.device, 'True',
                   f"{results['total_infer_time']:.6f}",
                   f"{time_per_sample:.6f}"] + pmodel_time_cols
        else:
            # Single 模式：各剪枝模型推理时间
            if not file_exists:
                header = ['timestamp', 'threshold', 'batch_size', 'device', 'split'] + \
                         [f'model_{i}_infer' for i in range(len(results))] + \
                         ['avg_time_per_sample']
                writer.writerow(header)

            time_list = [f"{r['infer_time']:.4f}" for r in results]
            # infer_time 已经是 batch 平均时间，直接作为单 batch 推理时间
            avg_time_per_sample = sum(r['infer_time'] for r in results) / len(results)
            row = [timestamp, args.threshold, args.batch_size, args.device, 'False'] + time_list + \
                  [f"{avg_time_per_sample:.6f}"]

        writer.writerow(row)

    print(f"推理结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
