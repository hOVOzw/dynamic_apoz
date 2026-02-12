#!/usr/bin/env python3
"""
Fusion 模型训练脚本
流程：
1. 加载多个剪枝模型（每个类别一个）
2. 从每个剪枝模型提取特征
3. 拼接特征，输入 Fusion 分类器
4. 训练 Fusion 模型

策略：FusionModel (ImprovedFusionClassifier 逻辑)
- 简单拼接 + BatchNorm + MLP 分类器
- 稳定、高效、推理友好
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import glob
import re
import argparse
from tqdm import tqdm
import random
import csv
from datetime import datetime
import numpy as np


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子已设置为: {seed}")


def get_pruned_model_structure(state_dict):
    """从状态字典中获取剪枝后的模型结构"""
    layer_channels = {}
    for key, value in state_dict.items():
        if 'conv' in key and 'weight' in key:
            layer_channels[key] = value.size(0)
            if 'conv1' in key:
                layer_channels[key + '_in'] = value.size(1)
    return layer_channels


class BasicBlock(nn.Module):
    def __init__(self, in_channels, conv1_channels, conv2_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(conv2_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class PrunedResNet(nn.Module):
    """剪枝后的 ResNet 模型"""
    def __init__(self, layer_channels, num_classes=10):
        super(PrunedResNet, self).__init__()
        conv1_out_channels = layer_channels.get('conv1.weight', 64)
        self.conv1 = nn.Conv2d(3, conv1_out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(in_channels=conv1_out_channels, channels=layer_channels, layer_name='layer1', stride=1)
        layer1_out_channels = layer_channels.get('layer1.1.conv2.weight', 64)
        self.layer2 = self._make_layer(in_channels=layer1_out_channels, channels=layer_channels, layer_name='layer2', stride=2)
        layer2_out_channels = layer_channels.get('layer2.1.conv2.weight', 128)
        self.layer3 = self._make_layer(in_channels=layer2_out_channels, channels=layer_channels, layer_name='layer3', stride=2)
        layer3_out_channels = layer_channels.get('layer3.1.conv2.weight', 256)
        self.layer4 = self._make_layer(in_channels=layer3_out_channels, channels=layer_channels, layer_name='layer4', stride=2)
        layer4_out_channels = layer_channels.get('layer4.1.conv2.weight', 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(layer4_out_channels, num_classes)

    def _make_layer(self, in_channels, channels, layer_name, stride):
        layers = []
        block1_conv1_out = channels.get(f'{layer_name}.0.conv1.weight', 64)
        block1_conv2_out = channels.get(f'{layer_name}.0.conv2.weight', 64)
        downsample1 = None
        if in_channels != block1_conv2_out or stride != 1:
            downsample1 = nn.Sequential(nn.Conv2d(in_channels, block1_conv2_out, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(block1_conv2_out))
        layers.append(BasicBlock(in_channels, block1_conv1_out, block1_conv2_out, stride, downsample1))
        block2_conv1_out = channels.get(f'{layer_name}.1.conv1.weight', 64)
        block2_conv2_out = channels.get(f'{layer_name}.1.conv2.weight', 64)
        downsample2 = None
        if block1_conv2_out != block2_conv2_out:
            downsample2 = nn.Sequential(nn.Conv2d(block1_conv2_out, block2_conv2_out, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(block2_conv2_out))
        layers.append(BasicBlock(block1_conv2_out, block2_conv1_out, block2_conv2_out, 1, downsample2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class FusionModel(nn.Module):
    """Fusion 分类器 - ImprovedFusionClassifier 逻辑"""
    def __init__(self, total_in_dim, pmodel_num=1, num_classes=10, hidden_dim=384):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 

    def forward(self, x):
        return self.classifier(x)


def load_pruned_models(threshold, device, models_dir='checkpoints', models_list=None):
    """加载剪枝模型"""
    model_files = []
    if models_list:
        model_files = models_list
    else:
        pattern1 = os.path.join(models_dir, f'pruned_class*_th{threshold}.pth')
        pattern2 = os.path.join(models_dir, f'pruned_class*_th{threshold:.1f}.pth')
        model_files = glob.glob(pattern1) or glob.glob(pattern2)
        if not model_files:
            pattern = os.path.join(models_dir, 'pruned_class*.pth')
            model_files = glob.glob(pattern)
        def _extract_first_idx(p):
            m = re.search(r'pruned_class(\d+)', os.path.basename(p))
            return int(m.group(1)) if m else 999
        model_files = sorted(model_files, key=_extract_first_idx)

    if not model_files:
        raise FileNotFoundError(f"No pruned model files found in '{models_dir}' for threshold {threshold}")

    models = []
    model_classes = []
    for path in model_files:
        basename = os.path.basename(path)
        m = re.search(r'pruned_class([\d\-]+)_', basename)
        if m:
            class_str = m.group(1)
            cls_indices = [int(x) for x in class_str.split('-')] if '-' in class_str else [int(class_str)]
        else:
            m = re.search(r'pruned_class(\d+)', basename)
            cls_indices = [int(m.group(1))] if m else [None]

        model = create_pruned_model_from_checkpoint(path, device)
        models.append(model)
        model_classes.append(cls_indices[0])

    return models, model_classes


def create_pruned_model_from_checkpoint(model_path, device):
    """从检查点文件创建剪枝后的模型"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    layer_channels = get_pruned_model_structure(state_dict)
    num_classes = state_dict.get('fc.weight', torch.zeros(10, 457)).shape[0]
    model = PrunedResNet(layer_channels, num_classes=num_classes).to(device)

    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            state_dict[key] = value.to(device)

    model.load_state_dict(state_dict)
    model = model.to(device)
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
        param.data = param.data.to(device)
    model.eval()
    return model


def create_fusion_model(models, weights, device):
    """
    创建融合模型 - 简化版，只使用 FusionModel
    """
    feature_dims = []
    feature_extractors = []
    for model in models:
        last_layer = model.layer4[-1].conv2
        out_channels = last_layer.out_channels
        feature_dims.append(out_channels)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        for param in feature_extractor.parameters():
            param.requires_grad = False
        feature_extractors.append(feature_extractor)

    total_in_dim = sum(feature_dims)
    num_models = len(models)

    fusion_model = FusionModel(total_in_dim, pmodel_num=num_models, num_classes=10, hidden_dim=min(384, total_in_dim // 2))
    print(f"\n使用 Fusion 模型: Input={total_in_dim}, Experts={num_models}")

    return feature_extractors, fusion_model, feature_dims


def extract_features(feature_extractors, x):
    """从所有特征提取器中提取特征"""
    features = []
    device = next(feature_extractors[0].parameters()).device
    x = x.to(device)
    for extractor in feature_extractors:
        extractor.eval()
        feat = extractor(x)
        feat = torch.flatten(feat, 1)
        # 归一化每个模型的特征，防止数值爆炸
        feat = torch.tanh(feat)
        features.append(feat)

    return torch.cat(features, dim=1)


def get_cifar10_loaders(batch_size=64, data_dir='data', num_workers=0):
    """获取 CIFAR-10 数据加载器"""
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
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def evaluate_all_classes(fusion_model, feature_extractors, test_loader, device):
    """评估 Fusion 模型在所有类别上的准确率"""
    fusion_model.eval()
    num_classes = 10
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            features = extract_features(feature_extractors, inputs)
            outputs = fusion_model(features)
            _, predicted = outputs.max(1)
            for c in range(num_classes):
                mask = (targets == c)
                class_total[c] += mask.sum().item()
                class_correct[c] += ((predicted == c) & mask).sum().item()

    class_acc = {}
    for c in range(num_classes):
        if class_total[c] > 0:
            class_acc[c] = 100. * class_correct[c] / class_total[c]
        else:
            class_acc[c] = 0.0
    return class_acc


def save_acc_to_csv(class_acc, args, best_acc, result_path='result/fusion_acc.csv'):
    """将 Fusion 模型的各类别准确率保存到 CSV 文件"""
    result_dir = os.path.dirname(result_path)
    if result_dir and not os.path.exists(result_dir):
        os.makedirs(result_dir)
    file_exists = os.path.isfile(result_path)

    with open(result_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ['timestamp', 'threshold', 'epochs', 'batch_size', 'lr', 'best_acc'] + [f'class_{i}' for i in range(10)] + ['mean_acc']
            writer.writerow(header)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        class_acc_list = [class_acc.get(i, 0.0) for i in range(10)]
        mean_acc = sum(class_acc_list) / len(class_acc_list)
        row = [timestamp, args.threshold, args.epochs, args.batch_size, args.lr, f'{best_acc:.2f}'] + [f'{acc:.2f}' for acc in class_acc_list] + [f'{mean_acc:.2f}']
        writer.writerow(row)
    print(f"准确率结果已保存到: {result_path}")


def main():
    """训练融合模型的主函数"""
    parser = argparse.ArgumentParser(description='Fusion Model Training')
    parser.add_argument('--threshold', type=float, required=True, help='pruning threshold')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--save-path', type=str, default='checkpoints/fusion_model.pth', help='save path')
    parser.add_argument('--data-dir', type=str, default='data', help='data directory')
    args = parser.parse_args()

    set_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用！")
    device = torch.device("cuda")
    print(f"使用设备: {device}")

    print("加载剪枝模型...")
    models, model_classes = load_pruned_models(args.threshold, device)

    print("创建 Fusion 模型...")
    feature_extractors, fusion_model, feature_dims = create_fusion_model(models, None, device)

    feature_extractors = [fe.to(device) for fe in feature_extractors]
    for fe in feature_extractors:
        fe.eval()
        for p in fe.parameters():
            p.requires_grad = False
    fusion_model = fusion_model.to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in fusion_model.parameters())
    print(f"Fusion 模型参数量: {total_params / 1e3:.2f}K")

    optimizer = optim.AdamW(fusion_model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader, test_loader = get_cifar10_loaders(args.batch_size, args.data_dir)

    best_acc = 0.0
    patience_counter = 0

    print("\n开始训练...")
    for epoch in range(args.epochs):
        fusion_model.train()
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            features = extract_features(feature_extractors, inputs)
            outputs = fusion_model(features)
            loss = F.cross_entropy(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
            optimizer.step()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})

        scheduler.step()

        fusion_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                features = extract_features(feature_extractors, inputs)
                outputs = fusion_model(features)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{args.epochs} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': fusion_model.state_dict(),
                'feature_dims': feature_dims,
                'best_acc': best_acc,
                'epoch': epoch,
            }, args.save_path)
            print(f"保存最佳模型: {args.save_path} (准确率: {best_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"早停: {args.patience} 个 epoch 没有改善")
            break

    print(f"\n训练完成! 最佳准确率: {best_acc:.2f}%")

    print("\n评估各类别准确率...")
    class_acc = evaluate_all_classes(fusion_model, feature_extractors, test_loader, device)

    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(10):
        print(f"  类别 {i} ({cifar10_classes[i]}): {class_acc[i]:.2f}%")
    mean_acc = sum(class_acc.values()) / 10
    print(f"  平均准确率: {mean_acc:.2f}%")

    save_acc_to_csv(class_acc, args, best_acc)


if __name__ == "__main__":
    main()
