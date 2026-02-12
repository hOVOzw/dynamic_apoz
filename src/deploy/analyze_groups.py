#!/usr/bin/env python3
"""
设备分组分析脚本
功能：
1. 加载 base 模型（完整 ResNet）
2. CIFAR-10 测试集推理，获取每个类别的输出
3. 计算类间协方差
4. 根据协方差聚类分组（遍历 2-9）
5. 保存结果到 result/deploy_inf.csv

用于 prune 之前决定每个剪枝模型负责哪几类
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from datetime import datetime
import csv

# 导入模型定义
from src.models.resnet import resnet18


# 路径配置
BASE_MODEL_PATH = "src/training/checkpoints/best_model.pth"
RESULT_FILE = "result/deploy_inf.csv"
DEVICE_NUMS = list(range(2, 10))  # 遍历 2-9
NUM_SAMPLES = 1000  # 采样数量


def load_base_model(device):
    """加载 base 模型（完整 ResNet）"""
    print(f"加载 base 模型: {BASE_MODEL_PATH}")
    checkpoint = torch.load(BASE_MODEL_PATH, map_location=device, weights_only=False)
    
    model = resnet18(num_classes=10)
    
    # 兼容不同检查点格式
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print("Base 模型加载完成")
    return model


def get_class_outputs(model, device, num_samples=1000):
    """获取每个类别的模型输出，用于计算类间协方差"""
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = datasets.CIFAR10(root="data", train=False, transform=transform, download=True)
    
    # 收集每个类别的输出
    class_outputs = [[] for _ in range(10)]  # 10个类别
    
    print(f"\n推理测试集，获取每个类别的输出...")
    for i in range(num_samples):
        img, label = testset[i]
        img = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img)  # (1, 10)
        
        class_outputs[label].append(output.squeeze().cpu().numpy())
        
        if (i + 1) % 500 == 0:
            print(f"  已处理 {i + 1}/{num_samples} 样本")
    
    # 计算每个类别的平均输出向量
    class_mean_outputs = []
    for c in range(10):
        outputs = np.array(class_outputs[c])  # (N_c, 10)
        mean_output = outputs.mean(axis=0)   # (10,)
        class_mean_outputs.append(mean_output)
        print(f"  类别 {c}: {len(class_outputs[c])} 样本")
    
    return class_outputs, class_mean_outputs


def calculate_class_covariance(class_outputs):
    """计算类别间的协方差矩阵"""
    print("\n计算类别间协方差矩阵...")
    
    # 每个类别的输出堆叠
    class_data = []
    for c in range(10):
        outputs = np.array(class_outputs[c])  # (N_c, 10)
        class_data.append(outputs)
    
    # 计算每个类别的均值
    class_means = [np.mean(data, axis=0) for data in class_data]  # (10,) x 10
    class_means = np.array(class_means)  # (10, 10)
    
    # 计算协方差矩阵（10x10，类别间相关性）
    cov_matrix = np.cov(class_means, rowvar=False)  # (10, 10)
    
    print(f"协方差矩阵 shape: {cov_matrix.shape}")
    return cov_matrix, class_means


def cluster_classes(class_means, device_num):
    """根据类别输出进行聚类分组"""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 使用余弦相似度
    similarity_matrix = cosine_similarity(class_means)  # (10, 10)
    distance_matrix = 1 - similarity_matrix
    
    # 确保对角线为0（解决浮点数精度问题）
    np.fill_diagonal(distance_matrix, 0)
    
    # 层次聚类
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')
    
    # 分组
    labels = fcluster(linkage_matrix, device_num, criterion='maxclust')
    
    # 整理分组结果
    groups = [[] for _ in range(device_num)]
    for i, label in enumerate(labels):
        groups[label - 1].append(i)  # 类别索引 0-9
    
    return groups, similarity_matrix


def save_results(all_groups, similarity_matrix):
    """保存所有分组结果到 CSV"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    
    # 清空旧文件，重新写入
    with open(RESULT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # 写入表头（threshold 列对分组分析无意义，已删除）
        max_groups = max(DEVICE_NUMS)
        header = ['timestamp', 'device_num'] + [f'group_{i}' for i in range(max_groups)]
        writer.writerow(header)
        
        # 写入每一行
        for device_num in DEVICE_NUMS:
            groups = all_groups[device_num]
            group_strs = [str(g) for g in groups]
            # 补齐空列
            while len(group_strs) < max_groups:
                group_strs.append('')
            row = [timestamp, device_num] + group_strs
            writer.writerow(row)
    
    print(f"\n结果已保存到: {RESULT_FILE}")
    
    # 保存相似度矩阵
    np.save("result/deploy_similarity_matrix.npy", similarity_matrix)
    print("相似度矩阵已保存到: result/deploy_similarity_matrix.npy")


def main():
    """主函数"""
    # 自动检测 CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")
    print(f"分组数量: 遍历 {DEVICE_NUMS[0]}-{DEVICE_NUMS[-1]}")
    print(f"采样数量: {NUM_SAMPLES}")
    
    # 1. 加载 base 模型
    print("\n=== 加载 Base 模型 ===")
    model = load_base_model(device)
    
    # 2. 获取每个类别的输出
    print("\n=== 获取类别输出 ===")
    class_outputs, class_means = get_class_outputs(model, device, NUM_SAMPLES)
    
    # 3. 计算协方差矩阵
    print("\n=== 计算协方差矩阵 ===")
    cov_matrix, class_means = calculate_class_covariance(class_outputs)
    
    # 4. 打印相似度矩阵
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(class_means)
    print("\n类别间相似度矩阵:")
    print(similarity_matrix)
    
    # 5. 聚类分组（遍历不同 device_num）
    print("\n=== 聚类分组 ===")
    all_results = {}
    for device_num in DEVICE_NUMS:
        groups, _ = cluster_classes(class_means, device_num)
        all_results[device_num] = groups
        print(f"Device {device_num}: {groups}")
    
    # 6. 保存结果
    print("\n=== 保存结果 ===")
    save_results(all_results, similarity_matrix)
    
    print("\n=== 完成 ===")
    return all_results


if __name__ == "__main__":
    main()
