#!/usr/bin/env python3
"""
设备分组分析入口
根据剪枝模型输出的协方差进行聚类分组
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.deploy.analyze_groups import main

if __name__ == "__main__":
    main()
