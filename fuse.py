#!/usr/bin/env python3
"""
Fusion 模型训练入口
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.fusion.train_fusion import main

if __name__ == "__main__":
    main()
