#!/usr/bin/env python3
"""
推理测试入口
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.infer.infer_core import main

if __name__ == "__main__":
    main()
