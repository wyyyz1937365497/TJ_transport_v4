#!/usr/bin/env python3
"""
SUMO进程清理工具 - 主目录版本
"""

import subprocess
import sys
import time


def kill_sumo():
    """杀死所有SUMO进程"""
    print("=" * 50)
    print("清理SUMO进程...")
    print("=" * 50)

    # 杀死SUMO进程
    subprocess.run(["killall", "-9", "sumo"], stderr=subprocess.DEVNULL)
    subprocess.run(["killall", "-9", "sumo-gui"], stderr=subprocess.DEVNULL)
    subprocess.run(["killall", "-9", "sumo.exe"], stderr=subprocess.DEVNULL)

    # 等待
    time.sleep(2)

    print("✅ 清理完成！")
    print()
    print("现在可以重新运行训练:")
    print("  python train.py")


if __name__ == "__main__":
    kill_sumo()
