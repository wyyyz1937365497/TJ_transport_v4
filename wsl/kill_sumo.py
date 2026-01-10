#!/usr/bin/env python3
"""
SUMO进程清理工具
用于清理所有SUMO相关进程和端口
"""

import subprocess
import sys
import time


def kill_sumo_processes():
    """杀死所有SUMO进程"""
    print("=" * 50)
    print("SUMO进程清理工具")
    print("=" * 50)
    print()

    # 尝试多种方式杀死SUMO进程
    commands = [
        ["killall", "-9", "sumo"],
        ["killall", "-9", "sumo-gui"],
        ["killall", "-9", "sumo.exe"],
    ]

    for cmd in commands:
        try:
            subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        except:
            pass

    # 检查残留进程
    print("检查残留进程...")
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        remaining = []
        for line in result.stdout.split('\n'):
            if 'sumo' in line.lower() and 'grep' not in line:
                remaining.append(line)

        if remaining:
            print(f"警告: 仍有 {len(remaining)} 个SUMO进程运行:")
            for line in remaining:
                print(f"  {line}")
        else:
            print("✓ 所有SUMO进程已清理")
    except:
        pass

    # 等待端口释放
    print()
    print("等待端口释放...")
    time.sleep(2)

    # 检查端口占用
    print()
    print("检查SUMO常用端口 (8813-8850)...")
    try:
        for port in range(8813, 8851):
            try:
                result = subprocess.run(
                    ["lsof", "-Pi", f":{port}", "-sTCP:LISTEN", "-t"],
                    capture_output=True,
                    text=True
                )
                if result.stdout.strip():
                    print(f"  端口 {port} 被占用: PID {result.stdout.strip()}")
            except:
                pass
        print("✓ 端口检查完成")
    except:
        print("  (跳过端口检查 - lsof不可用)")

    print()
    print("=" * 50)
    print("清理完成!")
    print("=" * 50)


if __name__ == "__main__":
    kill_sumo_processes()
