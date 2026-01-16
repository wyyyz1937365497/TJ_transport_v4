#!/usr/bin/env python3
"""
快速清理脚本 - 无需确认，直接清理

功能：快速清理Python缓存文件（__pycache__, .pyc）

使用方法：
    python clean_quick.py
"""

import os
import shutil
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent


def clean_pycache():
    """清理Python缓存"""
    print("[*] Cleaning Python cache files...\n")

    # 删除__pycache__目录
    pycache_dirs = list(PROJECT_ROOT.rglob('__pycache__'))
    removed_count = 0
    total_size = 0

    for dir_path in pycache_dirs:
        try:
            # 计算大小
            size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())

            # 删除目录
            shutil.rmtree(dir_path)

            total_size += size
            removed_count += 1

            # 格式化大小
            size_mb = size / (1024 * 1024)
            print(f"  [+] Deleted: {dir_path.relative_to(PROJECT_ROOT)} ({size_mb:.2f} MB)")

        except Exception as e:
            print(f"  [-] Failed: {dir_path.relative_to(PROJECT_ROOT)}: {e}")

    # 删除.pyc文件
    pyc_files = list(PROJECT_ROOT.rglob("*.pyc"))
    pyc_count = 0
    pyc_size = 0

    for file in pyc_files:
        try:
            size = file.stat().st_size
            file.unlink()
            pyc_size += size
            pyc_count += 1
        except Exception:
            pass

    # 打印总结
    total_size_mb = total_size / (1024 * 1024)
    pyc_size_mb = pyc_size / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"清理完成！")
    print(f"  __pycache__ 目录: {removed_count} 个 ({total_size_mb:.2f} MB)")
    print(f"  .pyc 文件: {pyc_count} 个 ({pyc_size_mb:.2f} MB)")
    print(f"  总计释放空间: {total_size_mb + pyc_size_mb:.2f} MB")
    print(f"{'='*60}\n")

    return removed_count + pyc_count > 0


def main():
    """主函数"""
    print("\n" + "="*60)
    print(" " * 20 + "Quick Clean Tool")
    print(" " * 18 + "Clean Python Cache")
    print("="*60 + "\n")

    print(f"Project dir: {PROJECT_ROOT}\n")

    # 执行清理
    success = clean_pycache()

    if success:
        print("[+] Project cleaned! Ready to start training.\n")
    else:
        print("[i] No files found to clean.\n")


if __name__ == "__main__":
    main()
